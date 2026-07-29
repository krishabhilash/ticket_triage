"""Thin FastAPI transport for the fitted classical ticket classifier."""

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ticket_triage.metadata import load_model_metadata
from ticket_triage.observability import OperationalMetrics, configure_structured_logging
from ticket_triage.predictor import Prediction, TicketPredictor, load_model

DEFAULT_MODEL_PATH = "artifacts/model.joblib"
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.60
MAX_MESSAGE_LENGTH = 4_000
MAX_BATCH_SIZE = 100
SERIALIZED_CONFIDENCE_DIGITS = 6
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

logger = logging.getLogger(__name__)

StrictMessage = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=MAX_MESSAGE_LENGTH),
]


class PredictRequest(BaseModel):
    """One ticket submitted for classification."""

    model_config = ConfigDict(extra="forbid")
    text: StrictMessage

    @field_validator("text")
    @classmethod
    def reject_whitespace(cls, text: str) -> str:
        """Reject messages that contain no visible characters."""
        if not text.strip():
            raise ValueError("Message must not be empty")
        return text


class BatchPredictRequest(BaseModel):
    """A bounded collection of tickets submitted for classification."""

    model_config = ConfigDict(extra="forbid")
    texts: Annotated[
        list[StrictMessage], Field(min_length=1, max_length=MAX_BATCH_SIZE)
    ]

    @field_validator("texts")
    @classmethod
    def reject_whitespace(cls, texts: list[str]) -> list[str]:
        """Reject the complete batch when any message is blank."""
        if any(not text.strip() for text in texts):
            raise ValueError("Messages must not be empty")
        return texts


class PredictionResponse(BaseModel):
    """Serialized prediction returned by the HTTP API."""

    label: Literal["account-access", "transaction-dispute", "fraud-report", "general"]
    confidence: float
    requires_review: bool


class BatchPredictionResponse(BaseModel):
    """Ordered predictions for a batch request."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Process liveness and model-loading state."""

    status: Literal["healthy"]
    model_loaded: bool


class PredictionService:
    """Apply operational policy around one immutable fitted predictor."""

    def __init__(
        self,
        predictor: TicketPredictor,
        low_confidence_threshold: float,
        metrics: OperationalMetrics,
    ) -> None:
        self._predictor = predictor
        self._threshold = low_confidence_threshold
        self._metrics = metrics
        self._lock = Lock()

    def predict(self, text: str) -> PredictionResponse:
        """Run and serialize one read-only prediction."""
        with self._lock:
            prediction = self._predictor.predict_with_confidence(text)
        return self._apply_policy(prediction)

    def predict_batch(self, texts: list[str]) -> list[PredictionResponse]:
        """Run an ordered batch without changing the fitted pipeline."""
        with self._lock:
            predictions = [
                self._predictor.predict_with_confidence(text) for text in texts
            ]
        return [self._apply_policy(prediction) for prediction in predictions]

    def _apply_policy(self, prediction: Prediction) -> PredictionResponse:
        """Mark low confidence for review without changing the predicted label."""
        requires_review = prediction.confidence < self._threshold
        self._metrics.record_prediction(prediction.label, requires_review)
        return PredictionResponse(
            label=prediction.label,  # type: ignore[arg-type]
            confidence=round(prediction.confidence, SERIALIZED_CONFIDENCE_DIGITS),
            requires_review=requires_review,
        )


def _confidence_threshold() -> float:
    """Read and validate the operational confidence threshold."""
    raw_value = os.environ.get(
        "LOW_CONFIDENCE_THRESHOLD", str(DEFAULT_LOW_CONFIDENCE_THRESHOLD)
    )
    threshold = float(raw_value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("LOW_CONFIDENCE_THRESHOLD must be between 0 and 1")
    return threshold


def _request_id(request: Request) -> str:
    """Reuse a safe caller request ID or generate one."""
    supplied = request.headers.get("X-Request-ID", "")
    return supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid4())


def create_app() -> FastAPI:
    """Create an API whose process loads one configured model at startup."""
    configure_structured_logging(logger)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.prediction_service = None
        application.state.metrics = OperationalMetrics()
        application.state.model_metadata = None
        application.state.model_version = "unknown"
        model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
        try:
            metadata = load_model_metadata(model_path)
            application.state.prediction_service = PredictionService(
                load_model(model_path),
                _confidence_threshold(),
                application.state.metrics,
            )
            application.state.model_metadata = metadata
            if metadata:
                application.state.model_version = metadata.get(
                    "package_version", "unknown"
                )
            logger.info(
                "model_loaded",
                extra={"model_version": application.state.model_version},
            )
        except Exception as error:
            application.state.metrics.record_model_loading_failure()
            logger.error(
                "model_initialization_failed",
                extra={"error_type": type(error).__name__},
            )
        yield
        application.state.prediction_service = None
        logger.info(
            "service_shutdown",
            extra={"model_version": application.state.model_version},
        )

    application = FastAPI(title="Ticket Triage API", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def request_observability(request: Request, call_next):
        request_id = _request_id(request)
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = perf_counter() - started
            metrics = getattr(request.app.state, "metrics", None)
            if metrics is not None:
                metrics.record_request(elapsed, status_code)
            route = request.scope.get("route")
            logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "route": getattr(route, "path", "unmatched"),
                    "status": status_code,
                    "latency_ms": round(elapsed * 1000, 3),
                    "model_version": getattr(
                        request.app.state, "model_version", "unknown"
                    ),
                },
            )

    def service(request: Request) -> PredictionService:
        loaded_service = getattr(request.app.state, "prediction_service", None)
        if loaded_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready",
            )
        return loaded_service

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Report process liveness independently from readiness."""
        loaded = getattr(request.app.state, "prediction_service", None) is not None
        return HealthResponse(status="healthy", model_loaded=loaded)

    @application.get("/ready", response_model=HealthResponse)
    def ready(request: Request) -> HealthResponse:
        """Return 200 only when the fitted model is ready for inference."""
        service(request)
        return HealthResponse(status="healthy", model_loaded=True)

    @application.post("/predict", response_model=PredictionResponse)
    def predict_one(payload: PredictRequest, request: Request) -> PredictionResponse:
        """Classify one validated ticket."""
        return service(request).predict(payload.text)

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        payload: BatchPredictRequest, request: Request
    ) -> BatchPredictionResponse:
        """Classify a bounded batch while preserving request order."""
        return BatchPredictionResponse(
            predictions=service(request).predict_batch(payload.texts)
        )

    return application


app = create_app()
