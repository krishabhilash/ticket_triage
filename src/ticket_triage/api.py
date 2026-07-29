"""Thin FastAPI transport for the fitted classical ticket classifier."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ticket_triage.predictor import Prediction, TicketPredictor, load_model

DEFAULT_MODEL_PATH = "artifacts/model.joblib"
MAX_MESSAGE_LENGTH = 4_000
MAX_BATCH_SIZE = 100
SERIALIZED_CONFIDENCE_DIGITS = 6

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
        list[StrictMessage],
        Field(min_length=1, max_length=MAX_BATCH_SIZE),
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

    label: Literal[
        "account-access",
        "transaction-dispute",
        "fraud-report",
        "general",
    ]
    confidence: float


class BatchPredictionResponse(BaseModel):
    """Ordered predictions for a batch request."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Process health and model-loading state."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool


class PredictionService:
    """Serialize concurrent access to one immutable fitted predictor."""

    def __init__(self, predictor: TicketPredictor) -> None:
        self._predictor = predictor
        self._lock = Lock()

    def predict(self, text: str) -> Prediction:
        """Run one read-only prediction under a short critical section."""
        with self._lock:
            return self._predictor.predict_with_confidence(text)

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        """Run an ordered batch without changing the fitted pipeline."""
        with self._lock:
            return [self._predictor.predict_with_confidence(text) for text in texts]


def _serialize(prediction: Prediction) -> PredictionResponse:
    """Round confidence only at the HTTP serialization boundary."""
    return PredictionResponse(
        label=prediction.label,  # type: ignore[arg-type]
        confidence=round(prediction.confidence, SERIALIZED_CONFIDENCE_DIGITS),
    )


def create_app() -> FastAPI:
    """Create an API whose process loads one configured model at startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.prediction_service = None
        model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
        try:
            application.state.prediction_service = PredictionService(
                load_model(model_path)
            )
        except Exception as error:
            logger.error("Model initialization failed (%s)", type(error).__name__)
        yield
        application.state.prediction_service = None

    application = FastAPI(
        title="Ticket Triage API",
        version="0.1.0",
        lifespan=lifespan,
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
        """Report process health and whether the model loaded."""
        loaded = getattr(request.app.state, "prediction_service", None) is not None
        return HealthResponse(
            status="healthy" if loaded else "unhealthy", model_loaded=loaded
        )

    @application.get("/ready", response_model=HealthResponse)
    def ready(request: Request) -> HealthResponse:
        """Return 200 only when the fitted model is ready for inference."""
        service(request)
        return HealthResponse(status="healthy", model_loaded=True)

    @application.post("/predict", response_model=PredictionResponse)
    def predict_one(payload: PredictRequest, request: Request) -> PredictionResponse:
        """Classify one validated ticket."""
        return _serialize(service(request).predict(payload.text))

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        payload: BatchPredictRequest, request: Request
    ) -> BatchPredictionResponse:
        """Classify a bounded batch while preserving request order."""
        predictions = service(request).predict_batch(payload.texts)
        return BatchPredictionResponse(
            predictions=[_serialize(prediction) for prediction in predictions]
        )

    return application


app = create_app()
