"""Small structured-logging and in-process instrumentation primitives."""

import json
import logging
from collections import Counter
from datetime import UTC, datetime
from threading import Lock
from typing import Any


class JsonFormatter(logging.Formatter):
    """Serialize allow-listed log context as one JSON object per event."""

    _context_fields = (
        "request_id",
        "route",
        "status",
        "latency_ms",
        "model_version",
        "error_type",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record without request bodies or arbitrary attributes."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for field in self._context_fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, separators=(",", ":"))


def configure_structured_logging(logger: logging.Logger) -> None:
    """Attach one JSON handler to the API logger."""
    if any(
        getattr(handler, "ticket_triage_json", False) for handler in logger.handlers
    ):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.ticket_triage_json = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class OperationalMetrics:
    """Thread-safe counters that can later be exported to a metrics backend."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.request_count = 0
        self.request_latency_seconds = 0.0
        self.predicted_labels: Counter[str] = Counter()
        self.validation_failures = 0
        self.low_confidence_predictions = 0
        self.model_loading_failures = 0

    def record_request(self, latency_seconds: float, status_code: int) -> None:
        """Record one completed HTTP request."""
        with self._lock:
            self.request_count += 1
            self.request_latency_seconds += latency_seconds
            if status_code == 422:
                self.validation_failures += 1

    def record_prediction(self, label: str, low_confidence: bool) -> None:
        """Record the predicted route and review-policy outcome."""
        with self._lock:
            self.predicted_labels[label] += 1
            if low_confidence:
                self.low_confidence_predictions += 1

    def record_model_loading_failure(self) -> None:
        """Record failure to initialize a model artifact."""
        with self._lock:
            self.model_loading_failures += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a copy suitable for a future Prometheus adapter."""
        with self._lock:
            return {
                "request_count": self.request_count,
                "request_latency_seconds": self.request_latency_seconds,
                "predicted_labels": dict(self.predicted_labels),
                "validation_failures": self.validation_failures,
                "low_confidence_predictions": self.low_confidence_predictions,
                "model_loading_failures": self.model_loading_failures,
            }
