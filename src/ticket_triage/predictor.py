"""Production-facing inference over a previously fitted local pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ticket_triage.constants import ALLOWED_LABELS


@dataclass(frozen=True)
class Prediction:
    """A predicted route and its uncalibrated maximum class probability."""

    label: str
    confidence: float


def validate_message(text: object) -> str:
    """Validate and return one inference message without changing its content."""
    if text is None:
        raise ValueError("Message must not be None")
    if not isinstance(text, str):
        raise TypeError("Message must be a string")
    if not text.strip():
        raise ValueError("Message must not be empty")
    return text


class TicketPredictor:
    """Validated prediction interface around a fitted scikit-learn pipeline."""

    def __init__(self, pipeline: Any) -> None:
        """Validate and retain a fitted probabilistic classifier."""
        if not callable(getattr(pipeline, "predict", None)):
            raise ValueError("Loaded model does not provide predict()")
        if not callable(getattr(pipeline, "predict_proba", None)):
            raise ValueError("Loaded model does not provide predict_proba()")
        try:
            model_labels = set(pipeline.classes_)
        except AttributeError as error:
            raise ValueError("Loaded model is not fitted or has no classes_") from error
        if model_labels != set(ALLOWED_LABELS):
            raise ValueError(
                "Loaded model classes do not match the approved ticket labels"
            )
        self._pipeline = pipeline

    def predict(self, text: object) -> str:
        """Return the approved route label for one valid message."""
        prediction = self.predict_with_confidence(text)
        return prediction.label

    def predict_with_confidence(self, text: object) -> Prediction:
        """Return a label and uncalibrated maximum class probability.

        Logistic-regression probabilities are useful relative confidence
        signals but are not claimed to be calibrated probabilities of
        correctness.
        """
        message = validate_message(text)
        label = str(self._pipeline.predict([message])[0])
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Model returned an unknown label: {label}")
        probabilities = self._pipeline.predict_proba([message])[0]
        return Prediction(label=label, confidence=float(max(probabilities)))


def load_model(path: str | Path) -> TicketPredictor:
    """Load a fitted local joblib model and return its inference interface."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    return TicketPredictor(joblib.load(model_path))
