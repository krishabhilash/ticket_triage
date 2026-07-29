"""Classical support-ticket classification package."""

from ticket_triage.constants import ALLOWED_LABELS
from ticket_triage.predictor import Prediction, TicketPredictor, load_model

__all__ = ["ALLOWED_LABELS", "Prediction", "TicketPredictor", "load_model"]
