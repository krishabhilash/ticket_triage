"""Shared constants for ticket classification."""

ALLOWED_LABELS: tuple[str, ...] = (
    "account-access",
    "transaction-dispute",
    "fraud-report",
    "general",
)
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
FRAUD_LABEL = "fraud-report"
RANDOM_STATE = 42

