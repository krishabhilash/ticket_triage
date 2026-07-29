"""Tests for single-message prediction and holdout scoring."""

from pathlib import Path

import pandas as pd
import pytest

from ticket_triage.constants import ALLOWED_LABELS
from ticket_triage.model import build_pipeline, fit_pipeline, save_pipeline
from ticket_triage.predictor import load_model
from ticket_triage.score import score_csv


@pytest.fixture()
def fitted_model(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "text": [
                "I cannot log into my account",
                "My password reset does not work",
                "I dispute this duplicate transaction",
                "My refund has not arrived",
                "Someone stole funds from my account",
                "This withdrawal was not authorized",
                "How does staking work",
                "Which assets do you support",
            ],
            "label": [
                "account-access",
                "account-access",
                "transaction-dispute",
                "transaction-dispute",
                "fraud-report",
                "fraud-report",
                "general",
                "general",
            ],
        }
    )
    pipeline = fit_pipeline(build_pipeline(class_weight="balanced"), frame)
    return save_pipeline(pipeline, tmp_path / "model.joblib")


def test_predictor_returns_approved_label_and_confidence(fitted_model: Path) -> None:
    predictor = load_model(fitted_model)
    result = predictor.predict_with_confidence("Someone accessed my account")
    assert predictor.predict("Someone accessed my account") in ALLOWED_LABELS
    assert result.label in ALLOWED_LABELS
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize(
    ("message", "error_type"),
    [(None, ValueError), (123, TypeError), ("", ValueError), ("  ", ValueError)],
)
def test_predictor_rejects_invalid_message(
    fitted_model: Path, message: object, error_type: type[Exception]
) -> None:
    predictor = load_model(fitted_model)
    with pytest.raises(error_type):
        predictor.predict(message)


def test_score_csv_preserves_rows_columns_and_order(
    fitted_model: Path, tmp_path: Path
) -> None:
    source = tmp_path / "holdout.csv"
    destination = tmp_path / "nested" / "predictions.csv"
    original = pd.DataFrame(
        {
            "ticket_id": [30, 10, 20],
            "message": [
                "My login is blocked",
                "I did not authorize this withdrawal",
                "What are the trading fees",
            ],
        }
    )
    original.to_csv(source, index=False)

    score_csv(
        model_path=fitted_model,
        input_path=source,
        output_path=destination,
        text_column="message",
    )
    scored = pd.read_csv(destination)

    assert scored["ticket_id"].tolist() == [30, 10, 20]
    assert scored["message"].tolist() == original["message"].tolist()
    assert set(scored["prediction"]).issubset(ALLOWED_LABELS)
    assert scored["confidence"].between(0.0, 1.0).all()


def test_score_csv_rejects_invalid_row(fitted_model: Path, tmp_path: Path) -> None:
    source = tmp_path / "invalid.csv"
    pd.DataFrame({"text": ["valid", None]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="CSV row 3"):
        score_csv(
            model_path=fitted_model,
            input_path=source,
            output_path=tmp_path / "output.csv",
        )
