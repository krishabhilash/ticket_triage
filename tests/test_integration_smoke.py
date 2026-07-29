"""Lightweight end-to-end smoke test for the classical local workflow."""

from pathlib import Path

import pandas as pd

from ticket_triage.constants import ALLOWED_LABELS
from ticket_triage.data import load_dataset
from ticket_triage.model import build_pipeline, fit_pipeline, save_pipeline
from ticket_triage.predictor import load_model
from ticket_triage.score import score_csv


def test_train_reload_and_score_smoke(tmp_path: Path) -> None:
    """Train on temporary data, reload, score, and verify written predictions."""
    training_path = tmp_path / "training.csv"
    model_path = tmp_path / "model.joblib"
    holdout_path = tmp_path / "holdout.csv"
    output_path = tmp_path / "results" / "predictions.csv"

    pd.DataFrame(
        {
            "text": [
                "I cannot log in",
                "My password is blocked",
                "My transfer is delayed",
                "I was charged twice",
                "Someone stole my funds",
                "I did not authorize this withdrawal",
                "How does staking work",
                "Which assets are supported",
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
    ).to_csv(training_path, index=False)
    training_frame = load_dataset(training_path)
    pipeline = fit_pipeline(build_pipeline(class_weight="balanced"), training_frame)
    save_pipeline(pipeline, model_path)

    predictor = load_model(model_path)
    assert predictor.predict("Help me access my account") in ALLOWED_LABELS

    pd.DataFrame(
        {"ticket_id": [2, 1], "text": ["My login failed", "Explain staking"]}
    ).to_csv(holdout_path, index=False)
    score_csv(
        model_path=model_path,
        input_path=holdout_path,
        output_path=output_path,
    )
    scored = pd.read_csv(output_path)

    assert scored["ticket_id"].tolist() == [2, 1]
    assert {"prediction", "confidence"}.issubset(scored.columns)
    assert set(scored["prediction"]).issubset(ALLOWED_LABELS)
