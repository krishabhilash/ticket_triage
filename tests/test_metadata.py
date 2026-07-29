"""Tests for safe model metadata sidecars."""

from pathlib import Path

import pandas as pd

from ticket_triage.evaluation import evaluate_predictions
from ticket_triage.metadata import (
    build_model_metadata,
    load_model_metadata,
    save_model_metadata,
)
from ticket_triage.model import build_pipeline, fit_pipeline, save_pipeline


def test_metadata_is_written_and_loaded_without_messages(tmp_path: Path) -> None:
    secret_message = "private customer message that must not be metadata"
    frame = pd.DataFrame(
        {
            "text": [
                secret_message,
                "delayed transfer",
                "stolen funds",
                "help article",
            ],
            "label": [
                "account-access",
                "transaction-dispute",
                "fraud-report",
                "general",
            ],
        }
    )
    pipeline = fit_pipeline(build_pipeline(class_weight="balanced"), frame)
    validation = evaluate_predictions(frame["label"].tolist(), frame["label"].tolist())
    model_path = save_pipeline(pipeline, tmp_path / "model.joblib")

    metadata = build_model_metadata(frame, pipeline, "balanced", validation)
    sidecar = save_model_metadata(model_path, metadata)
    loaded = load_model_metadata(model_path)

    assert sidecar.exists()
    assert loaded == metadata
    assert loaded is not None
    assert loaded["dataset_row_count"] == 4
    assert loaded["class_distribution"]["fraud-report"] == 1
    assert loaded["labels"] == [
        "account-access",
        "transaction-dispute",
        "fraud-report",
        "general",
    ]
    assert secret_message not in sidecar.read_text(encoding="utf-8")
