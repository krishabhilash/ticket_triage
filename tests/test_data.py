"""Tests for training-data validation."""

from pathlib import Path

import pandas as pd
import pytest

from ticket_triage.data import load_dataset, validate_dataset


def test_loading_rejects_missing_text_column(tmp_path: Path) -> None:
    source = tmp_path / "missing-text.csv"
    pd.DataFrame({"message": ["Help"], "label": ["general"]}).to_csv(
        source, index=False
    )

    with pytest.raises(ValueError, match=r"missing required column\(s\): text"):
        load_dataset(source)


def test_validation_rejects_empty_messages() -> None:
    frame = pd.DataFrame({"text": ["  "], "label": ["general"]})
    with pytest.raises(ValueError, match="empty messages"):
        validate_dataset(frame)


def test_loading_rejects_unknown_label(tmp_path: Path) -> None:
    source = tmp_path / "unknown-label.csv"
    pd.DataFrame({"text": ["Help"], "label": ["other"]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="unknown labels"):
        load_dataset(source)
