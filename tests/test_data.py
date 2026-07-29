"""Tests for training-data validation."""

import pandas as pd
import pytest

from ticket_triage.data import validate_dataset


def test_validation_rejects_empty_messages() -> None:
    frame = pd.DataFrame({"text": ["  "], "label": ["general"]})
    with pytest.raises(ValueError, match="empty messages"):
        validate_dataset(frame)


def test_validation_rejects_unknown_labels() -> None:
    frame = pd.DataFrame({"text": ["Help"], "label": ["other"]})
    with pytest.raises(ValueError, match="unknown labels"):
        validate_dataset(frame)

