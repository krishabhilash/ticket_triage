"""Tests for model construction."""

import pytest

from ticket_triage.model import build_pipeline


def test_pipeline_contains_expected_configuration() -> None:
    pipeline = build_pipeline(class_weight="balanced")
    assert pipeline["tfidf"].ngram_range == (1, 2)
    assert pipeline["classifier"].class_weight == "balanced"


def test_pipeline_rejects_invalid_class_weight() -> None:
    with pytest.raises(ValueError, match="class_weight"):
        build_pipeline(class_weight="invalid")  # type: ignore[arg-type]

