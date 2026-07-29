"""Model metadata sidecar construction and persistence."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from ticket_triage.constants import ALLOWED_LABELS, LABEL_COLUMN
from ticket_triage.evaluation import EvaluationResult


def metadata_path(model_path: str | Path) -> Path:
    """Return the JSON sidecar path for a persisted model."""
    path = Path(model_path)
    return path.with_name(f"{path.name}.metadata.json")


def _package_version() -> str:
    """Return the installed package version when distribution metadata exists."""
    try:
        return version("ticket-triage")
    except PackageNotFoundError:
        return "unknown"


def build_model_metadata(
    frame: pd.DataFrame,
    pipeline: Pipeline,
    class_weight: str | None,
    validation: EvaluationResult,
) -> dict[str, Any]:
    """Describe training inputs and configuration without storing ticket text."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    distribution = (
        frame[LABEL_COLUMN].value_counts().reindex(ALLOWED_LABELS, fill_value=0)
    )
    return {
        "training_timestamp": datetime.now(UTC).isoformat(),
        "dataset_row_count": len(frame),
        "class_distribution": {
            label: int(distribution[label]) for label in ALLOWED_LABELS
        },
        "model_type": type(classifier).__name__,
        "tfidf_configuration": {
            "analyzer": vectorizer.analyzer,
            "lowercase": vectorizer.lowercase,
            "ngram_range": list(vectorizer.ngram_range),
            "sublinear_tf": vectorizer.sublinear_tf,
        },
        "class_weight": class_weight,
        "labels": list(ALLOWED_LABELS),
        "validation_metrics": asdict(validation),
        "package_version": _package_version(),
    }


def save_model_metadata(model_path: str | Path, metadata: dict[str, Any]) -> Path:
    """Write a JSON metadata sidecar next to a model artifact."""
    path = metadata_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_model_metadata(model_path: str | Path) -> dict[str, Any] | None:
    """Load an optional model metadata sidecar."""
    path = metadata_path(model_path)
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Model metadata must be a JSON object")
    return loaded
