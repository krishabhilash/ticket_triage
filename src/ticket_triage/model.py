"""Model construction, fitting, prediction, and persistence."""

from pathlib import Path
from typing import Literal, TypeAlias

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ticket_triage.constants import LABEL_COLUMN, RANDOM_STATE, TEXT_COLUMN

ClassWeight: TypeAlias = Literal["balanced"] | None


def build_pipeline(*, class_weight: ClassWeight = None) -> Pipeline:
    """Build an unfitted word TF-IDF and logistic-regression pipeline."""
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be None or 'balanced'")
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), lowercase=True, sublinear_tf=True
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=1_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_pipeline(pipeline: Pipeline, train_frame: pd.DataFrame) -> Pipeline:
    """Fit the complete pipeline using only the supplied training rows."""
    pipeline.fit(train_frame[TEXT_COLUMN], train_frame[LABEL_COLUMN])
    return pipeline


def predict(pipeline: Pipeline, messages: pd.Series) -> list[str]:
    """Predict one route label for each message."""
    return pipeline.predict(messages).tolist()


def save_pipeline(pipeline: Pipeline, output_path: str | Path) -> Path:
    """Persist a fitted pipeline, creating its parent directory if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path

