"""Dataset loading, validation, and splitting."""

from pathlib import Path
from typing import TypeAlias

import pandas as pd
from sklearn.model_selection import train_test_split

from ticket_triage.constants import ALLOWED_LABELS, LABEL_COLUMN, RANDOM_STATE, TEXT_COLUMN

PathLike: TypeAlias = str | Path


def load_dataset(path: PathLike) -> pd.DataFrame:
    """Load a CSV dataset and validate its required training fields."""
    frame = pd.read_csv(path)
    validate_dataset(frame)
    return frame[[TEXT_COLUMN, LABEL_COLUMN]].copy()


def validate_dataset(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` when a training frame is malformed."""
    missing_columns = {TEXT_COLUMN, LABEL_COLUMN}.difference(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {names}")
    if frame[TEXT_COLUMN].isna().any():
        raise ValueError("Dataset contains missing messages")
    if frame[LABEL_COLUMN].isna().any():
        raise ValueError("Dataset contains missing labels")
    messages = frame[TEXT_COLUMN]
    if not messages.map(lambda value: isinstance(value, str)).all():
        raise ValueError("All messages must be strings")
    if messages.str.strip().eq("").any():
        raise ValueError("Dataset contains empty messages")
    unknown_labels = sorted(set(frame[LABEL_COLUMN]).difference(ALLOWED_LABELS))
    if unknown_labels:
        raise ValueError(f"Dataset contains unknown labels: {unknown_labels}")


def stratified_split(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible label-stratified train/validation split."""
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=validation_size,
        random_state=random_state,
        stratify=frame[LABEL_COLUMN],
    )
    return train_frame.reset_index(drop=True), validation_frame.reset_index(drop=True)

