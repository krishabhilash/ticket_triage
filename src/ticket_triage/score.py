"""Score an unseen CSV with a previously fitted local model."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ticket_triage.constants import ALLOWED_LABELS, TEXT_COLUMN
from ticket_triage.predictor import load_model, validate_message


def score_csv(
    *,
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    text_column: str = TEXT_COLUMN,
) -> Path:
    """Preserve input rows and append local model predictions and confidence."""
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")

    frame = pd.read_csv(source)
    if text_column not in frame.columns:
        raise ValueError(f"Input CSV is missing text column: {text_column}")

    messages: list[str] = []
    for row_number, value in enumerate(frame[text_column], start=2):
        try:
            messages.append(validate_message(value))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid message at CSV row {row_number}: {error}") from error

    predictor = load_model(model_path)
    predictions = [predictor.predict_with_confidence(message) for message in messages]
    labels = [prediction.label for prediction in predictions]
    unknown_labels = set(labels).difference(ALLOWED_LABELS)
    if unknown_labels:
        raise ValueError(f"Model produced unknown labels: {sorted(unknown_labels)}")

    output_frame = frame.copy()
    output_frame["prediction"] = labels
    output_frame["confidence"] = [
        prediction.confidence for prediction in predictions
    ]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(destination, index=False)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse holdout-scoring command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score a CSV with a fitted ticket classifier"
    )
    parser.add_argument("--model", required=True, help="Path to fitted joblib model")
    parser.add_argument("--input", required=True, help="Path to unseen input CSV")
    parser.add_argument("--output", required=True, help="Destination predictions CSV")
    parser.add_argument(
        "--text-column",
        default=TEXT_COLUMN,
        help=f"Input message column (default: {TEXT_COLUMN})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run holdout scoring without fitting or external data transfer."""
    args = parse_args(argv)
    destination = score_csv(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
    )
    print(f"Wrote predictions to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
