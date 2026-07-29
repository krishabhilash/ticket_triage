"""Command-line training entry point."""

import argparse
from collections.abc import Sequence

from ticket_triage.constants import LABEL_COLUMN, TEXT_COLUMN
from ticket_triage.data import load_dataset, stratified_split
from ticket_triage.evaluation import evaluate_predictions, format_evaluation
from ticket_triage.model import build_pipeline, fit_pipeline, predict, save_pipeline


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the ticket classifier")
    parser.add_argument("--data", required=True, help="Path to labeled training CSV")
    parser.add_argument(
        "--model-output", required=True, help="Destination for fitted joblib pipeline"
    )
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="Logistic-regression class weighting strategy",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Train, evaluate, refit on all data, and save the model."""
    args = parse_args(argv)
    frame = load_dataset(args.data)
    train_frame, validation_frame = stratified_split(frame)
    class_weight = None if args.class_weight == "none" else "balanced"
    validation_pipeline = build_pipeline(class_weight=class_weight)
    fit_pipeline(validation_pipeline, train_frame)
    predictions = predict(validation_pipeline, validation_frame[TEXT_COLUMN])
    result = evaluate_predictions(validation_frame[LABEL_COLUMN].tolist(), predictions)
    print(format_evaluation(result))
    print(
        "Warning: this stratified random split contains closely related message "
        "templates across partitions; its score does not prove generalization."
    )
    final_pipeline = build_pipeline(class_weight=class_weight)
    fit_pipeline(final_pipeline, frame)
    output_path = save_pipeline(final_pipeline, args.model_output)
    print(f"Saved fitted pipeline to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

