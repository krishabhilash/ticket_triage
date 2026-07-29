"""Optional live comparison of Gemini and the classical grouped holdout."""

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import statistics
import time

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ticket_triage.constants import FRAUD_LABEL, LABEL_COLUMN, RANDOM_STATE, TEXT_COLUMN
from ticket_triage.data import load_dataset
from ticket_triage.evaluation import evaluate_predictions
from ticket_triage.llm import LLMClassificationError, LLMClassifier
from ticket_triage.model import build_pipeline, fit_pipeline, predict
from ticket_triage.template_groups import build_template_groups


def _first_grouped_fold(frame: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Return the first deterministic grouped fold used for both comparisons."""
    groups = build_template_groups(frame[TEXT_COLUMN].tolist())
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_indices, evaluation_indices = next(
        splitter.split(frame[TEXT_COLUMN], frame[LABEL_COLUMN], groups)
    )
    return train_indices.tolist(), evaluation_indices.tolist()


def _metric_row(name: str, labels: list[str], predictions: list[str]) -> dict[str, object]:
    result = evaluate_predictions(labels, predictions)
    fraud = result.per_class[FRAUD_LABEL]
    return {
        "Classifier": name,
        "Macro F1": result.macro_f1,
        "Fraud precision": float(fraud["precision"]),
        "Fraud recall": float(fraud["recall"]),
        "Fraud F1": float(fraud["f1"]),
    }


def run_live_comparison(
    *,
    frame: pd.DataFrame,
    classifier: LLMClassifier,
    class_weight: str | None,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate both classifiers on one untouched template-grouped fold."""
    train_indices, evaluation_indices = _first_grouped_fold(frame)
    train_frame = frame.iloc[train_indices]
    evaluation_frame = frame.iloc[evaluation_indices]
    expected = evaluation_frame[LABEL_COLUMN].tolist()

    pipeline = fit_pipeline(
        build_pipeline(class_weight=class_weight), train_frame
    )
    started = time.perf_counter()
    classical_predictions = predict(pipeline, evaluation_frame[TEXT_COLUMN])
    classical_latency = time.perf_counter() - started

    llm_predictions: list[str] = []
    llm_expected: list[str] = []
    latencies: list[float] = []
    input_tokens = 0
    output_tokens = 0
    invalid_outputs = 0
    failures = 0
    cached_responses = 0
    for message, label in zip(evaluation_frame[TEXT_COLUMN], expected, strict=True):
        try:
            result = classifier.classify(message)
        except LLMClassificationError as error:
            invalid_outputs += error.invalid_outputs
            failures += 1
            continue
        llm_predictions.append(result.label)
        llm_expected.append(label)
        latencies.append(result.latency_seconds)
        input_tokens += result.input_tokens or 0
        output_tokens += result.output_tokens or 0
        invalid_outputs += result.invalid_outputs
        cached_responses += int(result.cached)

    rows = [
        _metric_row("Classical balanced" if class_weight else "Classical unweighted", expected, classical_predictions)
    ]
    if llm_predictions:
        rows.append(_metric_row("Gemini", llm_expected, llm_predictions))
    comparison = pd.DataFrame(rows).set_index("Classifier")

    cost = None
    if input_cost_per_million is not None and output_cost_per_million is not None:
        cost = (
            input_tokens * input_cost_per_million
            + output_tokens * output_cost_per_million
        ) / 1_000_000
    metadata: dict[str, object] = {
        "evaluation_examples": len(evaluation_frame),
        "llm_successes": len(llm_predictions),
        "llm_failures": failures,
        "invalid_or_retry_outputs": invalid_outputs,
        "cached_responses": cached_responses,
        "classical_total_latency_seconds": classical_latency,
        "llm_mean_latency_seconds": statistics.mean(latencies) if latencies else None,
        "llm_p95_latency_seconds": _percentile(latencies, 0.95),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
    }
    return comparison, metadata


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * percentile), len(ordered) - 1)]


def _optional_env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit live-evaluation arguments."""
    parser = argparse.ArgumentParser(description="Run optional live Gemini comparison")
    parser.add_argument("--data", required=True, help="Path to labeled CSV")
    parser.add_argument("--cache", default="artifacts/llm_cache.json")
    parser.add_argument(
        "--class-weight", choices=("none", "balanced"), default="balanced"
    )
    parser.add_argument(
        "--input-cost-per-million",
        type=float,
        default=_optional_env_float("GEMINI_INPUT_COST_PER_MILLION"),
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=float,
        default=_optional_env_float("GEMINI_OUTPUT_COST_PER_MILLION"),
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Acknowledge that uncached examples may create paid API requests",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional comparison only after explicit cost authorization."""
    args = parse_args(argv)
    if not args.confirm_live:
        raise SystemExit("Refusing live requests without --confirm-live")
    frame = load_dataset(args.data)
    classifier = LLMClassifier.from_env(cache_path=args.cache)
    class_weight = None if args.class_weight == "none" else "balanced"
    comparison, metadata = run_live_comparison(
        frame=frame,
        classifier=classifier,
        class_weight=class_weight,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))
    for name, value in metadata.items():
        print(f"{name}: {value}")
    print(
        "Optional comparison: results may vary across Gemini model versions. "
        "No evaluation ticket is used as a few-shot demonstration."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
