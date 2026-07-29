"""Random and leakage-aware grouped cross-validation comparison."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)

from ticket_triage.constants import FRAUD_LABEL, LABEL_COLUMN, RANDOM_STATE, TEXT_COLUMN
from ticket_triage.data import load_dataset
from ticket_triage.evaluation import evaluate_predictions
from ticket_triage.model import ClassWeight, build_pipeline
from ticket_triage.template_groups import build_template_groups


@dataclass(frozen=True)
class Experiment:
    """Configuration for one out-of-fold evaluation."""

    name: str
    grouped: bool
    class_weight: ClassWeight


EXPERIMENTS = (
    Experiment("Random CV, unweighted", grouped=False, class_weight=None),
    Experiment("Random CV, balanced", grouped=False, class_weight="balanced"),
    Experiment("Grouped CV, unweighted", grouped=True, class_weight=None),
    Experiment("Grouped CV, balanced", grouped=True, class_weight="balanced"),
)


def _splitter(*, grouped: bool) -> StratifiedKFold | StratifiedGroupKFold:
    """Build a reproducible five-fold splitter."""
    splitter_type = StratifiedGroupKFold if grouped else StratifiedKFold
    return splitter_type(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def _assert_group_isolation(
    frame: pd.DataFrame, groups: list[str], splitter: StratifiedGroupKFold
) -> None:
    """Fail fast if any template group crosses a grouped fold boundary."""
    group_series = pd.Series(groups)
    for train_indices, validation_indices in splitter.split(
        frame[TEXT_COLUMN], frame[LABEL_COLUMN], groups
    ):
        train_groups = set(group_series.iloc[train_indices])
        validation_groups = set(group_series.iloc[validation_indices])
        if train_groups.intersection(validation_groups):
            raise RuntimeError("Template group leakage detected across folds")


def run_experiment(
    frame: pd.DataFrame, groups: list[str], experiment: Experiment
) -> dict[str, str | float]:
    """Generate OOF predictions and calculate the required metrics."""
    splitter = _splitter(grouped=experiment.grouped)
    if experiment.grouped:
        assert isinstance(splitter, StratifiedGroupKFold)
        _assert_group_isolation(frame, groups, splitter)

    predictions = cross_val_predict(
        build_pipeline(class_weight=experiment.class_weight),
        frame[TEXT_COLUMN],
        frame[LABEL_COLUMN],
        cv=splitter,
        groups=groups if experiment.grouped else None,
        method="predict",
    )
    result = evaluate_predictions(frame[LABEL_COLUMN].tolist(), predictions.tolist())
    fraud_metrics = result.per_class[FRAUD_LABEL]
    return {
        "Experiment": experiment.name,
        "Accuracy": result.accuracy,
        "Macro F1": result.macro_f1,
        "Fraud precision": float(fraud_metrics["precision"]),
        "Fraud recall": float(fraud_metrics["recall"]),
        "Fraud F1": float(fraud_metrics["f1"]),
    }


def evaluate_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the fixed four-experiment robustness comparison once."""
    groups = build_template_groups(frame[TEXT_COLUMN].tolist())
    rows = [run_experiment(frame, groups, experiment) for experiment in EXPERIMENTS]
    return pd.DataFrame(rows).set_index("Experiment")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse robustness-evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Compare random and template-grouped cross-validation"
    )
    parser.add_argument("--data", required=True, help="Path to labeled training CSV")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Load validated data and print the robustness comparison table."""
    args = parse_args(argv)
    frame = load_dataset(args.data)
    group_count = len(set(build_template_groups(frame[TEXT_COLUMN].tolist())))
    print(f"Rows: {len(frame)}; heuristic template groups: {group_count}")
    print(
        evaluate_robustness(frame).to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )
    print(
        "Note: template normalization is an evaluation-only heuristic; "
        "the classifier always receives original messages."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
