"""Classification evaluation and reporting."""

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

from ticket_triage.constants import ALLOWED_LABELS, FRAUD_LABEL


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics produced for a validation prediction set."""

    accuracy: float
    macro_f1: float
    fraud_recall: float
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: list[list[int]]


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> EvaluationResult:
    """Calculate overall and class-specific routing metrics."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(ALLOWED_LABELS),
        output_dict=True,
        zero_division=0,
    )
    per_class: dict[str, dict[str, float | int]] = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in ALLOWED_LABELS
    }
    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        fraud_recall=float(
            recall_score(
                y_true,
                y_pred,
                labels=[FRAUD_LABEL],
                average=None,
                zero_division=0,
            )[0]
        ),
        per_class=per_class,
        confusion_matrix=confusion_matrix(
            y_true, y_pred, labels=list(ALLOWED_LABELS)
        ).tolist(),
    )


def format_evaluation(result: EvaluationResult) -> str:
    """Render metrics as a compact human-readable report."""
    lines = [
        f"Accuracy: {result.accuracy:.4f}",
        f"Macro F1: {result.macro_f1:.4f}",
        f"Fraud-report recall: {result.fraud_recall:.4f}",
        "Per-class metrics:",
    ]
    for label in ALLOWED_LABELS:
        metrics = result.per_class[label]
        lines.append(
            f"  {label}: precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
            f"support={metrics['support']}"
        )
    lines.extend(
        [
            f"Confusion matrix label order: {list(ALLOWED_LABELS)}",
            "Confusion matrix:",
            *[f"  {row}" for row in result.confusion_matrix],
        ]
    )
    return "\n".join(lines)

