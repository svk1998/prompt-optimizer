"""Hand-rolled classification metrics for evaluating VoC prompt predictions.

Pure functions only: everything here operates on lists of true/predicted labels (or on
the confusion matrix derived from them) passed in by the caller. Nothing talks to an
LLM or reads files.

Hard constraint: this module must never import scikit-learn. sklearn is a dev-only
dependency, used exclusively in tests/test_metrics.py to verify these functions against
a trusted reference implementation.

Usage:
    python -m src.metrics

Abstention (parse failure) semantics
-------------------------------------
Later phases (the LLM runner) may fail to parse a model response into one of the six
real classes; that gets passed in as a predicted label outside `classes` (e.g. the
sentinel "PARSE_FAILURE"). Any predicted value not in `classes` is routed into a single
dedicated confusion-matrix column keyed by the `PARSE_FAILURE` constant below,
regardless of what the actual out-of-class string was — the point is visibility
("we failed to parse N examples"), not the identity of the garbage prediction.

For `accuracy` and `per_class_prf`, this is unambiguous: a parse failure is simply a
wrong prediction. It never contributes to a class's TP, and it counts against that
class's recall/support the same as any other misclassification.

`micro_f1` needs an explicit judgment call, since micro-averaging aggregates TP/FP/FN
across all real classes and a parse failure is a prediction that matches none of them:

  - A parse failure ALWAYS counts as a false negative (FN) for its true class — the
    model was supposed to produce that label and didn't.
  - A parse failure NEVER counts as a false positive (FP) for any real class — the
    predicted value doesn't equal any real label, so it can't be mistaken for a
    "wrong but confident" prediction of a specific class.

This is exactly what you get from sklearn's own
`precision_recall_fscore_support(y_true, y_pred, labels=<real classes>, average='micro')`
when y_pred contains values outside `labels` (verified empirically; see
tests/test_metrics.py) — so it's not an idiosyncratic choice, it's what "restrict
averaging to labels=" already means in the library this project's metrics are checked
against. One consequence worth knowing: because every example lands in exactly one
true-class row (as its own TP or as a FN for that row), micro recall always equals
`accuracy`. Micro precision then divides by a smaller "attempted real-class
predictions" denominator (excluding parse failures), so `micro_f1 >= accuracy` whenever
parse failures are present.
"""
from __future__ import annotations

from typing import Iterable

PARSE_FAILURE = "PARSE_FAILURE"


def confusion_matrix(
    y_true: Iterable[str], y_pred: Iterable[str], classes: Iterable[str]
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix: matrix[true_label][pred_label] = count.

    Rows are always the given real `classes`, present even with zero support. Columns
    are the real classes plus one dedicated extra column keyed `PARSE_FAILURE` that
    collects any prediction not found in `classes` (so failed parses stay visible
    instead of being silently dropped or raising).

    Raises ValueError if `y_true` contains a value outside `classes` — true labels are
    expected to always be valid (e.g. sourced from src.dataset.load_dataset, which
    already validates them), so an unknown true label indicates a caller bug.
    """
    class_set = set(classes)
    matrix: dict[str, dict[str, int]] = {
        true_label: {c: 0 for c in class_set} | {PARSE_FAILURE: 0}
        for true_label in class_set
    }

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label not in class_set:
            raise ValueError(
                f"y_true contains label {true_label!r} not in classes {sorted(class_set)}"
            )
        if pred_label in class_set:
            matrix[true_label][pred_label] += 1
        else:
            matrix[true_label][PARSE_FAILURE] += 1

    return matrix


def per_class_prf(cm: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    """Compute per-class precision, recall, f1, and support from a confusion matrix.

    0/0 (a class with zero predicted-and-zero-actual instances) is defined as 0.0 for
    precision/recall/f1 rather than raising or producing nan.
    """
    classes = list(cm.keys())
    result: dict[str, dict[str, float | int]] = {}

    for cls in classes:
        true_positive = cm[cls][cls]
        predicted_count = sum(cm[t][cls] for t in classes)
        support = sum(cm[cls].values())

        precision = true_positive / predicted_count if predicted_count > 0 else 0.0
        recall = true_positive / support if support > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        result[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return result


def accuracy(cm: dict[str, dict[str, int]]) -> float:
    """Correct predictions over all predictions in the matrix.

    Parse failures count as wrong (they never land on the diagonal), so they reduce
    accuracy the same as any other misclassification.
    """
    total = sum(sum(row.values()) for row in cm.values())
    if total == 0:
        return 0.0
    correct = sum(cm[cls][cls] for cls in cm)
    return correct / total


def macro_f1(per_class: dict[str, dict[str, float | int]]) -> float:
    """Unweighted mean of F1 over the real classes in `per_class`.

    `per_class` (from per_class_prf) only ever has the real classes as keys — the
    parse-failure column is never a row in the confusion matrix — so this naturally
    excludes parse failures without any special-casing.
    """
    if not per_class:
        return 0.0
    return sum(m["f1"] for m in per_class.values()) / len(per_class)


def micro_f1(cm: dict[str, dict[str, int]]) -> float:
    """Global micro-averaged F1 computed from the confusion matrix.

    See the module docstring for the documented abstention semantics: parse failures
    count as a false negative for their true class but never as a false positive for
    any real class.
    """
    classes = list(cm.keys())
    true_positive = sum(cm[cls][cls] for cls in classes)
    # Misclassifications into another *real* class: each is simultaneously a FP for
    # the predicted class and a FN for the true class.
    real_misclassifications = sum(
        cm[t][c] for t in classes for c in classes if c != t
    )
    parse_failures = sum(cm[t][PARSE_FAILURE] for t in classes)

    false_positive = real_misclassifications
    false_negative = real_misclassifications + parse_failures

    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return 2 * true_positive / denominator


def format_report(cm: dict[str, dict[str, int]]) -> str:
    """Render a readable text report from a confusion matrix.

    Includes a per-class table (precision, recall, f1, support), macro/micro F1,
    accuracy, and the parse-failure rate. Takes only the confusion matrix — everything
    else needed (per-class stats, aggregates, totals) is derived from it, so callers
    (e.g. src.runner, src.gate) only ever need to build and hand off one object.
    """
    per_class = per_class_prf(cm)
    acc = accuracy(cm)
    macro = macro_f1(per_class)
    micro = micro_f1(cm)

    total = sum(sum(row.values()) for row in cm.values())
    parse_failures = sum(row.get(PARSE_FAILURE, 0) for row in cm.values())
    failure_rate = parse_failures / total if total > 0 else 0.0

    header = f"{'class':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    lines = [header, "-" * len(header)]
    for cls in sorted(per_class):
        m = per_class[cls]
        lines.append(
            f"{cls:<20}{m['precision']:>10.3f}{m['recall']:>10.3f}"
            f"{m['f1']:>10.3f}{m['support']:>10d}"
        )

    lines.append("")
    lines.append(f"accuracy:            {acc:.3f}")
    lines.append(f"macro_f1:            {macro:.3f}")
    lines.append(f"micro_f1:            {micro:.3f}")
    lines.append(
        f"parse_failure_rate:  {failure_rate:.3f} ({parse_failures}/{total})"
    )

    return "\n".join(lines)


def main() -> None:
    classes = {
        "bug_report",
        "feature_request",
        "billing_issue",
        "ux_complaint",
        "performance_issue",
        "praise",
    }
    y_true = [
        "bug_report", "bug_report", "feature_request", "billing_issue",
        "ux_complaint", "performance_issue", "praise", "praise", "bug_report",
    ]
    y_pred = [
        "bug_report", "feature_request", "feature_request", "billing_issue",
        "PARSE_FAILURE", "performance_issue", "praise", "bug_report", "bug_report",
    ]

    cm = confusion_matrix(y_true, y_pred, classes)
    print(format_report(cm))


if __name__ == "__main__":
    main()
