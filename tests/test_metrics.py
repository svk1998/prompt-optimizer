"""Tests for src.metrics — hand-rolled classification metrics, verified against sklearn.

scikit-learn is imported ONLY in this test file, never in src/metrics.py (a hard
constraint of the module). We use it purely as a reference oracle.

Parse-failure fixture note: when checking the parse-failure fixture against sklearn, we
call `precision_recall_fscore_support(y_true, y_pred, labels=sorted(CLASSES), ...)`
WITHOUT dropping or filtering any samples first. sklearn handles y_pred values outside
`labels` gracefully in this mode: they never count as a false positive for any real
class (since the predicted value never equals a real label), but they DO still count as
a false negative for their true class (since true != pred). This means parse failures
are fully "visible" in the sklearn comparison (they reduce recall/support-weighted
correctness) without silently vanishing — which is exactly the pitfall the task brief
warns about. Passing `labels=` restricts which labels are *averaged*, not which samples
are *counted*.
"""
from __future__ import annotations

import math

import pytest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.dataset import CLASSES
from src.metrics import (
    PARSE_FAILURE,
    accuracy,
    confusion_matrix,
    format_report,
    macro_f1,
    micro_f1,
    per_class_prf,
)

SIX_CLASSES = sorted(CLASSES)


class TestConfusionMatrix:
    def test_rows_are_all_classes_even_with_zero_support(self):
        y_true = ["bug_report", "bug_report"]
        y_pred = ["bug_report", "praise"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        assert set(cm.keys()) == set(CLASSES)

    def test_diagonal_counts_correct_predictions(self):
        y_true = ["bug_report", "bug_report", "praise"]
        y_pred = ["bug_report", "praise", "praise"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        assert cm["bug_report"]["bug_report"] == 1
        assert cm["bug_report"]["praise"] == 1
        assert cm["praise"]["praise"] == 1

    def test_parse_failure_prediction_counted_in_dedicated_column(self):
        y_true = ["bug_report", "praise"]
        y_pred = ["PARSE_FAILURE", "also_not_a_class"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        assert cm["bug_report"][PARSE_FAILURE] == 1
        assert cm["praise"][PARSE_FAILURE] == 1
        # real-class columns for that row stay at zero
        assert cm["bug_report"]["bug_report"] == 0

    def test_unknown_true_label_raises_valueerror(self):
        with pytest.raises(ValueError):
            confusion_matrix(["not_a_real_class"], ["praise"], CLASSES)

    def test_mismatched_lengths_raises_valueerror(self):
        with pytest.raises(ValueError):
            confusion_matrix(["bug_report", "praise"], ["bug_report"], CLASSES)

    def test_empty_classes_raises_valueerror(self):
        with pytest.raises(ValueError, match="classes"):
            confusion_matrix([], [], [])


class TestPerClassPrfAgainstSklearn:
    def test_normal_case_all_classes_represented(self):
        y_true = [
            "bug_report", "bug_report", "feature_request", "billing_issue",
            "ux_complaint", "performance_issue", "praise", "praise",
            "bug_report", "feature_request",
        ]
        y_pred = [
            "bug_report", "feature_request", "feature_request", "billing_issue",
            "ux_complaint", "praise", "praise", "praise",
            "bug_report", "feature_request",
        ]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        per_class = per_class_prf(cm)

        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            y_true, y_pred, labels=SIX_CLASSES, average=None, zero_division=0
        )
        for cls, p, r, f, s in zip(SIX_CLASSES, precisions, recalls, f1s, supports):
            assert per_class[cls]["precision"] == pytest.approx(p)
            assert per_class[cls]["recall"] == pytest.approx(r)
            assert per_class[cls]["f1"] == pytest.approx(f)
            assert per_class[cls]["support"] == s

    def test_absent_class_zero_true_and_zero_predicted_is_zero_not_nan(self):
        # 'praise' never appears in y_true or y_pred.
        y_true = ["bug_report", "feature_request", "billing_issue"]
        y_pred = ["bug_report", "feature_request", "billing_issue"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        per_class = per_class_prf(cm)

        assert per_class["praise"]["precision"] == 0.0
        assert per_class["praise"]["recall"] == 0.0
        assert per_class["praise"]["f1"] == 0.0
        assert per_class["praise"]["support"] == 0
        assert not math.isnan(per_class["praise"]["f1"])

        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            y_true, y_pred, labels=SIX_CLASSES, average=None, zero_division=0
        )
        for cls, p, r, f, s in zip(SIX_CLASSES, precisions, recalls, f1s, supports):
            assert per_class[cls]["precision"] == pytest.approx(p)
            assert per_class[cls]["recall"] == pytest.approx(r)
            assert per_class[cls]["f1"] == pytest.approx(f)
            assert per_class[cls]["support"] == s

    def test_parse_failure_case_matches_sklearn_with_full_samples(self):
        y_true = ["bug_report", "bug_report", "praise", "billing_issue"]
        y_pred = ["bug_report", "PARSE_FAILURE", "praise", "PARSE_FAILURE"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        per_class = per_class_prf(cm)

        # No filtering of y_true/y_pred here — see module docstring for why this is
        # the correct comparison call (labels= restricts averaging, not sample counting).
        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            y_true, y_pred, labels=SIX_CLASSES, average=None, zero_division=0
        )
        for cls, p, r, f, s in zip(SIX_CLASSES, precisions, recalls, f1s, supports):
            assert per_class[cls]["precision"] == pytest.approx(p)
            assert per_class[cls]["recall"] == pytest.approx(r)
            assert per_class[cls]["f1"] == pytest.approx(f)
            assert per_class[cls]["support"] == s

        # bug_report: true count 2, 1 correct, 1 parse failure -> recall 0.5, precision 1.0
        assert per_class["bug_report"]["recall"] == pytest.approx(0.5)
        assert per_class["bug_report"]["precision"] == pytest.approx(1.0)
        # billing_issue: true count 1, 0 correct (went to parse failure) -> recall 0.0
        assert per_class["billing_issue"]["recall"] == pytest.approx(0.0)


class TestAccuracy:
    def test_matches_sklearn_normal_case(self):
        y_true = ["bug_report", "praise", "feature_request", "bug_report"]
        y_pred = ["bug_report", "praise", "billing_issue", "bug_report"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        assert accuracy(cm) == pytest.approx(accuracy_score(y_true, y_pred))

    def test_parse_failures_count_as_wrong(self):
        y_true = ["bug_report", "bug_report"]
        y_pred = ["bug_report", "PARSE_FAILURE"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        # sklearn's accuracy_score treats the parse-failure string as just another
        # wrong label — same effective semantics as ours here.
        assert accuracy(cm) == pytest.approx(accuracy_score(y_true, y_pred))
        assert accuracy(cm) == pytest.approx(0.5)


class TestMacroF1:
    def test_macro_f1_is_unweighted_mean_over_six_real_classes(self):
        y_true = ["bug_report", "bug_report", "praise", "billing_issue"]
        y_pred = ["bug_report", "PARSE_FAILURE", "praise", "PARSE_FAILURE"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        per_class = per_class_prf(cm)
        macro = macro_f1(per_class)

        _, _, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=SIX_CLASSES, average=None, zero_division=0
        )
        assert macro == pytest.approx(sum(f1s) / len(f1s))

        _, _, macro_sklearn, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=SIX_CLASSES, average="macro", zero_division=0
        )
        assert macro == pytest.approx(macro_sklearn)


class TestMicroF1:
    def test_normal_case_matches_sklearn(self):
        y_true = ["bug_report", "praise", "feature_request", "bug_report", "praise"]
        y_pred = ["bug_report", "praise", "billing_issue", "bug_report", "feature_request"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        micro = micro_f1(cm)

        _, _, micro_sklearn, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=SIX_CLASSES, average="micro", zero_division=0
        )
        assert micro == pytest.approx(micro_sklearn)

    def test_parse_failure_case_matches_sklearn_micro_with_labels_restricted(self):
        # See module docstring: passing labels=SIX_CLASSES (not including the parse
        # failure sentinel) to sklearn with the *full*, unfiltered y_true/y_pred
        # reproduces our documented abstention semantics: a parse failure can never
        # be a false positive for a real class (it doesn't equal any real label), but
        # it always counts as a false negative for its true class.
        y_true = ["bug_report", "bug_report", "praise"]
        y_pred = ["bug_report", "PARSE_FAILURE", "praise"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        micro = micro_f1(cm)

        _, _, micro_sklearn, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=SIX_CLASSES, average="micro", zero_division=0
        )
        assert micro == pytest.approx(micro_sklearn)
        # Hand-verified: tp=2, fp=0 (no wrong prediction landed on a real class),
        # fn=1 (the parse failure) -> precision=1.0, recall=2/3, f1=0.8
        assert micro == pytest.approx(0.8)

    def test_micro_recall_component_equals_accuracy(self):
        # Documented property of our chosen semantics: since every example
        # contributes to exactly one true-class row (either its own TP or a FN for
        # that row, whether via misclassification or parse failure), micro recall
        # always equals accuracy. Micro F1 combines that with micro precision (which
        # excludes parse failures from the false-positive count), so micro_f1 >=
        # accuracy whenever there are parse failures.
        y_true = ["bug_report", "bug_report", "praise", "billing_issue"]
        y_pred = ["bug_report", "PARSE_FAILURE", "praise", "PARSE_FAILURE"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        assert micro_f1(cm) >= accuracy(cm)


class TestFormatReport:
    def test_returns_string_containing_key_sections(self):
        y_true = ["bug_report", "bug_report", "praise", "billing_issue"]
        y_pred = ["bug_report", "PARSE_FAILURE", "praise", "PARSE_FAILURE"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        report = format_report(cm)

        assert isinstance(report, str)
        assert "bug_report" in report
        assert "praise" in report
        assert "accuracy" in report.lower()
        assert "macro" in report.lower()
        assert "micro" in report.lower()
        assert "parse" in report.lower()

    def test_parse_failure_rate_reflected_in_report(self):
        y_true = ["bug_report", "bug_report"]
        y_pred = ["bug_report", "PARSE_FAILURE"]
        cm = confusion_matrix(y_true, y_pred, CLASSES)
        report = format_report(cm)
        assert "0.5" in report or "50" in report
