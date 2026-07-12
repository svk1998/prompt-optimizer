"""Tests for src.gate — promote/revert decisions and the baseline pointer file."""
from __future__ import annotations

from pathlib import Path

import pytest

import src.gate as gate
import src.runner as runner
from src.dataset import CLASSES


def _per_class(f1_by_class: dict[str, float]) -> dict[str, dict[str, float | int]]:
    return {
        cls: {"precision": f1, "recall": f1, "f1": f1, "support": 8}
        for cls, f1 in f1_by_class.items()
    }


def _metrics_slice(macro_f1: float, per_class_f1: dict[str, float], parse_failure_rate: float = 0.0) -> dict:
    return {
        "accuracy": macro_f1,
        "macro_f1": macro_f1,
        "micro_f1": macro_f1,
        "parse_failure_rate": parse_failure_rate,
        "support": 48,
        "per_class": _per_class(per_class_f1),
    }


def _run_record(
    *,
    run_id: str = "20260101T000000Z_v1",
    prompt_version: str = "v1",
    dataset_fingerprint: str = "ds-fingerprint",
    model: str = "test-model",
    macro_f1: float = 0.60,
    per_class_f1: dict[str, float] | None = None,
    parse_failure_rate: float = 0.0,
    cost: float = 0.01,
    latency_ms: float = 1000.0,
) -> runner.RunRecord:
    if per_class_f1 is None:
        per_class_f1 = {cls: macro_f1 for cls in CLASSES}
    overall = _metrics_slice(macro_f1, per_class_f1, parse_failure_rate)
    return runner.RunRecord(
        run_id=run_id,
        timestamp="20260101T000000Z",
        prompt_version=prompt_version,
        prompt_fingerprint="prompt-fingerprint",
        dataset_fingerprint=dataset_fingerprint,
        model=model,
        params={"temperature": 0.0, "max_tokens": 20, "n_repeats": 1},
        per_example=[],
        metrics={"overall": overall, "easy": overall, "hard": overall},
        totals={
            "latency_ms": latency_ms,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "est_cost_usd": cost,
        },
    )


class TestEvaluateGateVerdicts:
    def test_clean_promote_when_all_rules_pass(self):
        baseline = _run_record(macro_f1=0.60, cost=0.010, latency_ms=1000)
        candidate = _run_record(macro_f1=0.62, cost=0.011, latency_ms=1050)

        decision = gate.evaluate_gate(candidate, baseline, gate.GateConfig())

        assert decision.verdict == "PROMOTE"
        assert len(decision.reasons) == 5
        assert all(r.startswith("[PASS]") for r in decision.reasons)

    def test_reverts_when_one_class_collapses_despite_aggregate_gain(self):
        baseline_per_class = {cls: 0.60 for cls in CLASSES}
        collapsed_class = sorted(CLASSES)[0]
        candidate_per_class = {cls: 0.90 for cls in CLASSES}
        candidate_per_class[collapsed_class] = 0.0  # collapses far past max_per_class_f1_drop

        baseline = _run_record(macro_f1=0.60, per_class_f1=baseline_per_class)
        candidate = _run_record(
            macro_f1=sum(candidate_per_class.values()) / len(candidate_per_class),
            per_class_f1=candidate_per_class,
        )

        decision = gate.evaluate_gate(candidate, baseline, gate.GateConfig())

        assert decision.verdict == "REVERT"
        per_class_reason = next(r for r in decision.reasons if "per-class" in r)
        assert per_class_reason.startswith("[FAIL]")
        assert collapsed_class in per_class_reason

    def test_reverts_on_parse_rate_violation(self):
        baseline = _run_record(macro_f1=0.60, parse_failure_rate=0.0)
        candidate = _run_record(macro_f1=0.65, parse_failure_rate=0.05)  # parse rate 0.95 < 0.99

        decision = gate.evaluate_gate(candidate, baseline, gate.GateConfig())

        assert decision.verdict == "REVERT"
        parse_reason = next(r for r in decision.reasons if "parse rate" in r)
        assert parse_reason.startswith("[FAIL]")

    def test_reverts_on_cost_multiplier_violation(self):
        baseline = _run_record(macro_f1=0.60, cost=0.010)
        candidate = _run_record(macro_f1=0.65, cost=0.020)  # 2x > 1.5x max

        decision = gate.evaluate_gate(candidate, baseline, gate.GateConfig())

        assert decision.verdict == "REVERT"
        cost_reason = next(r for r in decision.reasons if "cost" in r)
        assert cost_reason.startswith("[FAIL]")

    def test_reverts_on_latency_multiplier_violation(self):
        baseline = _run_record(macro_f1=0.60, latency_ms=1000)
        candidate = _run_record(macro_f1=0.65, latency_ms=2000)  # 2x > 1.5x max

        decision = gate.evaluate_gate(candidate, baseline, gate.GateConfig())

        assert decision.verdict == "REVERT"
        latency_reason = next(r for r in decision.reasons if "latency" in r)
        assert latency_reason.startswith("[FAIL]")

    def test_reverts_on_insufficient_macro_f1_gain(self):
        baseline = _run_record(macro_f1=0.60)
        candidate = _run_record(macro_f1=0.601)  # gain 0.001 < min 0.005

        decision = gate.evaluate_gate(candidate, baseline, gate.GateConfig())

        assert decision.verdict == "REVERT"
        gain_reason = next(r for r in decision.reasons if "macro_f1 gain" in r)
        assert gain_reason.startswith("[FAIL]")

    def test_zero_baseline_cost_does_not_trigger_cost_violation(self):
        baseline = _run_record(macro_f1=0.60, cost=0.0)
        candidate = _run_record(macro_f1=0.65, cost=0.05)

        decision = gate.evaluate_gate(candidate, baseline, gate.GateConfig())

        cost_reason = next(r for r in decision.reasons if "cost" in r)
        assert cost_reason.startswith("[PASS]")


class TestEvaluateGateRaises:
    def test_raises_on_mismatched_dataset_fingerprint(self):
        baseline = _run_record(dataset_fingerprint="fp-a")
        candidate = _run_record(dataset_fingerprint="fp-b")

        with pytest.raises(ValueError, match=r"dataset"):
            gate.evaluate_gate(candidate, baseline, gate.GateConfig())

    def test_raises_on_mismatched_model(self):
        baseline = _run_record(model="model-a")
        candidate = _run_record(model="model-b")

        with pytest.raises(ValueError, match=r"model"):
            gate.evaluate_gate(candidate, baseline, gate.GateConfig())


class TestBaselinePointer:
    def test_load_baseline_pointer_returns_none_when_missing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)

        assert gate.load_baseline_pointer() is None

    def test_save_and_load_baseline_pointer_round_trips(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)

        gate.save_baseline_pointer("v2", "20260101T000000Z_v2")
        pointer = gate.load_baseline_pointer()

        assert pointer == {"prompt_version": "v2", "run_id": "20260101T000000Z_v2"}

    def test_load_baseline_run_returns_none_when_no_pointer(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)

        assert gate.load_baseline_run() is None

    def test_load_baseline_run_loads_full_run_record(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        record = _run_record(run_id="20260101T000000Z_v1", prompt_version="v1")
        runner.save_run_record(record)
        gate.save_baseline_pointer("v1", record.run_id)

        loaded = gate.load_baseline_run()

        assert loaded == record
