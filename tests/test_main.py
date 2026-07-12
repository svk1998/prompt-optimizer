"""Integration tests for main.py — the run -> gate -> (promote|revert) orchestrator.

Deliberately thin: per-rule gate behavior is covered exhaustively in test_gate.py
against evaluate_gate directly. These tests only exercise main.py's own wiring:
bootstrap-when-no-baseline, revert-leaves-pointer-unchanged, and that a raised
ValueError (mismatched fingerprints) is not swallowed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import main as main_module
import src.gate as gate
import src.runner as runner


class TestMainBootstrap:
    def test_bootstraps_baseline_when_none_exists(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        candidate, decision = main_module.main(["--prompt", "v1", "--dry-run"])

        assert decision is None
        pointer = gate.load_baseline_pointer()
        assert pointer == {"prompt_version": "v1", "run_id": candidate.run_id}


class TestMainGateWiring:
    def test_second_run_reverts_and_leaves_baseline_pointer_unchanged(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        first, _ = main_module.main(["--prompt", "v1", "--dry-run"])
        second, decision = main_module.main(["--prompt", "v1", "--dry-run"])

        assert decision is not None
        assert decision.verdict == "REVERT"
        pointer = gate.load_baseline_pointer()
        assert pointer == {"prompt_version": "v1", "run_id": first.run_id}
        assert pointer["run_id"] != second.run_id

    def test_mismatched_dataset_fingerprint_raises_and_is_not_swallowed(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        stale_baseline = runner.RunRecord(
            run_id="20200101T000000Z_v1",
            timestamp="20200101T000000Z",
            prompt_version="v1",
            prompt_fingerprint="does-not-matter",
            dataset_fingerprint="stale-fingerprint-does-not-match",
            model=runner.MODEL,
            params={"temperature": 0.0, "max_tokens": 20, "n_repeats": 1},
            per_example=[],
            metrics={
                "overall": {
                    "accuracy": 0.5,
                    "macro_f1": 0.5,
                    "micro_f1": 0.5,
                    "parse_failure_rate": 0.0,
                    "support": 1,
                    "per_class": {},
                }
            },
            totals={"latency_ms": 1.0, "prompt_tokens": 1, "completion_tokens": 1, "est_cost_usd": 0.0},
        )
        runner.save_run_record(stale_baseline)
        gate.save_baseline_pointer("v1", stale_baseline.run_id)

        with pytest.raises(ValueError, match=r"dataset"):
            main_module.main(["--prompt", "v1", "--dry-run"])
