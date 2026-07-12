"""Promote/revert gate: compares a candidate RunRecord against the current baseline
RunRecord and decides whether the candidate becomes the new baseline.

Also owns the baseline pointer file (runs/baseline.json), which records only
{prompt_version, run_id} for whichever run is currently "the baseline" — the full
RunRecord is loaded from runs/<run_id>.json on demand via src.runner.load_run_record.

Usage:
    python -m src.gate
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from src.dataset import CLASSES
import src.runner as runner
from src.runner import RunRecord


@dataclass(frozen=True)
class GateConfig:
    min_macro_f1_gain: float = 0.005
    max_per_class_f1_drop: float = 0.05
    min_parse_rate: float = 0.99
    max_cost_multiplier: float = 1.5
    max_latency_multiplier: float = 1.5


@dataclass(frozen=True)
class GateDecision:
    verdict: Literal["PROMOTE", "REVERT"]
    reasons: list[str]


def _check_macro_f1_gain(candidate: RunRecord, baseline: RunRecord, cfg: GateConfig) -> tuple[bool, str]:
    cand_macro = candidate.metrics["overall"]["macro_f1"]
    base_macro = baseline.metrics["overall"]["macro_f1"]
    gain = cand_macro - base_macro
    passed = gain >= cfg.min_macro_f1_gain
    status = "PASS" if passed else "FAIL"
    return passed, (
        f"[{status}] macro_f1 gain: {gain:+.4f} "
        f"({cand_macro:.4f} vs baseline {base_macro:.4f}, min required {cfg.min_macro_f1_gain:+.4f})"
    )


def _check_per_class_f1_drop(candidate: RunRecord, baseline: RunRecord, cfg: GateConfig) -> tuple[bool, str]:
    cand_per_class = candidate.metrics["overall"]["per_class"]
    base_per_class = baseline.metrics["overall"]["per_class"]

    worst_class = None
    worst_drop = float("-inf")
    for cls in CLASSES:
        drop = base_per_class[cls]["f1"] - cand_per_class[cls]["f1"]
        if drop > worst_drop:
            worst_drop = drop
            worst_class = cls

    passed = worst_drop <= cfg.max_per_class_f1_drop
    status = "PASS" if passed else "FAIL"
    return passed, (
        f"[{status}] max per-class f1 drop: {worst_drop:+.4f} on {worst_class!r} "
        f"(max allowed {cfg.max_per_class_f1_drop:.4f})"
    )


def _check_parse_rate(candidate: RunRecord, cfg: GateConfig) -> tuple[bool, str]:
    parse_rate = 1.0 - candidate.metrics["overall"]["parse_failure_rate"]
    passed = parse_rate >= cfg.min_parse_rate
    status = "PASS" if passed else "FAIL"
    return passed, f"[{status}] parse rate: {parse_rate:.4f} (min required {cfg.min_parse_rate:.4f})"


def _check_cost_multiplier(candidate: RunRecord, baseline: RunRecord, cfg: GateConfig) -> tuple[bool, str]:
    base_cost = baseline.totals["est_cost_usd"]
    cand_cost = candidate.totals["est_cost_usd"]
    if base_cost <= 0:
        return True, f"[PASS] cost multiplier: baseline cost is 0, skipping (candidate cost ${cand_cost:.6f})"
    multiplier = cand_cost / base_cost
    passed = multiplier <= cfg.max_cost_multiplier
    status = "PASS" if passed else "FAIL"
    return passed, f"[{status}] cost multiplier: {multiplier:.2f}x (max allowed {cfg.max_cost_multiplier:.2f}x)"


def _check_latency_multiplier(candidate: RunRecord, baseline: RunRecord, cfg: GateConfig) -> tuple[bool, str]:
    base_latency = baseline.totals["latency_ms"]
    cand_latency = candidate.totals["latency_ms"]
    if base_latency <= 0:
        return True, (
            f"[PASS] latency multiplier: baseline latency is 0, skipping "
            f"(candidate latency {cand_latency:.0f}ms)"
        )
    multiplier = cand_latency / base_latency
    passed = multiplier <= cfg.max_latency_multiplier
    status = "PASS" if passed else "FAIL"
    return passed, (
        f"[{status}] latency multiplier: {multiplier:.2f}x (max allowed {cfg.max_latency_multiplier:.2f}x)"
    )


def evaluate_gate(candidate: RunRecord, baseline: RunRecord, cfg: GateConfig = GateConfig()) -> GateDecision:
    """Decide whether `candidate` should replace `baseline`.

    Refuses (raises ValueError) to compare runs evaluated on different datasets or
    different models — that comparison is meaningless, not merely risky, so this is a
    hard error rather than a gate rule with a reason line.

    Every rule below always contributes a reason line, whether it passed or failed, so
    a REVERT is always explainable and a PROMOTE is always auditable.
    """
    if candidate.dataset_fingerprint != baseline.dataset_fingerprint:
        raise ValueError(
            f"cannot compare runs evaluated on different datasets: "
            f"candidate={candidate.dataset_fingerprint!r} baseline={baseline.dataset_fingerprint!r}"
        )
    if candidate.model != baseline.model:
        raise ValueError(
            f"cannot compare runs evaluated on different models: "
            f"candidate={candidate.model!r} baseline={baseline.model!r}"
        )

    checks = [
        _check_macro_f1_gain(candidate, baseline, cfg),
        _check_per_class_f1_drop(candidate, baseline, cfg),
        _check_parse_rate(candidate, cfg),
        _check_cost_multiplier(candidate, baseline, cfg),
        _check_latency_multiplier(candidate, baseline, cfg),
    ]

    verdict = "PROMOTE" if all(passed for passed, _ in checks) else "REVERT"
    return GateDecision(verdict=verdict, reasons=[reason for _, reason in checks])


def load_baseline_pointer() -> dict[str, str] | None:
    """Read runs/baseline.json, or None if no baseline has been set yet."""
    path = runner.RUNS_DIR / "baseline.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline_pointer(prompt_version: str, run_id: str) -> None:
    """Write runs/baseline.json, overwriting any existing pointer."""
    runner.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = runner.RUNS_DIR / "baseline.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({"prompt_version": prompt_version, "run_id": run_id}, f, indent=2)


def load_baseline_run() -> RunRecord | None:
    """Load the full RunRecord the baseline pointer refers to, or None if unset."""
    pointer = load_baseline_pointer()
    if pointer is None:
        return None
    path = runner.RUNS_DIR / f"{pointer['run_id']}.json"
    return runner.load_run_record(path)


def main() -> None:
    pointer = load_baseline_pointer()
    if pointer is None:
        print("No baseline set yet.")
        return
    baseline = load_baseline_run()
    print(f"Current baseline: {pointer['prompt_version']} (run_id={pointer['run_id']})")
    print(f"  macro_f1={baseline.metrics['overall']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
