"""End-to-end pipeline: run a prompt version against the golden dataset, gate it
against the current baseline, and promote or revert.

Usage:
    python main.py --prompt v1                  # bootstraps the baseline if none exists
    python main.py --prompt v2 --dry-run         # no API key needed
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from src.dataset import load_dataset
from src.gate import GateConfig, GateDecision, evaluate_gate, load_baseline_run, save_baseline_pointer
from src.registry import load_prompt
from src.runner import MODEL, DEFAULT_DATASET_PATH, RunRecord, build_client, run_eval, save_run_record


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a prompt version, gate it against the baseline, and promote or revert."
    )
    parser.add_argument("--prompt", required=True, help="prompt version to evaluate, e.g. v1")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument(
        "--dry-run", action="store_true", help="use a mock client instead of the real Groq API"
    )
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[RunRecord, GateDecision | None]:
    args = _parse_args(argv)
    load_dotenv()

    examples = load_dataset(args.dataset)
    prompt = load_prompt(args.prompt)
    client = build_client(args.dry_run)

    candidate = run_eval(
        prompt,
        examples,
        MODEL,
        client,
        n_repeats=args.n_repeats,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
    )
    save_run_record(candidate)

    baseline = load_baseline_run()
    if baseline is None:
        print(f"No baseline set - bootstrapping with {candidate.prompt_version} (run_id={candidate.run_id})")
        save_baseline_pointer(candidate.prompt_version, candidate.run_id)
        return candidate, None

    decision = evaluate_gate(candidate, baseline, GateConfig())
    print(f"Gate decision: {decision.verdict}")
    for reason in decision.reasons:
        print(f"  {reason}")

    if decision.verdict == "PROMOTE":
        save_baseline_pointer(candidate.prompt_version, candidate.run_id)
        print(f"Baseline updated to {candidate.prompt_version} (run_id={candidate.run_id})")

    return candidate, decision


if __name__ == "__main__":
    main()
