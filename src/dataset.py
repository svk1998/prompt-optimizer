"""Golden dataset loader, validator, and fingerprinter for the VoC classification task.

Usage:
    python -m src.dataset datasets/voc_golden_v1.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CLASSES: frozenset[str] = frozenset(
    {
        "bug_report",
        "feature_request",
        "billing_issue",
        "ux_complaint",
        "performance_issue",
        "praise",
    }
)

_VALID_DIFFICULTIES = frozenset({"easy", "hard"})


@dataclass(frozen=True)
class EvalExample:
    id: str
    text: str
    label: str
    difficulty: str
    note: str | None = None


def load_dataset(path: str | Path) -> list[EvalExample]:
    """Parse a JSONL golden dataset file into a list of EvalExample.

    Fails loudly with ValueError (including the 1-indexed offending line number) on:
    duplicate ids, unknown labels, empty text, or invalid difficulty.
    """
    examples: list[EvalExample] = []
    seen_ids: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            row = json.loads(stripped)

            example_id = row["id"]
            text = row["text"]
            label = row["label"]
            difficulty = row["difficulty"]
            note = row.get("note")

            if example_id in seen_ids:
                raise ValueError(
                    f"line {line_number}: duplicate id {example_id!r}"
                )
            if label not in CLASSES:
                raise ValueError(
                    f"line {line_number}: unknown label {label!r} (must be one of {sorted(CLASSES)})"
                )
            if not text or not text.strip():
                raise ValueError(f"line {line_number}: empty text")
            if difficulty not in _VALID_DIFFICULTIES:
                raise ValueError(
                    f"line {line_number}: invalid difficulty {difficulty!r} (must be 'easy' or 'hard')"
                )

            seen_ids.add(example_id)
            examples.append(
                EvalExample(
                    id=example_id,
                    text=text,
                    label=label,
                    difficulty=difficulty,
                    note=note,
                )
            )

    return examples


def dataset_fingerprint(examples: list[EvalExample]) -> str:
    """Compute a stable fingerprint over the dataset, independent of input order.

    Canonicalization: sort examples by id, serialize each to JSON with sort_keys=True,
    join the per-example JSON strings with newlines, then take the first 12 hex chars
    of the SHA-256 digest of the resulting UTF-8 bytes.
    """
    sorted_examples = sorted(examples, key=lambda e: e.id)
    lines = [
        json.dumps(
            {
                "id": e.id,
                "text": e.text,
                "label": e.label,
                "difficulty": e.difficulty,
                "note": e.note,
            },
            sort_keys=True,
        )
        for e in sorted_examples
    ]
    canonical = "\n".join(lines)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:12]


def class_distribution(examples: list[EvalExample]) -> dict[str, int]:
    """Count examples per label."""
    return dict(Counter(e.label for e in examples))


def summarize(examples: list[EvalExample]) -> None:
    """Print per-class counts and easy/hard split to stdout."""
    print("Class distribution:")
    for label, count in sorted(class_distribution(examples).items()):
        print(f"  {label}: {count}")
    difficulty_counts = Counter(e.difficulty for e in examples)
    print(f"Difficulty split: easy={difficulty_counts.get('easy', 0)} hard={difficulty_counts.get('hard', 0)}")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m src.dataset <path-to-jsonl>", file=sys.stderr)
        raise SystemExit(2)

    path = sys.argv[1]
    examples = load_dataset(path)
    fingerprint = dataset_fingerprint(examples)

    print(f"Loaded {len(examples)} examples from {path}")
    print(f"Fingerprint: {fingerprint}")
    summarize(examples)


if __name__ == "__main__":
    main()
