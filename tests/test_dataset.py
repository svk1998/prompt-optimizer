"""Tests for src.dataset — JSONL loader, validation, and fingerprinting."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset import (
    CLASSES,
    EvalExample,
    class_distribution,
    dataset_fingerprint,
    load_dataset,
)

REAL_DATASET = Path(__file__).resolve().parent.parent / "datasets" / "voc_golden_v1.jsonl"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _valid_rows() -> list[dict]:
    return [
        {"id": "a-1", "text": "The app crashes on launch.", "label": "bug_report", "difficulty": "easy"},
        {"id": "a-2", "text": "Please add a dark mode.", "label": "feature_request", "difficulty": "hard", "note": "recurring request"},
        {"id": "a-3", "text": "I was double charged.", "label": "billing_issue", "difficulty": "easy"},
    ]


class TestClasses:
    def test_classes_contains_exactly_six_expected_labels(self):
        assert CLASSES == frozenset(
            {
                "bug_report",
                "feature_request",
                "billing_issue",
                "ux_complaint",
                "performance_issue",
                "praise",
            }
        )


class TestLoadDatasetValid:
    def test_loads_real_golden_dataset_with_48_examples(self):
        examples = load_dataset(REAL_DATASET)
        assert len(examples) == 48

    def test_loaded_examples_are_eval_example_instances(self):
        examples = load_dataset(REAL_DATASET)
        assert all(isinstance(e, EvalExample) for e in examples)

    def test_loads_synthetic_valid_dataset(self, tmp_path: Path):
        path = _write_jsonl(tmp_path / "valid.jsonl", _valid_rows())
        examples = load_dataset(path)
        assert len(examples) == 3
        assert examples[0].id == "a-1"
        assert examples[0].label == "bug_report"
        assert examples[0].difficulty == "easy"
        assert examples[0].note is None

    def test_note_field_is_preserved_when_present(self, tmp_path: Path):
        path = _write_jsonl(tmp_path / "valid.jsonl", _valid_rows())
        examples = load_dataset(path)
        assert examples[1].note == "recurring request"


class TestLoadDatasetErrors:
    def test_duplicate_id_raises_valueerror_with_line_number(self, tmp_path: Path):
        rows = _valid_rows()
        rows.append({"id": "a-1", "text": "Another one.", "label": "praise", "difficulty": "easy"})
        path = _write_jsonl(tmp_path / "dup.jsonl", rows)
        with pytest.raises(ValueError, match=r"line 4"):
            load_dataset(path)

    def test_unknown_label_raises_valueerror_with_line_number(self, tmp_path: Path):
        rows = _valid_rows()
        rows[1]["label"] = "not_a_real_label"
        path = _write_jsonl(tmp_path / "bad_label.jsonl", rows)
        with pytest.raises(ValueError, match=r"line 2"):
            load_dataset(path)

    def test_empty_text_raises_valueerror_with_line_number(self, tmp_path: Path):
        rows = _valid_rows()
        rows[2]["text"] = ""
        path = _write_jsonl(tmp_path / "empty_text.jsonl", rows)
        with pytest.raises(ValueError, match=r"line 3"):
            load_dataset(path)

    def test_invalid_difficulty_raises_valueerror_with_line_number(self, tmp_path: Path):
        rows = _valid_rows()
        rows[0]["difficulty"] = "medium"
        path = _write_jsonl(tmp_path / "bad_difficulty.jsonl", rows)
        with pytest.raises(ValueError, match=r"line 1"):
            load_dataset(path)


class TestFingerprint:
    def test_reordering_lines_leaves_fingerprint_unchanged(self, tmp_path: Path):
        rows = _valid_rows()
        forward_path = _write_jsonl(tmp_path / "forward.jsonl", rows)
        reversed_path = _write_jsonl(tmp_path / "reversed.jsonl", list(reversed(rows)))

        forward_fp = dataset_fingerprint(load_dataset(forward_path))
        reversed_fp = dataset_fingerprint(load_dataset(reversed_path))

        assert forward_fp == reversed_fp

    def test_editing_one_label_changes_fingerprint(self, tmp_path: Path):
        rows = _valid_rows()
        original_path = _write_jsonl(tmp_path / "original.jsonl", rows)
        original_fp = dataset_fingerprint(load_dataset(original_path))

        edited_rows = _valid_rows()
        edited_rows[0]["label"] = "praise"
        edited_path = _write_jsonl(tmp_path / "edited.jsonl", edited_rows)
        edited_fp = dataset_fingerprint(load_dataset(edited_path))

        assert original_fp != edited_fp

    def test_fingerprint_is_12_lowercase_hex_chars(self, tmp_path: Path):
        path = _write_jsonl(tmp_path / "valid.jsonl", _valid_rows())
        fp = dataset_fingerprint(load_dataset(path))
        assert len(fp) == 12
        assert all(c in "0123456789abcdef" for c in fp)

    def test_editing_note_changes_fingerprint(self, tmp_path: Path):
        rows = _valid_rows()
        original_path = _write_jsonl(tmp_path / "original.jsonl", rows)
        original_fp = dataset_fingerprint(load_dataset(original_path))

        edited_rows = _valid_rows()
        edited_rows[1]["note"] = "a different note"
        edited_path = _write_jsonl(tmp_path / "edited.jsonl", edited_rows)
        edited_fp = dataset_fingerprint(load_dataset(edited_path))

        assert original_fp != edited_fp

    def test_real_dataset_fingerprint_is_deterministic_across_loads(self):
        fp1 = dataset_fingerprint(load_dataset(REAL_DATASET))
        fp2 = dataset_fingerprint(load_dataset(REAL_DATASET))
        assert fp1 == fp2


class TestClassDistribution:
    def test_counts_examples_per_label(self, tmp_path: Path):
        rows = _valid_rows()
        rows.append({"id": "a-4", "text": "Another bug.", "label": "bug_report", "difficulty": "hard"})
        path = _write_jsonl(tmp_path / "dist.jsonl", rows)
        examples = load_dataset(path)

        dist = class_distribution(examples)

        assert dist == {
            "bug_report": 2,
            "feature_request": 1,
            "billing_issue": 1,
        }

    def test_real_dataset_distribution_sums_to_48(self):
        examples = load_dataset(REAL_DATASET)
        dist = class_distribution(examples)
        assert sum(dist.values()) == 48
