"""Tests for src.registry — YAML prompt loader, fingerprinting, and version listing."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import src.registry as registry
from src.dataset import CLASSES
from src.registry import PromptVersion, list_versions, load_prompt

REAL_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _write_prompt(dir_path: Path, version: str, **overrides) -> Path:
    """Write a synthetic prompt YAML file with sensible defaults, override as needed."""
    data = {
        "version": version,
        "parent": None,
        "changelog": "baseline",
        "system": "Respond with JSON {\"category\": \"<label>\"}.",
        "user_template": "Feedback: {text}",
    }
    data.update(overrides)
    path = dir_path / f"{version}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


class TestLoadPromptSynthetic:
    def test_loads_synthetic_valid_prompt_with_all_fields(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        _write_prompt(tmp_path, "v1")

        pv = load_prompt("v1")

        assert isinstance(pv, PromptVersion)
        assert pv.version == "v1"
        assert pv.parent is None
        assert pv.changelog == "baseline"
        assert "category" in pv.system
        assert pv.user_template == "Feedback: {text}"
        assert pv.fingerprint

    def test_parent_string_is_loaded_correctly(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        _write_prompt(tmp_path, "v2", parent="v1")

        pv = load_prompt("v2")

        assert pv.parent == "v1"


class TestLoadPromptErrors:
    def test_missing_file_raises_clear_error_with_path(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match=r"nonexistent"):
            load_prompt("nonexistent")

    def test_missing_required_field_raises_valueerror(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        path = tmp_path / "v1.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"version": "v1", "parent": None, "changelog": "baseline"}, f)

        with pytest.raises(ValueError, match=r"missing"):
            load_prompt("v1")

    def test_empty_system_field_raises_valueerror(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        _write_prompt(tmp_path, "v1", system="   ")

        with pytest.raises(ValueError, match=r"empty"):
            load_prompt("v1")

    def test_empty_user_template_raises_valueerror(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        _write_prompt(tmp_path, "v1", user_template="")

        with pytest.raises(ValueError, match=r"empty"):
            load_prompt("v1")

    def test_malformed_yaml_raises_valueerror(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        path = tmp_path / "v1.yaml"
        with path.open("w", encoding="utf-8") as f:
            f.write("version: v1\n  parent: [unbalanced\n")

        with pytest.raises(ValueError, match=r"YAML"):
            load_prompt("v1")

    def test_non_mapping_yaml_raises_valueerror(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        path = tmp_path / "v1.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(["not", "a", "mapping"], f)

        with pytest.raises(ValueError, match=r"mapping"):
            load_prompt("v1")


class TestFingerprint:
    def test_changing_system_text_changes_fingerprint(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        _write_prompt(tmp_path, "v1", system="Original system instruction.")
        original_fp = load_prompt("v1").fingerprint

        _write_prompt(tmp_path, "v1", system="Edited system instruction.")
        edited_fp = load_prompt("v1").fingerprint

        assert original_fp != edited_fp

    def test_changing_user_template_changes_fingerprint(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        _write_prompt(tmp_path, "v1", user_template="Feedback: {text}")
        original_fp = load_prompt("v1").fingerprint

        _write_prompt(tmp_path, "v1", user_template="Customer said: {text}")
        edited_fp = load_prompt("v1").fingerprint

        assert original_fp != edited_fp

    def test_reordering_yaml_keys_leaves_fingerprint_unchanged(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        data = {
            "version": "v1",
            "parent": None,
            "changelog": "baseline",
            "system": "Respond with JSON.",
            "user_template": "Feedback: {text}",
        }

        forward_dir = tmp_path / "forward"
        forward_dir.mkdir()
        reversed_dir = tmp_path / "reversed"
        reversed_dir.mkdir()

        with (forward_dir / "v1.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        with (reversed_dir / "v1.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(dict(reversed(list(data.items()))), f, sort_keys=False)

        monkeypatch.setattr(registry, "PROMPTS_DIR", forward_dir)
        forward_fp = load_prompt("v1").fingerprint
        monkeypatch.setattr(registry, "PROMPTS_DIR", reversed_dir)
        reversed_fp = load_prompt("v1").fingerprint

        assert forward_fp == reversed_fp

    def test_fingerprint_is_12_lowercase_hex_chars(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        _write_prompt(tmp_path, "v1")

        fp = load_prompt("v1").fingerprint

        assert len(fp) == 12
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_is_deterministic_across_loads(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        _write_prompt(tmp_path, "v1")

        fp1 = load_prompt("v1").fingerprint
        fp2 = load_prompt("v1").fingerprint

        assert fp1 == fp2


class TestListVersions:
    def test_returns_sorted_versions_for_synthetic_files(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        _write_prompt(tmp_path, "v3")
        _write_prompt(tmp_path, "v1")
        _write_prompt(tmp_path, "v2")

        assert list_versions() == ["v1", "v2", "v3"]

    def test_returns_empty_list_when_no_prompt_files(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)

        assert list_versions() == []

    def test_ignores_non_yaml_files(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        _write_prompt(tmp_path, "v1")
        (tmp_path / "README.md").write_text("not a prompt", encoding="utf-8")

        assert list_versions() == ["v1"]


class TestRealPromptFiles:
    """Exercise the actual prompts/v1.yaml and prompts/v2.yaml shipped in the repo."""

    def test_load_v1_loads_successfully_with_all_fields_populated(self):
        pv = load_prompt("v1")

        assert pv.version == "v1"
        assert pv.parent is None
        assert pv.changelog.strip()
        assert pv.system.strip()
        assert pv.user_template.strip()
        assert pv.fingerprint
        assert len(pv.fingerprint) == 12

    def test_load_v2_loads_successfully_with_all_fields_populated(self):
        pv = load_prompt("v2")

        assert pv.version == "v2"
        assert pv.changelog.strip()
        assert pv.system.strip()
        assert pv.user_template.strip()
        assert pv.fingerprint
        assert len(pv.fingerprint) == 12

    def test_v2_parent_is_v1(self):
        pv = load_prompt("v2")

        assert pv.parent == "v1"

    def test_v1_system_mentions_every_class_in_src_dataset_classes(self):
        pv = load_prompt("v1")

        for label in CLASSES:
            assert label in pv.system

    def test_v2_system_mentions_every_class_in_src_dataset_classes(self):
        pv = load_prompt("v2")

        for label in CLASSES:
            assert label in pv.system

    def test_v1_and_v2_have_different_fingerprints(self):
        assert load_prompt("v1").fingerprint != load_prompt("v2").fingerprint

    def test_v1_and_v2_share_the_same_user_template(self):
        assert load_prompt("v1").user_template == load_prompt("v2").user_template

    def test_list_versions_returns_v1_and_v2(self):
        assert list_versions() == ["v1", "v2"]
