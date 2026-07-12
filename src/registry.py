"""Prompt version registry: loads YAML prompt files from prompts/ into typed
PromptVersion instances.

Nothing in this module calls an LLM or performs any templating — it just parses and
validates the YAML, and exposes the raw `system` / `user_template` strings for later
phases (`src.runner`) to use.

Usage:
    python -m src.registry
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_REQUIRED_FIELDS = ("version", "parent", "changelog", "system", "user_template")
_NON_NULLABLE_FIELDS = ("version", "changelog", "system", "user_template")


@dataclass(frozen=True)
class PromptVersion:
    version: str
    parent: str | None
    changelog: str
    system: str
    user_template: str
    fingerprint: str


def _fingerprint(
    version: str, parent: str | None, changelog: str, system: str, user_template: str
) -> str:
    """Canonical-hash fingerprint over the prompt's YAML content fields.

    Same technique as src.dataset.dataset_fingerprint: first 12 hex chars of SHA-256
    over a canonical json.dumps(..., sort_keys=True) serialization, so key order in the
    source YAML never affects the result but any content change does.

    This is a small private helper rather than a shared module with src.dataset:
    dataset_fingerprint canonicalizes a *list* of records (sort by id, join per-line
    JSON), while this hashes a single flat record — the underlying "sort_keys JSON +
    sha256[:12]" idea is the same, but the two shapes don't share enough real logic to
    justify a shared module for exactly two callers.
    """
    canonical = json.dumps(
        {
            "version": version,
            "parent": parent,
            "changelog": changelog,
            "system": system,
            "user_template": user_template,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:12]


def load_prompt(version: str) -> PromptVersion:
    """Load prompts/<version>.yaml into a PromptVersion, computing its fingerprint.

    Raises FileNotFoundError (including the file path) if the file does not exist.
    Raises ValueError (including the file path) if the file fails to parse as a YAML
    mapping, is missing a required field, or has a required field that is empty
    (`parent` is the one exception: it may legitimately be null/absent for a baseline
    prompt with no ancestor, but if present it must be a non-empty string).
    """
    path = PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"prompt version {version!r} not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: invalid YAML ({exc})") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    missing = [key for key in _REQUIRED_FIELDS if key not in data]
    if missing:
        raise ValueError(f"{path}: missing required field(s) {missing}")

    empty = [key for key in _NON_NULLABLE_FIELDS if not str(data[key]).strip()]
    parent = data["parent"]
    if parent is not None and not str(parent).strip():
        empty.append("parent")
    if empty:
        raise ValueError(f"{path}: empty required field(s) {empty}")

    version_field = str(data["version"])
    if version_field != version:
        raise ValueError(
            f"{path}: 'version' field {version_field!r} does not match filename {version!r}"
        )

    changelog = str(data["changelog"])
    system = str(data["system"])
    user_template = str(data["user_template"])

    fingerprint = _fingerprint(version_field, parent, changelog, system, user_template)

    return PromptVersion(
        version=version_field,
        parent=parent,
        changelog=changelog,
        system=system,
        user_template=user_template,
        fingerprint=fingerprint,
    )


def list_versions() -> list[str]:
    """Scan prompts/ for *.yaml files and return their version strings, sorted."""
    if not PROMPTS_DIR.exists():
        return []
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.yaml"))


def main() -> None:
    versions = list_versions()
    if not versions:
        print(f"No prompt versions found in {PROMPTS_DIR}")
        return

    for version in versions:
        pv = load_prompt(version)
        print(f"{pv.version}  parent={pv.parent}  fingerprint={pv.fingerprint}")
        print(f"  changelog: {pv.changelog}")


if __name__ == "__main__":
    main()
