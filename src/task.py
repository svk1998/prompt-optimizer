"""The classification task definition — the one file to edit to retarget the pipeline.

Everything else (dataset validation, prediction parsing, metrics, the gate) reads the
task from here. To point the pipeline at a different single-label classification task,
change these two values (and supply matching prompts + dataset); no other module needs
to change.

- CLASSES: the exact set of allowed labels.
- RESPONSE_KEY: the JSON field the model must return, i.e. {"<RESPONSE_KEY>": "<label>"}.

Scope note: this centralizes the *label set* and the *response contract*. It does not
turn the pipeline into a general LLM-eval framework — parsing and metrics still assume
one categorical label per example. A non-classification task (summarization,
extraction, ...) needs a different scoring layer, not just a different task.py.
"""
from __future__ import annotations

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

RESPONSE_KEY: str = "category"
