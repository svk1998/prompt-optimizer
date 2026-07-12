"""Tests for src.task — the single place the classification task is defined.

These lock in the goal of centralization: retargeting the pipeline to a different
classification task must be a one-file change (edit src/task.py), which is only true
if every other module reads the task definition from here rather than keeping its own
copy.
"""
from __future__ import annotations

import json

import src.dataset
import src.gate
import src.runner
import src.task
from src.runner import parse_prediction


def test_classes_are_the_six_voc_labels():
    assert src.task.CLASSES == frozenset(
        {
            "bug_report",
            "feature_request",
            "billing_issue",
            "ux_complaint",
            "performance_issue",
            "praise",
        }
    )


def test_dataset_runner_gate_share_one_class_set():
    # `is`, not `==`: proves there is exactly one frozenset object, so no module can
    # drift out of sync with a stale private copy.
    assert src.dataset.CLASSES is src.task.CLASSES
    assert src.runner.CLASSES is src.task.CLASSES
    assert src.gate.CLASSES is src.task.CLASSES


def test_response_key_is_defined():
    assert src.task.RESPONSE_KEY == "category"


def test_parse_prediction_reads_the_configured_response_key():
    # Build the JSON from the constant, so if RESPONSE_KEY and the parser ever drift
    # apart this fails. Both read the same source, so they stay consistent.
    raw = json.dumps({src.task.RESPONSE_KEY: "praise"})
    assert parse_prediction(raw) == "praise"
