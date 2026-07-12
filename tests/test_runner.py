"""Tests for src.runner — LLM eval execution, parsing, retry, and run records."""
from __future__ import annotations

import json
from pathlib import Path

import groq
import httpx
import pytest

import src.runner as runner
from src.dataset import CLASSES, EvalExample
from src.registry import PromptVersion
from src.runner import parse_prediction


def _prompt(version: str = "v1") -> PromptVersion:
    return PromptVersion(
        version=version,
        parent=None,
        changelog="test prompt",
        system="Respond with JSON only.",
        user_template="Feedback: {text}",
        fingerprint="fingerprint01",
    )


def _example(id: str, text: str = "some feedback", label: str = "bug_report", difficulty: str = "easy") -> EvalExample:
    return EvalExample(id=id, text=text, label=label, difficulty=difficulty, note=None)


def _completion(content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> "runner.Completion":
    return runner.Completion(content=content, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class TestParsePrediction:
    def test_strict_json_with_valid_label_returns_label(self):
        assert parse_prediction('{"category": "bug_report"}') == "bug_report"

    def test_json_wrapped_in_surrounding_text_extracts_object(self):
        raw = 'Sure, here you go:\n{"category": "praise"}\nHope that helps!'
        assert parse_prediction(raw) == "praise"

    def test_completely_malformed_text_returns_parse_failure(self):
        assert parse_prediction("not json at all") == runner.PARSE_FAILURE

    def test_valid_json_with_unknown_label_returns_parse_failure(self):
        assert parse_prediction('{"category": "not_a_real_class"}') == runner.PARSE_FAILURE

    def test_valid_json_array_returns_parse_failure(self):
        assert parse_prediction('["bug_report"]') == runner.PARSE_FAILURE

    def test_valid_json_missing_category_key_returns_parse_failure(self):
        assert parse_prediction('{"label": "bug_report"}') == runner.PARSE_FAILURE

    def test_category_value_wrong_type_returns_parse_failure(self):
        assert parse_prediction('{"category": 5}') == runner.PARSE_FAILURE

    def test_every_real_class_round_trips(self):
        for label in CLASSES:
            assert parse_prediction(f'{{"category": "{label}"}}') == label


class _FlakyError(Exception):
    """Standalone exception used only to exercise retry logic in tests."""


class TestWithRetry:
    def test_succeeds_immediately_when_no_error_raised(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return "ok"

        result = runner._with_retry(
            fn, retry_on=(_FlakyError,), max_retries=3, base_delay=0.01, sleep=lambda s: None
        )

        assert result == "ok"
        assert calls["n"] == 1

    def test_retries_on_matching_error_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _FlakyError("rate limited")
            return "ok"

        sleeps: list[float] = []
        result = runner._with_retry(
            flaky, retry_on=(_FlakyError,), max_retries=5, base_delay=0.01, sleep=sleeps.append
        )

        assert result == "ok"
        assert calls["n"] == 3
        assert sleeps == [0.01, 0.02]

    def test_raises_after_exhausting_max_retries(self):
        def always_fails():
            raise _FlakyError("still rate limited")

        sleeps: list[float] = []
        try:
            runner._with_retry(
                always_fails,
                retry_on=(_FlakyError,),
                max_retries=3,
                base_delay=0.01,
                sleep=sleeps.append,
            )
            assert False, "expected _FlakyError to propagate"
        except _FlakyError:
            pass

        assert len(sleeps) == 2  # 3 attempts total, backoff sleeps between attempts only

    def test_non_matching_error_propagates_immediately_without_retry(self):
        calls = {"n": 0}

        def raises_other():
            calls["n"] += 1
            raise ValueError("not the retryable kind")

        try:
            runner._with_retry(
                raises_other, retry_on=(_FlakyError,), max_retries=3, base_delay=0.01, sleep=lambda s: None
            )
            assert False, "expected ValueError to propagate"
        except ValueError:
            pass

        assert calls["n"] == 1


class TestMockClient:
    def test_cycles_through_responses_in_order(self):
        responses = [
            runner.Completion(content="a", prompt_tokens=1, completion_tokens=1),
            runner.Completion(content="b", prompt_tokens=2, completion_tokens=2),
        ]
        client = runner.MockClient(responses)

        first = client.complete(system="s", user="u", model="m", temperature=0, max_tokens=10)
        second = client.complete(system="s", user="u", model="m", temperature=0, max_tokens=10)
        third = client.complete(system="s", user="u", model="m", temperature=0, max_tokens=10)

        assert first.content == "a"
        assert second.content == "b"
        assert third.content == "a"  # wraps around

    def test_raises_on_empty_responses(self):
        try:
            runner.MockClient([])
            assert False, "expected ValueError for empty responses"
        except ValueError:
            pass


class TestRunEval:
    def test_run_record_carries_identifying_fields(self):
        prompt = _prompt("v1")
        examples = [_example("e1", label="bug_report")]
        client = runner.MockClient([_completion('{"category": "bug_report"}')])

        record = runner.run_eval(prompt, examples, "test-model", client, max_workers=1)

        assert record.prompt_version == "v1"
        assert record.prompt_fingerprint == prompt.fingerprint
        assert record.dataset_fingerprint  # non-empty, computed from examples
        assert record.model == "test-model"
        assert record.params["temperature"] == 0.0
        assert record.params["n_repeats"] == 1
        assert record.run_id
        assert record.timestamp

    def test_per_example_results_match_examples_and_predictions(self):
        prompt = _prompt()
        examples = [
            _example("e1", text="it crashed", label="bug_report"),
            _example("e2", text="great app!", label="praise"),
        ]
        client = runner.MockClient(
            [_completion('{"category": "bug_report"}'), _completion('{"category": "praise"}')]
        )

        record = runner.run_eval(prompt, examples, "test-model", client, max_workers=1)

        assert [r.id for r in record.per_example] == ["e1", "e2"]
        assert [r.expected for r in record.per_example] == ["bug_report", "praise"]
        assert [r.predicted for r in record.per_example] == ["bug_report", "praise"]
        assert record.per_example[0].raw_output == '{"category": "bug_report"}'
        assert record.per_example[0].prompt_tokens == 10
        assert record.per_example[0].completion_tokens == 5
        assert record.per_example[0].latency_ms >= 0

    def test_unparseable_response_becomes_parse_failure(self):
        prompt = _prompt()
        examples = [_example("e1", label="bug_report")]
        client = runner.MockClient([_completion("garbage, not json")])

        record = runner.run_eval(prompt, examples, "test-model", client, max_workers=1)

        assert record.per_example[0].predicted == runner.PARSE_FAILURE

    def test_overall_metrics_reflect_all_examples(self):
        prompt = _prompt()
        examples = [
            _example("e1", label="bug_report"),
            _example("e2", label="praise"),
        ]
        client = runner.MockClient(
            [_completion('{"category": "bug_report"}'), _completion('{"category": "feature_request"}')]
        )

        record = runner.run_eval(prompt, examples, "test-model", client, max_workers=1)

        overall = record.metrics["overall"]
        assert overall["accuracy"] == 0.5
        assert set(overall["per_class"].keys()) == CLASSES

    def test_metrics_are_split_by_difficulty(self):
        prompt = _prompt()
        examples = [
            _example("e1", label="bug_report", difficulty="easy"),
            _example("e2", label="praise", difficulty="hard"),
        ]
        client = runner.MockClient(
            [_completion('{"category": "bug_report"}'), _completion('{"category": "billing_issue"}')]
        )

        record = runner.run_eval(prompt, examples, "test-model", client, max_workers=1)

        assert record.metrics["easy"]["accuracy"] == 1.0
        assert record.metrics["hard"]["accuracy"] == 0.0

    def test_n_repeats_runs_each_example_multiple_times(self):
        prompt = _prompt()
        examples = [_example("e1", label="bug_report")]
        client = runner.MockClient([_completion('{"category": "bug_report"}')])

        record = runner.run_eval(prompt, examples, "test-model", client, n_repeats=3, max_workers=1)

        assert len(record.per_example) == 3
        assert all(r.id == "e1" for r in record.per_example)
        assert record.params["n_repeats"] == 3

    def test_totals_sum_latency_and_tokens_across_examples(self):
        prompt = _prompt()
        examples = [_example("e1"), _example("e2")]
        client = runner.MockClient(
            [
                _completion('{"category": "bug_report"}', prompt_tokens=10, completion_tokens=5),
                _completion('{"category": "bug_report"}', prompt_tokens=20, completion_tokens=8),
            ]
        )

        record = runner.run_eval(prompt, examples, "test-model", client, max_workers=1)

        assert record.totals["prompt_tokens"] == 30
        assert record.totals["completion_tokens"] == 13
        assert record.totals["latency_ms"] >= 0
        assert "est_cost_usd" in record.totals


class TestSaveRunRecord:
    def test_writes_json_file_named_by_timestamp_and_version(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        prompt = _prompt("v1")
        examples = [_example("e1")]
        client = runner.MockClient([_completion('{"category": "bug_report"}')])
        record = runner.run_eval(prompt, examples, "test-model", client, max_workers=1)

        path = runner.save_run_record(record)

        assert path == tmp_path / f"{record.timestamp}_v1.json"
        assert path.exists()

    def test_written_json_round_trips_expected_fields(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        prompt = _prompt("v1")
        examples = [_example("e1", label="praise")]
        client = runner.MockClient([_completion('{"category": "praise"}')])
        record = runner.run_eval(prompt, examples, "test-model", client, max_workers=1)

        path = runner.save_run_record(record)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["run_id"] == record.run_id
        assert data["prompt_version"] == "v1"
        assert data["dataset_fingerprint"] == record.dataset_fingerprint
        assert data["per_example"][0]["id"] == "e1"
        assert data["per_example"][0]["predicted"] == "praise"
        assert data["metrics"]["overall"]["accuracy"] == 1.0
        assert data["totals"]["prompt_tokens"] == 10


class TestLoadRunRecord:
    def test_round_trips_a_saved_run_record(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        prompt = _prompt("v1")
        examples = [_example("e1", label="praise"), _example("e2", label="bug_report")]
        client = runner.MockClient(
            [_completion('{"category": "praise"}'), _completion('{"category": "bug_report"}')]
        )
        original = runner.run_eval(prompt, examples, "test-model", client, max_workers=1)
        path = runner.save_run_record(original)

        loaded = runner.load_run_record(path)

        assert loaded == original

    def test_run_id_matches_filename_stem(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        prompt = _prompt("v1")
        examples = [_example("e1")]
        client = runner.MockClient([_completion('{"category": "bug_report"}')])
        record = runner.run_eval(prompt, examples, "test-model", client, max_workers=1)
        path = runner.save_run_record(record)

        assert path.stem == record.run_id


class TestGroqClient:
    def _fake_response(self, content: str, prompt_tokens: int = 12, completion_tokens: int = 4):
        class _FakeMessage:
            pass

        class _FakeChoice:
            pass

        class _FakeUsage:
            pass

        class _FakeResponse:
            pass

        message = _FakeMessage()
        message.content = content
        choice = _FakeChoice()
        choice.message = message
        usage = _FakeUsage()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        response = _FakeResponse()
        response.choices = [choice]
        response.usage = usage
        return response

    def test_complete_extracts_content_and_token_usage(self):
        client = runner.GroqClient(api_key="test-key")
        client._client.chat.completions.create = lambda **kwargs: self._fake_response(
            '{"category": "bug_report"}'
        )

        completion = client.complete(system="s", user="u", model="m", temperature=0.0, max_tokens=10)

        assert completion.content == '{"category": "bug_report"}'
        assert completion.prompt_tokens == 12
        assert completion.completion_tokens == 4

    def test_complete_retries_on_rate_limit_then_succeeds(self):
        client = runner.GroqClient(api_key="test-key", max_retries=3, base_delay=0.001)
        calls = {"n": 0}

        def _flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                resp = httpx.Response(
                    status_code=429, request=httpx.Request("POST", "https://api.groq.com/x")
                )
                raise groq.RateLimitError("rate limited", response=resp, body=None)
            return self._fake_response('{"category": "praise"}')

        client._client.chat.completions.create = _flaky

        completion = client.complete(system="s", user="u", model="m", temperature=0.0, max_tokens=10)

        assert completion.content == '{"category": "praise"}'
        assert calls["n"] == 3


class TestMainDryRun:
    def test_dry_run_produces_run_record_without_api_key(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        record = runner.main(["--prompt", "v1", "--dry-run"])

        assert isinstance(record, runner.RunRecord)
        assert record.model == runner.MODEL
        assert len(record.per_example) > 0
        assert any(r.predicted == runner.PARSE_FAILURE for r in record.per_example)
        saved_files = list(tmp_path.glob("*.json"))
        assert len(saved_files) == 1

    def test_without_dry_run_and_no_api_key_raises_clear_error(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        with pytest.raises(SystemExit, match=r"GROQ_API_KEY"):
            runner.main(["--prompt", "v1"])

    def test_dry_run_prints_cost_and_token_totals(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        runner.main(["--prompt", "v1", "--dry-run"])

        out = capsys.readouterr().out
        assert "est_cost_usd" in out
        assert "prompt_tokens" in out
