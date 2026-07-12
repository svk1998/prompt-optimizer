"""LLM eval runner: executes a PromptVersion against the golden dataset via an LLM
client, parses responses into predicted labels, and produces a serialized RunRecord.

Usage:
    python -m src.runner --prompt v1
    python -m src.runner --prompt v1 --dry-run   # no API key needed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, TypeVar

import groq
from dotenv import load_dotenv

from src.dataset import EvalExample, dataset_fingerprint, load_dataset
from src.metrics import (
    PARSE_FAILURE,
    accuracy,
    confusion_matrix,
    format_report,
    macro_f1,
    micro_f1,
    per_class_prf,
)
from src.registry import PromptVersion, load_prompt
from src.task import CLASSES, RESPONSE_KEY

_T = TypeVar("_T")

MODEL = "llama-3.3-70b-versatile"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "voc_golden_v1.jsonl"


@dataclass(frozen=True)
class Completion:
    """A single LLM response: raw text plus the token counts billed for it."""

    content: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    """Anything that can turn a (system, user) prompt pair into a Completion.

    GroqClient implements this against the real Groq API; MockClient implements it
    with canned responses for --dry-run and for tests, so src.runner's evaluation
    logic never has to know or care which one it's talking to.
    """

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, max_tokens: int
    ) -> Completion: ...


class MockClient:
    """Cycles through a fixed list of canned Completions, wrapping around at the end.

    Used for --dry-run (exercise the full pipeline without an API key) and for tests
    (deterministic, no network).
    """

    def __init__(self, responses: Sequence[Completion]):
        if not responses:
            raise ValueError("responses must not be empty")
        self._responses = list(responses)
        self._calls = 0

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, max_tokens: int
    ) -> Completion:
        response = self._responses[self._calls % len(self._responses)]
        self._calls += 1
        return response


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_prediction(raw_output: str) -> str:
    """Parse a raw model response into one of CLASSES, or PARSE_FAILURE.

    Strategy: try strict json.loads first; on failure, fall back to extracting the
    first {...} block via regex and parsing that. Either way, the result must be a
    JSON object with a string RESPONSE_KEY field whose value is one of CLASSES —
    anything else (wrong shape, missing key, unknown label) is PARSE_FAILURE.
    """
    obj = _try_json_loads(raw_output)
    if obj is None:
        match = _JSON_OBJECT_RE.search(raw_output)
        if match is not None:
            obj = _try_json_loads(match.group(0))

    if not isinstance(obj, dict):
        return PARSE_FAILURE

    label = obj.get(RESPONSE_KEY)
    if not isinstance(label, str) or label not in CLASSES:
        return PARSE_FAILURE

    return label


def _try_json_loads(text: str) -> object | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _with_retry(
    fn: Callable[[], _T],
    *,
    retry_on: tuple[type[BaseException], ...],
    max_retries: int,
    base_delay: float,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Call fn(), retrying up to max_retries total attempts on any error in retry_on.

    Delay between attempts doubles each time starting at base_delay (exponential
    backoff). `sleep` is injectable so tests can assert on backoff timing without
    real delays. Errors not in `retry_on` propagate immediately, unretried.
    """
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except retry_on:
            if attempt == max_retries:
                raise
            sleep(delay)
            delay *= 2


@dataclass(frozen=True)
class PerExampleResult:
    id: str
    expected: str
    predicted: str
    raw_output: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    timestamp: str
    prompt_version: str
    prompt_fingerprint: str
    dataset_fingerprint: str
    model: str
    params: dict[str, Any]
    per_example: list[PerExampleResult]
    metrics: dict[str, Any]
    totals: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metrics_summary(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Confusion-matrix-derived metrics for one slice of results (overall/easy/hard).

    Built on top of src.metrics rather than duplicating any averaging logic — this is
    purely "call the Phase 2 functions and shape the result for JSON serialization".
    """
    cm = confusion_matrix(y_true, y_pred, CLASSES)
    per_class = per_class_prf(cm)
    total = sum(sum(row.values()) for row in cm.values())
    parse_failures = sum(row.get(PARSE_FAILURE, 0) for row in cm.values())
    return {
        "accuracy": accuracy(cm),
        "macro_f1": macro_f1(per_class),
        "micro_f1": micro_f1(cm),
        "parse_failure_rate": parse_failures / total if total else 0.0,
        "support": total,
        "per_class": per_class,
    }


def _timestamp_now() -> str:
    """Millisecond-precision UTC timestamp, used as both RunRecord.timestamp and the
    run_id/filename stem. Second-precision was not enough: back-to-back run_eval calls
    (e.g. Phase 6's optimize loop, or two dry-runs in a test) can land in the same
    second and would otherwise silently collide on the same runs/<run_id>.json file.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"


def run_eval(
    prompt: PromptVersion,
    examples: list[EvalExample],
    model: str,
    client: LLMClient,
    *,
    n_repeats: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 20,
    max_workers: int = 2,
) -> RunRecord:
    """Evaluate `prompt` against `examples` via `client`, producing a full RunRecord.

    Each example is sent n_repeats times (consecutively) through client.complete().
    Execution runs on a small thread pool (max_workers) — modest by default since
    free-tier Groq rate limits are real. Retry/backoff on 429s is the client's own
    concern (see GroqClient, which wraps every call in _with_retry); run_eval treats
    `client` as an opaque LLMClient and never sees rate-limit errors itself. Only
    `text` is ever sent to the model; `label` and `note` stay local.
    """
    tasks = [ex for ex in examples for _ in range(n_repeats)]

    def _run_one(ex: EvalExample) -> PerExampleResult:
        user_content = prompt.user_template.format(text=ex.text)
        start = time.perf_counter()
        completion = client.complete(
            system=prompt.system,
            user=user_content,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        predicted = parse_prediction(completion.content)
        return PerExampleResult(
            id=ex.id,
            expected=ex.label,
            predicted=predicted,
            raw_output=completion.content,
            latency_ms=latency_ms,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_run_one, tasks))

    difficulty_by_id = {ex.id: ex.difficulty for ex in examples}
    easy_results = [r for r in results if difficulty_by_id[r.id] == "easy"]
    hard_results = [r for r in results if difficulty_by_id[r.id] == "hard"]

    metrics = {
        "overall": _metrics_summary([r.expected for r in results], [r.predicted for r in results]),
        "easy": _metrics_summary(
            [r.expected for r in easy_results], [r.predicted for r in easy_results]
        ),
        "hard": _metrics_summary(
            [r.expected for r in hard_results], [r.predicted for r in hard_results]
        ),
    }

    totals = {
        "latency_ms": sum(r.latency_ms for r in results),
        "prompt_tokens": sum(r.prompt_tokens for r in results),
        "completion_tokens": sum(r.completion_tokens for r in results),
        "est_cost_usd": _estimate_cost(
            model,
            sum(r.prompt_tokens for r in results),
            sum(r.completion_tokens for r in results),
        ),
    }

    timestamp = _timestamp_now()
    return RunRecord(
        run_id=f"{timestamp}_{prompt.version}",
        timestamp=timestamp,
        prompt_version=prompt.version,
        prompt_fingerprint=prompt.fingerprint,
        dataset_fingerprint=dataset_fingerprint(examples),
        model=model,
        params={"temperature": temperature, "max_tokens": max_tokens, "n_repeats": n_repeats},
        per_example=results,
        metrics=metrics,
        totals=totals,
    )


# Approximate Groq on-demand pricing, USD per 1M tokens: (input_price, output_price).
# Verify against https://groq.com/pricing before relying on this for real budgeting —
# it's here so runs surface a cost estimate, not as an authoritative price source.
_PRICING_USD_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = _PRICING_USD_PER_MILLION_TOKENS.get(model, (0.0, 0.0))
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


class GroqClient:
    """LLMClient backed by the real Groq SDK.

    Every call is wrapped in exponential backoff on 429s (RateLimitError) — free-tier
    Groq rate limits are real, and this is the one place in the module that talks to
    the network, so this is where that concern lives.
    """

    def __init__(self, api_key: str, *, max_retries: int = 5, base_delay: float = 1.0):
        self._client = groq.Groq(api_key=api_key)
        self._max_retries = max_retries
        self._base_delay = base_delay

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, max_tokens: int
    ) -> Completion:
        def _call():
            return self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        response = _with_retry(
            _call,
            retry_on=(groq.RateLimitError,),
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )
        choice = response.choices[0]
        usage = response.usage
        return Completion(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


def save_run_record(record: RunRecord) -> Path:
    """Serialize `record` to runs/<record.run_id>.json, return the path.

    `run_id` is `<timestamp>_<prompt_version>` precisely so the filename can be
    reconstructed from a baseline pointer's run_id alone (see src.gate) without
    needing to store a separate path.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{record.run_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)
    return path


def load_run_record(path: Path) -> RunRecord:
    """Parse a JSON file written by save_run_record back into a RunRecord."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    per_example = [PerExampleResult(**row) for row in data["per_example"]]
    return RunRecord(
        run_id=data["run_id"],
        timestamp=data["timestamp"],
        prompt_version=data["prompt_version"],
        prompt_fingerprint=data["prompt_fingerprint"],
        dataset_fingerprint=data["dataset_fingerprint"],
        model=data["model"],
        params=data["params"],
        per_example=per_example,
        metrics=data["metrics"],
        totals=data["totals"],
    )


# Canned responses for --dry-run: enough variety to exercise correct predictions, a
# misclassification, and (deliberately) one malformed output so PARSE_FAILURE shows up
# in the report without needing a live API key.
DRY_RUN_RESPONSES = [
    Completion(content='{"category": "bug_report"}', prompt_tokens=40, completion_tokens=8),
    Completion(content='{"category": "praise"}', prompt_tokens=38, completion_tokens=7),
    Completion(content='{"category": "feature_request"}', prompt_tokens=42, completion_tokens=8),
    Completion(content="sorry, I cannot help with that", prompt_tokens=35, completion_tokens=6),
]


def build_client(dry_run: bool) -> LLMClient:
    """Build the LLMClient a CLI entrypoint should use.

    Shared by src.runner's own __main__ and main.py, so "dry-run uses MockClient,
    otherwise require GROQ_API_KEY and use the real Groq SDK" is defined exactly once.
    """
    if dry_run:
        return MockClient(DRY_RUN_RESPONSES)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set (see .env.example) — or pass --dry-run")
    return GroqClient(api_key=api_key)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a prompt version against the golden dataset.")
    parser.add_argument("--prompt", required=True, help="prompt version to evaluate, e.g. v1")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument(
        "--dry-run", action="store_true", help="use a mock client instead of the real Groq API"
    )
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> RunRecord:
    args = _parse_args(argv)
    load_dotenv()

    examples = load_dataset(args.dataset)
    prompt = load_prompt(args.prompt)
    client = build_client(args.dry_run)

    record = run_eval(
        prompt,
        examples,
        MODEL,
        client,
        n_repeats=args.n_repeats,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
    )

    path = save_run_record(record)
    print(f"Saved run record to {path}")

    cm = confusion_matrix(
        [r.expected for r in record.per_example], [r.predicted for r in record.per_example], CLASSES
    )
    print(format_report(cm))
    print()
    print(
        f"totals: latency_ms={record.totals['latency_ms']:.0f} "
        f"prompt_tokens={record.totals['prompt_tokens']} "
        f"completion_tokens={record.totals['completion_tokens']} "
        f"est_cost_usd={record.totals['est_cost_usd']:.6f}"
    )

    return record


if __name__ == "__main__":
    main()
