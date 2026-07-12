# PLAN.md — Data-driven prompt optimization pipeline

## Goal

Build a framework-agnostic prompt optimization pipeline for a VoC (Voice of Customer)
classification task: versioned prompts are evaluated against a frozen golden dataset,
scored with hand-rolled classification metrics, and passed through a regression gate
that promotes or reverts each candidate. Stretch goal: a mini-OPRO optimizer that
generates candidates automatically.

This is a learning project. Prioritize readable, well-factored code over cleverness.
Every module must be independently runnable with a smoke test. Do NOT use LangChain,
LlamaIndex, or any orchestration framework — direct SDK calls only.

## Tech constraints

- Python 3.11+, package layout under `src/` (plain modules, no packaging ceremony needed)
- LLM: Groq SDK (`groq` package), model `llama-3.3-70b-versatile`, `temperature=0`
- API key from `GROQ_API_KEY` env var via `python-dotenv`; never hardcode
- `pyyaml` for prompt files
- `scikit-learn` as a DEV dependency only — used exclusively in tests to verify the
  hand-rolled metrics. Production code paths must not import sklearn.
- No other heavy dependencies. Standard library preferred.
- Type hints everywhere. Dataclasses or Pydantic for data types (pick one, be consistent).

## Repo layout

```
prompt_optimizer/
├── datasets/
│   └── voc_golden_v1.jsonl      # PROVIDED — do not modify contents, treat as frozen
├── prompts/
│   ├── v1.yaml
│   └── v2.yaml
├── runs/                        # run records, append-only JSON, gitignored optional
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── registry.py
│   ├── runner.py
│   ├── gate.py
│   └── optimize.py              # phase 6, stretch
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_gate.py
├── main.py
├── requirements.txt
└── .env.example
```

## Dataset (provided)

`datasets/voc_golden_v1.jsonl` — 48 examples, JSONL, one object per line:
`{"id", "text", "label", "difficulty", "note"?}`.
Classes (exactly these six):
`bug_report, feature_request, billing_issue, ux_complaint, performance_issue, praise`.
`difficulty` is `easy` or `hard`. `note` is human documentation of why a hard case is
hard — it must NEVER be sent to the model.

---

## Phase 1 — src/dataset.py

- `CLASSES: frozenset[str]` — the six labels above.
- `EvalExample` — frozen dataclass: `id, text, label, difficulty, note: str | None = None`.
- `load_dataset(path) -> list[EvalExample]` — parse JSONL. Fail loudly at load time with
  `ValueError` including the offending line number for: duplicate ids, unknown labels,
  empty text, invalid difficulty.
- `dataset_fingerprint(examples) -> str` — first 12 hex chars of SHA-256 over a canonical
  serialization: sort examples by `id`, serialize each with `json.dumps(..., sort_keys=True)`.
  Line order in the file must NOT affect the fingerprint. Exclude nothing — `note` is part
  of identity.
- `class_distribution(examples) -> dict[str, int]` and `summarize(examples)` printing
  per-class counts and easy/hard split.
- `__main__` block: `python -m src.dataset datasets/voc_golden_v1.jsonl` prints count,
  fingerprint, distribution.

**Acceptance:** tests prove (a) reordering lines leaves fingerprint unchanged,
(b) editing one label changes it, (c) duplicate id / bad label / empty text each raise
with line number.

## Phase 2 — src/metrics.py

Hand-rolled, pure functions, no sklearn imports in this module.

- `confusion_matrix(y_true, y_pred, classes) -> dict[str, dict[str, int]]` —
  matrix[true][pred]. Predictions outside `classes` (e.g. `PARSE_FAILURE`) count in a
  dedicated extra column so failed parses are visible, not silently dropped.
- `per_class_prf(cm) -> dict[str, {precision, recall, f1, support}]` — define 0/0 as 0.0.
- `accuracy(cm) -> float`
- `macro_f1(per_class) -> float` — unweighted mean over the six real classes.
- `micro_f1(cm) -> float`
- `format_report(...) -> str` — readable text report: per-class table, macro/micro,
  accuracy, parse-failure rate.

**Acceptance:** `tests/test_metrics.py` checks against `sklearn.metrics`
(`precision_recall_fscore_support`, `accuracy_score`) on 3+ fixtures including one with
an absent class (0/0 case) and one with parse failures.

## Phase 3 — src/registry.py

Prompts are YAML files in `prompts/`:

```yaml
version: v1
parent: null            # or the version this was derived from
changelog: "baseline"
system: |
  <system instruction; must demand a JSON object {"category": "<label>"} and enumerate
  the six allowed labels>
user_template: |
  Feedback: {text}
```

- `PromptVersion` dataclass mirroring the YAML plus `fingerprint` (same canonical-hash
  technique as the dataset).
- `load_prompt(version) -> PromptVersion`, `list_versions() -> list[str]`.
- Write `prompts/v1.yaml`: a deliberately plain baseline (short instruction, label list,
  JSON output demand, no few-shot examples). It should be mediocre on purpose — the
  pipeline needs headroom to demonstrate improvement.
- Write `prompts/v2.yaml`: one single change vs v1 (e.g. add 2–3 few-shot examples
  covering hard-case patterns, OR add per-label definitions — pick ONE change, record it
  in `changelog`, set `parent: v1`).

## Phase 4 — src/runner.py

- `run_eval(prompt: PromptVersion, examples, model, n_repeats=1) -> RunRecord`.
- Serial or small-thread-pool execution with basic exponential backoff on 429s
  (free-tier Groq rate limits are real — default to modest concurrency).
- `temperature=0`, fixed `max_tokens`, pinned model string recorded in the run.
- Parsing: attempt strict `json.loads`; on failure, one lenient fallback (extract first
  `{...}` block via regex). If still unparseable or label not in `CLASSES`, prediction
  becomes the sentinel `PARSE_FAILURE`. Track parse-failure rate as a first-class metric.
- `RunRecord` (serialized to `runs/<timestamp>_<prompt_version>.json`):
  `run_id, timestamp, prompt_version, prompt_fingerprint, dataset_fingerprint, model,
  params, per_example: [{id, expected, predicted, raw_output, latency_ms,
  prompt_tokens, completion_tokens}], metrics: {...}, totals: {latency, tokens, est_cost}`.
- Metrics computed via Phase 2 and embedded in the record. Also compute metrics on the
  easy/hard slices separately.
- `__main__`: `python -m src.runner --prompt v1` runs the full dataset and prints the
  Phase 2 report.

**Acceptance:** a `--dry-run` flag that uses a mock client (returns canned outputs,
including one malformed) so the whole path is testable without an API key.

## Phase 5 — src/gate.py

- `GateConfig` defaults: `min_macro_f1_gain=0.005` (calibrate later),
  `max_per_class_f1_drop=0.05`, `min_parse_rate=0.99`,
  `max_cost_multiplier=1.5`, `max_latency_multiplier=1.5`.
- `evaluate_gate(candidate: RunRecord, baseline: RunRecord, cfg) -> GateDecision` where
  `GateDecision` = `{verdict: "PROMOTE" | "REVERT", reasons: list[str]}`. Every rule
  contributes a human-readable reason line whether it passed or failed.
- Refuse to compare runs with different `dataset_fingerprint` or model — raise, don't warn.
- Baseline pointer: a small `runs/baseline.json` file `{prompt_version, run_id}` updated
  only on PROMOTE.
- `main.py` orchestrates: load dataset → run candidate → load baseline run → gate →
  print decision with reasons → update baseline pointer if promoted. First-ever run of
  v1 bootstraps the baseline.

**Acceptance:** `tests/test_gate.py` with synthetic RunRecords covering: clean promote,
aggregate-up-but-one-class-collapsed (must revert), parse-rate violation (must revert),
mismatched fingerprints (must raise).

## Phase 6 (stretch) — src/optimize.py — mini-OPRO

- `propose_candidate(history: list[tuple[PromptVersion, float]], k_last=5) -> str` —
  builds a meta-prompt containing the last k (system instruction, macro-F1) pairs sorted
  ascending by score, plus 3–5 currently-misclassified examples from the latest run, and
  asks the optimizer model (same Groq model is fine) to propose ONE improved system
  instruction. Parse it, save as the next `prompts/vN.yaml` with
  `changelog: "opro-generated"`.
- `optimize_loop(iterations=5)`: propose → run → gate → (promote|revert) → repeat.
  Reverted candidates stay in history — failures are signal for the optimizer.
- Guardrails: cap iterations, stop early if two consecutive candidates revert, print a
  final table of all versions with scores.

## Cross-cutting rules

1. Never send `note` or `label` fields to the model during evaluation.
2. Every run record must carry both fingerprints (prompt + dataset). No fingerprint,
   no comparison.
3. All randomness seeded; anything nondeterministic (API output) is recorded raw.
4. Costs matter: estimate token cost per run and surface it in the report.
5. Keep functions small; each module's `__main__` smoke test must work in isolation.
6. After each phase, STOP and summarize: what was built, how it was verified, and one
   design decision worth explaining — the user is building this to learn, and will review
   each phase before you proceed to the next.

## Suggested commit sequence

One commit per phase, message format: `phase-N: <module> — <one-line summary>`.
