# 05 — ANALYSIS: Deterministic Profiler + Data Analyst Agent

- **Phase:** 1 (step 1.2) · **Status:** implemented · **Date:** 2026-10-19

## Goal

The ANALYSIS phase: turn raw CSVs into a schema-validated `DataReport` + `FeaturePlan`
that the planner (Phase 2) and ML researcher consume.

## Key design decision: profiler interprets, not codegens

The plan originally said "data analyst writes & runs EDA code". This step implements a
**deterministic profiler + LLM interpretation** instead:

- The profiler (our own tested pandas code) computes: shapes, per-column dtype /
  missingness / cardinality / stats / top values, target analysis (class balance or
  distribution), numeric correlations with the target, a problem-type *guess*.
- The data analyst agent reads the rendered `profile.md` + competition context and
  produces the judgment layer: findings, quality issues, risks, feature ideas.

Why: with a free-tier model (Q0001), LLM-written EDA code is the single biggest
fragility (broken imports, wrong dtypes, hallucinated columns) and gains little —
EDA statistics are commodity work. Judgment is where the LLM adds value. This matches
the plan's risk mitigation ("known-good skeletons instead of writing from scratch").
The sandbox executor (for LLM-written code) arrives in Phase 2 with the researcher.

## Interface

```python
profile_data(run) -> DatasetProfile        # writes eda/profile.json + profile.md

DataAnalystAgent(llm, config, journal, run_id)
  .analyze(profile_md, competition_json) -> DataAnalystOutput

DataAnalystOutput:
  report: DataReport        # problem_type, target_column, summary, key_findings,
                            # data_quality_issues, risks
  feature_plan: FeaturePlan # ideas[]: name, rationale, expected_value, based_on_columns
```

- Schemas live in `slayer/schemas/` (first residents; planner/researcher models join in
  Phase 2 per plan §7).
- Outputs land in `workspace/reports/`: `data_report.json`, `feature_plan.json`,
  `data_report.md` (human-readable render).

## Agent base & journaling

- `Agent` (slayer/agents/base.py): loads its system prompt from `prompts/<name>.md`,
  wraps `LLMClient.complete_structured` with the agent's model config from
  `config/agents.yaml`, and journals **every LLM call** as `agent.call` with model,
  token usage, and cost (feeds per-run budget accounting in Phase 3).
- `agent.output` events record structured summaries (not full payloads — the artifacts
  on disk are the source of truth; the journal stays the audit trail).

## INIT rules probe (the manual-rules issue)

Kaggle deliberately exposes **no API to accept competition rules** — it's a one-time,
per-competition, browser-only legal step (the SDK has no AcceptRules RPC; the website
requires the click). Mitigations implemented:

- INIT now probes data access via `list_files` right after `find_competition`: a 403
  fails the run **at the very start** with the exact rules URL to open, instead of
  failing mid-download (the file list also lands in the `competition.context` event).
- The Phase 4 `resume` command is the second half of the fix: after accepting rules,
  `slayer resume <run>` continues the paused/failed run instead of re-running INIT +
  DATA_COLLECTION from scratch.

## Tests

- Profiler: target detection (binary / multiclass / regression guesses), missingness,
  correlations, files written, missing sample_submission error.
- Agent: prompt loading, system message included, `agent.call` journaled with
  usage/cost, missing prompt file error.
- Analysis phase: full phase on synthetic data with a fake LLM → artifacts + journal.
- Engine/CLI: engine tests inject a handler subset (machine vs phases); CLI run test
  fakes the LLM client and asserts the full pipeline output on disk.

## Retro

Implemented as designed. Live findings (real analyst call on `playground-series-s6e9`,
free reasoning model `minimax/minimax-m3.1-flash:free`):

- **The free model is a reasoning model** — it burns completion tokens on reasoning
  *before* emitting content. With `max_tokens=4096` the analyst call exhausted the
  budget during reasoning and returned `content: null`. Fix: `max_tokens` default
  raised to 16384 (agents.yaml + code default); the LLM client now raises an
  actionable error on empty content with `finish_reason=length` instead of a pydantic
  crash. Full analyst call used ~6.8k tokens.
- **`response_format: json_schema` is not honored by this provider** — the model
  writes its own JSON shape. Fix: `Agent.run_structured` embeds the JSON schema in
  the user message (belt and braces on top of `response_format`), and the client's
  braces-extraction + pydantic validation remain the safety net.
- The deterministic-profiler decision paid off immediately: the report is grounded in
  real statistics (17.5/82.5 class balance, sequential-id warning, missing-evidence
  caveats) with zero codegen fragility. The analyst's output quality on the free model
  is genuinely good.
- One profiler gap surfaced by the analyst itself: it flagged the absence of
  feature-target correlations in the profile. Reality: `Will_Buy_EV` is a string
  target (`Yes`/`No`) so the numeric-correlation branch skipped it. Lesson noted:
  encode binary string targets numerically before correlation (small backlog item
  for the next pass over the profiler).
