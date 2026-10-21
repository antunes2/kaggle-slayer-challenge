# 08 — BASELINE: Researcher Codegen → First Scored Candidate

- **Phase:** 2 (step 2.3) · **Status:** implemented · **Date:** 2026-10-20
- **Depends on:** design 06 (harness), design 07 (StrategyPlan); [D0006](../decisions/D0006-model-routing-gemini-flash.md) (codegen needs a fast, capable model)

## Goal

The first end-to-end modelling loop: the researcher agent writes a `pipeline.py`
plugin from the strategy's roadmap step 1, the sandbox harness scores it, and the
registry records a real scored candidate.

## Interface

```python
PipelineProposal(model_family, description, pipeline_code, feature_ideas_used, notes)

MLResearcherAgent.plan(profile_md, data_report_json, feature_plan_json,
                       guidance, roadmap_step) -> PipelineProposal

baseline_phase(ctx)  # propose → register → write code → syntax check → harness → record CV
```

## Phase flow

1. Load `strategy_plan.json` (guidance + roadmap), analysis artifacts, and
   `eda/profile.md` (exact column names/dtypes — the researcher cannot write correct
   code without them).
2. Researcher proposes `pipeline_code` for the roadmap's priority-1 step.
3. `registry.add(...)` → candidate id; write the code to
   `workspace/models/<cand-id>/pipeline.py` (defensively stripped of markdown fences).
4. **Syntax check** (`compile()`) before spawning anything — a broken candidate costs
   one second, not a subprocess round trip.
5. `run_candidate(...)` via the harness (protocol from `build_cv_protocol`).
6. `status=ok` → `registry.record_cv(mean, std)` (auto-advances proposed → scored);
   failure/timeout → candidate status `failed` + `CandidateRunError` (Phase 3 adds
   repair-retry with the harness log as feedback).

New lifecycle state: `CandidateStatus.FAILED` — failed builds stay recorded (lineage +
audit) instead of vanishing.

## Prompt contract (the critical piece)

`prompts/ml_researcher.md` pins the harness interface exactly: `build_model()`
required, `make_features(df)` optional and fit-free (never sees the target), no
fitting at import time, allowed libraries, guidance = mandatory instructions. The
guidance from the planner (imbalance handling, leakage rules, metric discipline) is
injected verbatim — this is how the challenge's *nature* reaches the code.

## Tests

- Agent: prompt contains profile columns, step, guidance; journaling works.
- Phase (real harness in subprocess, synthetic data): happy path → candidate scored
  with CV recorded in registry + artifacts on disk; error pipeline → candidate failed
  + raised error; syntax-error code → failed without spawning the harness.
- Registry: FAILED status transitions.

## Retro

Live-validated end to end on the real run with the new default model (D0006):

- The full pipeline (init → cache reuse → profile → analyst → planner → researcher →
  harness → registry) completed in one foreground execution; **all LLM calls finished
  in seconds** where the free reasoning model took 1.5–3 minutes each.
- The researcher's generated baseline is genuinely good: follows guidance (id dropped,
  categoricals encoded, AUC objective), and scores **0.941779 ± 0.0008 AUC** over 5
  folds — matching a hand-written LightGBM baseline (0.94135) within noise. Strong
  evidence the plugin contract works.
- Quality nuance for the reviewer (step 2.4) to catch: the generated code *assumes*
  the harness will set `scale_pos_weight` from fold class balance ("placeholder") —
  it won't. Intent-vs-implementation gaps like this are exactly the reviewer's job.
- Harness runtime (~162s for 5×LightGBM-1000 on 668k rows) is now the bottleneck —
  compute, not LLM. Foreground runs still need generous process timeouts; the
  `slayer resume` ops item remains the robustness fix for interruptions.
