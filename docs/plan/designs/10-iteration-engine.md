# 10 — ITERATION: The Improvement Loop

- **Phase:** 3 (step 3.1) · **Status:** implemented · **Date:** 2026-10-20
- **Decisions:** [D0003](../decisions/D0003-orchestrator-deterministic-pipeline.md) (loop lives *inside* the iterate phase — the engine stays linear)
- **Feeds:** [Q0002](../questions/Q0002-harness-timeout-handling.md) lands here next (step 3.3)

## Goal

Turn the pipeline into a system that *improves*: one candidate per pending roadmap step,
each evaluated, reviewed, and (if gates pass) submitted — with repair-retry on failures
and LLM-budget enforcement.

## Design: the loop is inside the phase

The engine's `PHASE_ORDER` stays linear and deterministic (D0003). `iterate_phase`
runs the improvement cycle *internally*:

```
for each pending roadmap step (priority order, resume-aware):
    if LLM budget exhausted -> stop (journaled budget.warning)
    researcher.propose(step, iteration_context)
    evaluate through the harness          <- shared with baseline via helper
    if failed:
        one repair attempt (error feedback into the proposal context)
        if failed again -> next step (both attempts recorded as failed)
    review_phase(ctx)  -> reviewer judges the best unreviewed candidate
    submit_phase(ctx)  -> CV gate, quota gate, approval gate, sink
```

**Resume-awareness for free**: pending steps are derived from the registry — each
candidate records `params={"roadmap_step", "priority"}`, so a re-run (or a future
`slayer resume`) skips steps that already have a non-failed candidate.

## Iteration context (the feedback bundle)

Each proposal receives: the registry summary (all candidates + CV/LB scores — bounded
markdown via `summary_for_agent`), the last reviewer verdict (decision + reasoning +
risk flags, now journaled), and the tail of the last failed candidate's `run.log`.
This is how the system *acts on its own quality control*: REVISE feedback and crash
tracebacks become the next proposal's constraints.

## Budget enforcement

`ctx.llm.cost.total_usd` (accumulated by the LLM client since run start) is checked
before every proposal and repair. Exhaustion journals `budget.warning` and stops
iteration — the run still completes REVIEW/SUBMIT for what exists.

## Shared evaluation helper

`_evaluate_proposal()` (register → write code → syntax check → harness → record) is
extracted from BASELINE and reused by ITERATE — one code path, identical semantics:
syntax errors and harness failures mark the candidate `failed` without raising;
BASELINE chooses to raise (first candidate failing the run is fatal), ITERATE repairs.

## Tests

- Iterate executes the pending step only (step 1 pre-seeded): new candidate scored →
  reviewed (approve) → mock-submitted; registry + journal verified.
- Repair-retry: broken proposal first, good second → candidate scored, both recorded.
- Double failure → step skipped, loop continues.
- Budget exhaustion (fake client cost ≥ run budget) → no proposals, `budget.warning`.
- No pending steps → no LLM calls.
- Baseline unchanged behavior through the shared helper (existing tests).

## Retro

Implemented as designed; validated with 5 phase tests (real harness subprocesses on
synthetic data). Notes:

- The repair-retry + CV-gate interplay surfaced a test-design lesson: on tiny synthetic
  data, 2-fold AUC is wildly noisy (0.25!), and the submit gate *correctly* refused a
  non-improving candidate — the gates fire in tests exactly as they will live.
- Baseline and ITERATE now share `_evaluate_proposal()` — one code path for
  register/write/compile-check/harness/record; BASELINE raises on failure (first
  candidate failing is fatal), ITERATE repairs.
- Review verdicts now journal their full `reasoning` — REVISE feedback is what the
  next proposal's iteration context consumes (the self-improvement loop's fuel).
- Live validation of a full iterate loop (3-5 roadmap steps × real CV) takes 15-40
  minutes — deliberately deferred pending user approval (per the new AGENTS rule).
