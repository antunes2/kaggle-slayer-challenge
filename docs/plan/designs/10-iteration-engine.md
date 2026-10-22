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
synthetic data) — the repair-retry + CV-gate interplay fired correctly even in tests
(noise-CV candidates were gate-blocked) — then **live-validated twice** on the real
competition:

Test-phase notes retained: baseline and ITERATE share `_evaluate_proposal()` (one code
path); reviewer verdicts journal full `reasoning` — the self-improvement loop's fuel.

### Live validation 1 (first full iterate loop)

All loop machinery fired (repair, gates, review, pause) — but **zero net improvement**:
every iteration candidate died on a systematic codegen bug. The upgraded researcher
prompt pushed `category`-dtype handling (correct ML practice) without specifying
*order* — the model converted to category dtype *before* arithmetic → TypeError, 9
consecutive failures including repairs. Lesson: prompt knowledge must include
execution-order rules, not just preferences. (Fixed: "category conversion is the LAST
step of make_features"; also hardened the no-hardcoded-ratios rule.)

### Live validation 2 (after prompt fix + resume)

Resumed the run that a Gemini 429 rate limit had killed mid-iteration — resume skipped
analysis/planning/baseline and continued from step 3 (the feature working exactly as
built). Results across the 6-step roadmap (7 candidates):

| candidate | step | CV | outcome |
|-----------|------|-----|---------|
| cand-0001 | baseline LightGBM raw | 0.94135 | scored |
| cand-0002 | ratio features | 0.94159 | approved, **gate-blocked** (below submitted best) |
| cand-0003 | interaction features | **0.94185** | **approved + mock-submitted** |
| cand-0004 | encoding features | 0.94156 | scored (no gain) |
| cand-0005/6 | CatBoost benchmark | — | **probe-fold aborted** (338s fold → ~2000s projected vs 600s budget) |
| cand-0007 | tuning | 0.94169 | scored, within noise |

**Measurable CV improvement 0.94135 → 0.94185** with every gate firing correctly at
least once: repair-retry, probe-fold budget catch, CV-improvement gate (blocked two
noise-level candidates), reviewer approve/revise honesty. Phase 3 acceptance met.

Also surfaced live: OpenRouter 429s need provider-grade backoff — the LLM client now
honors `Retry-After` and backs off 15/30/45s on rate limits instead of 0.5–4s.

### Addendum 2026-10-21: bounded retries (from a live resume incident)

Live finding: failed roadmap steps were retried on *every* resume (deliberate for
crash-repair, unbounded in practice — a step that failed its pass consumed another
candidate per resume). Fixed: **a roadmap step allows at most 3 candidate attempts
total** (initial + repair + one resume retry), counted from the registry including
failed and stuck-proposed candidates; the baseline raises `exhausted 3 attempts`
instead of blind retries. Along the way: terminal narration now announces each
harness evaluation ("evaluating (5 folds, budget Ns) — silence during CV is normal")
and failure lines carry the first line of the actual error, since "failed (error)"
read ambiguously to a human watching the loop.
