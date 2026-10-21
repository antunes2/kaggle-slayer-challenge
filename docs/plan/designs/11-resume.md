# 11 — RESUME: Continue Without Redoing

- **Phase:** 3 (step 3.2) · **Status:** implemented · **Date:** 2026-10-20
- **Decisions:** [C0003](../changes/C0003-reorder-2b-and-scope.md) (pulled forward from Phase 4)

## Goal

`slayer resume <run_id>` continues a paused/failed/finished run without redoing
completed work. Motivation (proven live): an 8-minute pipeline that restarts from zero
on any interruption is unusable for long iteration loops.

## Design: idempotent phases, cheap checks

Resume doesn't need new machinery — the engine already runs phases in order; each
*expensive* phase gains a **skip predicate** checked at entry:

| Phase | Skip when | Why safe |
|-------|-----------|----------|
| INIT | never skipped | two cheap API calls; rewrites identical context |
| DATA_COLLECTION | implicitly | cache makes it a no-op extraction (~5s) |
| ANALYSIS | `data_report.json` + `feature_plan.json` exist | LLM call avoided |
| PLANNING | `strategy_plan.json` exists | LLM call avoided |
| BASELINE | a non-failed candidate exists for roadmap step 1 (registry `params.roadmap_step`) | a *failed* baseline re-proposes — resume is also the repair path |
| ITERATE | already resume-aware | pending steps derived from the registry |
| REVIEW / SUBMIT | naturally idempotent | no-op without unreviewed/approved candidates; submitted candidates are excluded |

Skip messages stream to the user (`analysis artifacts exist — skipping`), so it's
visible what was and wasn't redone. The journal keeps the full picture across both
executions.

## Interface

```python
slayer resume <run_id> [--config-dir] [--env-file]
```

`run` and `resume` share one `_execute_run()` (config + clients + engine); `run`
creates the run first, `resume` looks it up and reports its prior status. A missing
run id exits with an error. Resuming re-activates the run through the normal engine
start (status transition journaled).

## Tests

- CLI resume end-to-end on a pre-populated paused run: analysis/planning/baseline skip,
  reviewer + submit still execute for the seeded candidate, mock submission recorded.
- Skip predicates verified through streamed messages; fresh-run path unaffected
  (existing e2e test).

## Retro

Implemented as designed; validated with CLI end-to-end tests (fake clients, real
engine). Notes:

- The skip predicates are deliberately *inside* the phase handlers, not the engine —
  the engine stays a dumb deterministic sequence, phases stay self-contained, and a
  fresh run is unaffected (the checks are cheap file/registry probes).
- `run` and `resume` now share `_execute_run()`; `run` pre-checks secrets *before*
  creating a run (no junk runs from a broken environment) and `resume` reports the
  prior status before re-activating.
- A resumed run whose baseline *failed* re-proposes it (failed candidates don't count
  as "done") — resume doubles as the crash-repair path for the harness-timeout case
  that motivated it.
- Test infrastructure note: `tests/` became a package (`__init__.py`) so tests can
  share fake clients; and `import slayer.cli.main as x` binds the shadowed `main`
  *function* — use `importlib.import_module` (bit us twice).
