# C0002 — Loop phase order: PLANNING moves after ANALYSIS

- **Date:** 2026-10-19

## What changed

Original order (plan §5.2): `INIT → PLANNING → DATA_COLLECTION → ANALYSIS → BASELINE → …`

New order: `INIT → DATA_COLLECTION → ANALYSIS → PLANNING → BASELINE → …`

## Why

- The planner agent writes the `StrategyPlan` from **comp metadata + the DataReport** —
  a strategy written before seeing the data (EDA, target analysis, feature ideas) would
  be generic and discarded anyway. Strong human Kaggle teams plan after EDA, not before.
- It makes **Phase 1** (`INIT → DATA_COLLECTION → ANALYSIS`) contiguous: no stub or
  skip logic is needed for phases that arrive in later milestones, and the loop pauses
  cleanly at the first unimplemented phase.
- Keeps Phase 1 LLM-light: only the data analyst runs before the planner enters.

## Plan impact

- `00-implementation-plan.md` §5.2 — phase diagram + phase responsibilities reordered.
- §11 — Phase 1 scope now includes ANALYSIS (EDA → DataReport); Phase 2 scope starts
  at PLANNING (`PLANNING → BASELINE → REVIEW → SUBMIT`).
- Engine (`PHASE_ORDER` in `slayer/loop/engine.py`) implements the new order.
