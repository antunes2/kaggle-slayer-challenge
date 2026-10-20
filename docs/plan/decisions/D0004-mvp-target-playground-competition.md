# D0004 — MVP target: Playground Series tabular competition

- **Status:** accepted
- **Date:** 2026-10-19

## Context

The MVP slice (Phases 0–2b) needs a concrete target competition to run against.

## Options considered

1. **Playground Series tabular competition** — monthly, fast feedback, forgiving
   leaderboard, generous quota; low stakes while the system is unstable.
2. **Real ongoing competition directly** — highest signal for the actual goal (top 20%
   on real comps), but slow feedback and precious quota while debugging.
3. **Historical/finished competition** — data and leaderboard still available, zero
   quota pressure, but no live LB dynamics and submissions don't produce a live rank.

## Decision

Target a **Playground Series tabular competition** for Phases 0–2b. Phase 3 (iteration
engine) moves to a real ongoing competition; selection criteria for that transition will
be defined as a question in `questions/` when the time comes.

## Consequences

- Top 20% on a playground comp is **not** the final goal (G2 refers to real
  competitions); it is the MVP acceptance gate.
- The tabular toolchain (sklearn / LightGBM / CatBoost) is exercised from day one,
  matching the v1 non-goals (no deep learning).
