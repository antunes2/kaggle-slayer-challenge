# C0003 — Phase 2b (live submission) moves after Phase 3; resume pulled forward

- **Date:** 2026-10-20

## What changed

- **Phase 2b** ("first live submission") is postponed until **after Phase 3**
  (iteration engine). It was originally the immediate follow-up to Phase 2.
- **`slayer resume`** moves from Phase 4 into Phase 3 (step 3.2).
- The proposed manual submission override (`slayer submit <run> <cand>`) is
  **rejected for now** — approve-first + agentic autonomy cover current needs.

## Why

- Phase 2's live validation showed the reviewer returning REVISE on the baseline —
  submitting to Kaggle *before* the loop can act on its own quality control would
  spend real quota on a candidate the system itself doesn't fully trust. The first
  real submission should be a candidate that survived review after iteration.
- Long iteration loops (Phase 3) make interruption likely (timeouts, Ctrl+C); without
  resume, every interruption costs a full re-run of an 8-minute pipeline. Resume is
  the robustness prerequisite *for* Phase 3, not a Phase 4 luxury.
- The harness timeout incident (see [Q0002](../questions/Q0002-harness-timeout-handling.md))
  reinforced that long-run resilience must exist before we ask the loop to iterate
  autonomously.

## Plan impact

- §11: Phase 3 row scope + step breakdown (3.1 iteration engine, 3.2 resume,
  3.3 harness timeout handling); Phase 2b row re-sequenced after Phase 3;
  Phase 4 row drops "pause/resume checkpoints" (now 3.2) and budget enforcement
  note (lands in 3.1).
- `dry_run` default flipped to **true** per user direction — mock submissions are
  the default; going live is an explicit opt-in (`--live`).
