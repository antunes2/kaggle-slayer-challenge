# Q0005 — How does the run act on falling short of the target? (iterate re-entry)

- **Status:** settled → [D0007](../decisions/D0007-bounded-reentry-rounds.md)
- **Raised:** 2026-10-22
- **Blocks:** step 4.2

## Question

MONITOR reports the position vs `target_percentile` but the run then ends. When the
target is not met, what mechanism lets the team keep improving — and when must it stop?

## Context

- Live evidence (`playground-series-s6e9`, run `20261022-001955`): best LB 0.94149 ≈
  top 71%; the top-20% line on the downloaded leaderboard is **0.94640** — a **+0.005
  AUC gap**. All six scored candidates span 0.94147–0.94183 CV (spread ≈ 0.0004, below
  fold noise σ ≈ 0.0008). More roadmap steps *of the same kind* cannot close the gap;
  the re-planner must see the gap **in score units** next to the spread of what was tried.
- The engine is a deterministic phase machine (D0003); ITERATE already derives pending
  work from the registry + `strategy_plan.json` roadmap titles.
- The LLM budget was enforced from the in-process client cost — it **reset on every
  resume**, so multi-round/multi-day campaigns would never hit the cap.

## Options

1. **Automatic bounded rounds** — after MONITOR a deterministic gate (target not met,
   rounds left, budget left, deadline allows, planner hasn't stopped) re-enters
   REPLAN → ITERATE → REVIEW → SUBMIT → MONITOR, up to `max_rounds`. Fits G5 ("zero
   commands to let it finish").
2. **One round per resume** — MONITOR only recommends; each `slayer resume` performs one
   extension. More manual control, slower campaigns.
3. **Agentic orchestrator decides** — the Phase 5 upgrade; untestable today.

## Recommendation

**(1)**, with the gap signal (LB score at the target percentile), a CV-based position
proxy when no real LB score exists (dry runs, pending scoring), and journal-based
budget accounting. User agreed 2026-10-22.
