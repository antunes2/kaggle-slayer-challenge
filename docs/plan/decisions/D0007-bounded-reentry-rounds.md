# D0007 — Shortfall handling: automatic, bounded re-entry rounds

- **Status:** accepted
- **Date:** 2026-10-22
- **Settles:** [Q0005](../questions/Q0005-iterate-reentry-on-shortfall.md)

## Context

MONITOR senses the position vs target; nothing acted on a shortfall. Live data showed
the gap to top-20% (+0.005 AUC) was ~10× the spread of everything the first roadmap
tried — incremental steps cannot close it, so re-entry must re-plan, not repeat.

## Options considered

Automatic bounded rounds; one round per explicit resume; agentic orchestrator
(Phase 5). See Q0005.

## Decision

- After the linear pass, a **deterministic gate** decides re-entry: target not met,
  `rounds_done < max_rounds` (config, default 2), LLM spend (summed from the journal)
  below budget, deadline not closer than one harness budget, and the planner has not
  declared the campaign exhausted.
- A round is `REPLAN → ITERATE → REVIEW → SUBMIT → MONITOR`. REPLAN is the planner
  extending the roadmap from the full experiment history plus the **score gap** (LB
  score at the target percentile vs our best). The planner may return `stop` with a
  reason — that ends the campaign for the run.
- Position falls back to a **CV-based estimate** when no real LB score exists (dry
  runs, pending scoring), flagged `basis: "cv"`.
- The LLM budget is the **journal's** `agent.call` cost sum (max with the live client),
  so it is cumulative across resumes.

## Consequences

- The engine gains one bounded outer loop; `PHASE_ORDER` stays the linear first pass
  (D0003 holds: code owns control flow, the LLM owns roadmap content).
- Resume of a finished-short run continues the campaign until a stop condition holds —
  multi-day campaigns are "resume once per day".
- Every round's decision is journaled (`reentry.decision`, `roadmap.extended`).
