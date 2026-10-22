# 13 — RE-ENTRY: Bounded Rounds on Target Shortfall

- **Phase:** 4 (step 4.2) · **Status:** implemented · **Date:** 2026-10-22
- **Decisions:** [D0007](../decisions/D0007-bounded-reentry-rounds.md) (settles Q0005)

## Goal

Give MONITOR's sensing an acting organ: when the run is short of its target and
resources allow, re-plan from the experiment history and iterate again — bounded,
journaled, resumable.

## Engine

```
first pass:  INIT → … → ITERATE → REVIEW → SUBMIT → MONITOR      (unchanged, PHASE_ORDER)
then:        while decide_reentry(ctx).reenter:
                 REPLAN → ITERATE → REVIEW → SUBMIT → MONITOR      (ROUND_ORDER)
```

- `decide_reentry` (deterministic, `loop/campaign.py`) journals `reentry.decision`
  `{reenter, reason, round, ...signals}` and stops on the first failing check:
  1. planner previously declared the campaign exhausted (`roadmap.extended` stop),
  2. `rounds_done >= run.max_rounds` (rounds = successful `roadmap.extended` events),
  3. no position signal (no `monitor.position` with a percentile),
  4. target met,
  5. LLM spend (journal sum) ≥ budget,
  6. competition deadline in the future but closer than `harness_timeout_s`
     (a past deadline does not block: late submissions are still scored).
- REPLAN raising `CampaignStop` (planner said stop) ends the loop cleanly → `finished`.
- Re-entry only runs with the default handler set (or when the caller passes a
  decider) — partial handler maps in tests keep their pause semantics.
- Round phases journal `phase.enter` with `{"phase", "round"}`.

## Gap signal (MONITOR)

- `target_score` = score of the leaderboard entry at rank `max(1, floor(total ×
  target / 100))`; `gap` = direction-aware shortfall of our best score (positive =
  short). Both added to `monitor.position` and printed.
- Metric direction comes from the strategy plan's CV protocol when available (fallback:
  leaderboard ordering).
- Position `basis`: `listed` (our entry on the LB), `lb` (best real LB score inserted
  into the field — late submissions), `cv` (no real LB score yet: best CV inserted —
  dry runs and pending scoring).

## REPLAN (planner extension)

`PlannerExtensionAgent` (routing: the `planner` config; prompt `prompts/planner_extend.md`)
receives: the current strategy (summary, roadmap, guidance), the **full experiment
history** (every candidate: step, family, status, CV ± std, LB, first error line for
failures), recent reviewer verdicts, the latest position + gap, deterministic stats
(scored-CV spread, median fold std, gap/spread ratio), round number, remaining budget,
and the harness time budget. Output `RoadmapExtension`:

```
assessment: str            # why we are short; what the history rules out
steps: list[RoadmapStep]   # 1-4 new steps when not stopping
new_guidance: list[str]
stop: bool; stop_reason: str | None
```

Merge (code): titles made unique (` (round N)` suffix on collision), priorities
renumbered after the current max, guidance appended with dedupe; `strategy_plan.json`
+ `.md` rewritten; `reports/roadmap_round_N.json` stores the extension verbatim.
ITERATE then picks the new steps up as pending — the existing attempt caps, repair,
review, and submit gates apply unchanged.

## Budget (journal-based)

`llm_spend(journal)` sums `agent.call.cost_usd`; `_budget_out` uses
`max(journal spend, live client cost)` — cumulative across resumes.

## Config / CLI

`defaults.max_rounds` (default 2, 0 disables) → `Run.max_rounds`; `slayer run
--max-rounds N`. `slayer status` shows rounds done and the last re-entry decision; the
"paused" line only appears for paused runs.

## Tests

- `decide_reentry`: each stop reason + the re-enter path; deadline parsing.
- `llm_spend` sums journal costs; budget stop triggers from journal spend alone.
- MONITOR: `target_score`/`gap` for listed and estimated positions; CV-basis proxy.
- REPLAN: extension merged (unique titles, priorities, guidance dedupe, round file);
  planner stop raises `CampaignStop` and journals the stop.
- Engine: rounds run until the decider says stop; `CampaignStop` ends the run
  `finished`; partial handler maps unchanged.

## Retro

Implemented as designed (21 new tests; suite green). Additions found during the work:
`slayer resume --max-rounds N` (persisted via `RunStore.update_settings`, journaled
`run.settings_changed`) — without it a multi-day campaign is capped at `max_rounds`
total; history rows for stuck `proposed` candidates read "never completed (status=…)".
Offline check against the real run `20261022-001955` (copied to a temp dir): MONITOR
reports `target line 0.94640, gap +0.00491`; the gate re-enters round 1; the planner's
position input shows `gap / spread ratio: 13.9x`. Live validation pending.

### Live validation (2026-10-22, two runs)

- **Mechanism: works end-to-end.** Dry-run fork of `20261022-001955` and the fresh run
  `20261022-031417` (claude-sonnet-5.5 researcher): the gate re-entered on the real
  and the CV-basis gap, REPLAN cited the gap/spread ratio (14×, 9×, 8×) and planned
  structural-sounding steps; rounds 1-2 ran unattended; `rounds: 2 / 2` reported.
- **Value: none yet.** All round steps scored 0.9415-0.9418 — the planner's levers were
  as generic as the first roadmap's (target encoding of string columns, stacking, seed
  bagging, class-weight ablation). Re-entry multiplies the quality of the ideas it is
  given; it cannot create information. Addressed in [C0004](../changes/C0004-evidence-driven-strategy.md)
  (notebook digest + evidence-ordered levers).
- **Bug found:** a REVISE'd candidate stays `scored`, so REVIEW re-picks it every time
  (cand-0008 reviewed 5×; three newer candidates never reviewed). Fix scheduled with
  the paired-fold gate step.
