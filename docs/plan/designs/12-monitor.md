# 12 — MONITOR: LB-Score Collection & Position Awareness

- **Phase:** 4 (step 4.1) · **Status:** implemented · **Date:** 2026-10-21
- **Answers:** the user's "top-50% vs top-20% — how does the program deal with falling short?" — MONITOR is the sensing organ; re-entry is the acting organ (step 4.2).

## Goal

Close the feedback loop with the real world: collect real LB scores for submitted
candidates, compute our leaderboard position vs the run's target percentile, journal
and report both.

## MONITOR phase (runs once, last in the engine order)

1. **LB-score collection** — poll `kaggle.submissions(competition)` (read-only);
   for each registry candidate that is `submitted` with `lb_score is None` and a
   non-mock `submission_id`, match the kaggle submission by ref:
   - score present → `registry.record_lb(...)` (journals `candidate.lb_scored`),
   - no score yet (pending scoring) → skip with a status message,
   - mock submissions (`mock-*` ids) are skipped — simulated LB stays simulated.
2. **Position vs target** — fetch `kaggle.leaderboard(competition)` (top page),
   locate our entry by matching `team_name` against the authenticated username
   (case-insensitive), compute `percentile = rank / total * 100` (top-20% target ⇔
   percentile ≤ 20), journal `monitor.position`, and report met / NOT met.
   Not found on the fetched page → honest message (v1 fetches the first page only).
3. **Engine completion** — with MONITOR implemented, `PHASE_ORDER` is fully
   implemented: the run reaches `FINISHED` (resume re-enters MONITOR for fresh data).
   Deliberately *no automatic re-entry* in v1: MONITOR senses and reports; re-entry
   (iterate-more on shortfall) is a decision organ — see step 4.2 / Q0005, implemented
   in [design 13](13-reentry-rounds.md).

## Percentile semantics

`percentile = rank / total * 100`, rank 1 = best. "Top 20%" ⇔ percentile ≤ 20.
Matches the user's manual reading ("top 50 percentile" ≈ rank/total ≈ 0.5).

## CLI

`slayer leaderboard <competition>` — top-10 table + our position (if on the fetched
page), same matching as MONITOR. Requires kaggle auth; read-only.

## Tests

- Fake kaggle (submissions + leaderboard + ping): pending-score recorded via
  `record_lb` + journaled; pending-no-score skipped with message; mock ids skipped;
  already-scored no-op; percentile computation met vs NOT met vs target; not-on-LB
  graceful; CLI leaderboard renders entries + our position.

## Retro

Live hardening (2026-10-22): the paginated leaderboard approach couldn't reach our
field (3576 teams) and username/display-name matching was fragile. Replaced by the
**full leaderboard CSV download** — authoritative total, per-team member usernames
(exact match, no display-name guessing), and the CSV persists as a run artifact.
Late submissions to closed comps are scored but never listed, so MONITOR also
computes an **estimated position** from the best recorded LB score (insert-rank
against the listed field, tie band included). Live-validated:
`best LB 0.94149 → rank ~2545/3577 (top 71.15%), tie band 19 — target 20% NOT met`.

Implemented as designed; the natural bonus: with MONITOR in place, **all nine engine
phases exist** — runs now reach `FINISHED` instead of pausing, and `resume` re-enters
MONITOR for fresh leaderboard data.

- Implementation surfaced a real API gap: there was no way to record a *real
  submission awaiting its score* (KaggleSink used `set_status` and lost the kaggle
  submission id). Added `CandidateRegistry.record_submission()` (sets the real
  submission id + status, LB stays None until MONITOR fills it); the journal-replay
  handler was extended to restore submission ids on rebuild. KaggleSink now uses it.
- Score strings from Kaggle are parsed defensively (`_parse_score`); pending
  submissions are skipped with a status message; mock submission ids (`mock-*`) never
  touch the real LB path.
- Matching our LB entry is username-based (case-insensitive) — the display-name
  question will be settled by the first live MONITOR run.