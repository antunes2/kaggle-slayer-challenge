# Project Status & Next Steps

> **Read this first in any new session** — it's the living summary of where the
> project stands, what the live evidence says, and what to do next. For the deep
> structure: `docs/architecture.md` (system), `docs/loop.md` (phases),
> `docs/usage.md` (operating). Master plan: `docs/plan/00-implementation-plan.md`.

*Updated: 2026-10-22*

## Where we are

The full pipeline is **implemented and live-proven end-to-end**: all 9 engine phases,
the five agents, the CV harness sandbox, resume, monitoring. A real Kaggle submission
has been made and accepted.

| Milestone | Status |
|---|---|
| Phase 0 — infrastructure (config, LLM client, kaggle client, journal/registry) | ✅ |
| Phase 1 — data collection + EDA | ✅ live |
| Phase 2 — planner/researcher/reviewer, mock submission end-to-end | ✅ live |
| Phase 2b — first real Kaggle submission | ✅ submission `57503219` accepted (run `20261022-001955-playground-series-s6e9`, candidate `cand-0006`, LightGBM, CV 0.94183); **LB score recording pending one resume** (description-matching fix landed after the run) |
| Phase 3 — iteration engine, resume, timeout handling | ✅ live (CV 0.94135 → 0.94185 in one pass) |
| Phase 4 — monitoring (4.1 done: LB collection + percentile-vs-target; 4.2 open) | 🔶 |
| Phase 5 — daemon/attach/chat, agentic orchestrator | ⬜ |

## Live evidence (closed playground comp `playground-series-s6e9`)

- ~668k rows; the loop runs **baseline → 5-6 roadmap candidates → reviews →
  (mock or real) submission → monitor** unattended.
- CV improvement in one pass: **0.94135 → 0.94185**; every gate fired correctly at
  least once live (repair-retry, probe-fold budget abort, CV gate, reviewer
  approve/revise honesty, quota separation).
- Real-world placement of the submitted score: **~top 50 percentile** of the
  leaderboard — short of the top-20% goal. That gap is *by design* the job of
  step 4.2 (iterate re-entry) + Phase 5: more iterations, ensembles (XGBoost was
  only added to the toolbox late — no run has used it yet), seed averaging,
  and a better researcher model (see below).

## Known issues (all documented)

- `docs/plan/questions/Q0003` — Ctrl+C leaves run status `active` (graceful SIGINT
  pending); orphaned harness processes can outlive their budget if the parent is
  killed during a long C fit (live incident: 84 min at 600% CPU).
- `Q0004` — competition discovery UX (`slayer competitions` browse command proposed).
- Monitor leaderboard matching is username/display-name heuristic.
- Iterate re-entry on target shortfall is not yet a mechanism (step 4.2).

## Next steps (recommended order)

1. **Resume run `20261022-001955-playground-series-s6e9`** — with the
   description-matching + pagination fixes, MONITOR will record cand-0006's real LB
   score and our position; run reaches `finished`. Closes 2b's last checkmark.
2. **Step 4.2 — deadline awareness + iterate re-entry** (the "deal with falling
   short" mechanism): when MONITOR reports percentile > target and budget/quota/
   deadline allow, let the planner write a *new* roadmap segment from the registry's
   experiment history (what's been tried, what worked) and re-enter ITERATE. Also:
   graceful SIGINT (Q0003.2) and possibly the `slayer competitions` command (Q0004).
3. **Researcher model A/B (Q0001 note)** — the next session intends to run with a
   better model. Change **only `ml_researcher`** in `config/agents.yaml` first
   (strong reasoning model, e.g. `anthropic/claude-sonnet-4.5` or a frontier
   OpenAI/Gemini model); measure: CV trajectory, failed-candidate count, repair
   cycles, vs. the recorded baselines above. ~$1-3 per run vs. $10 budget.
   Watch: max_tokens may need raising for reasoning models; 429 backoff already
   handles rate limits.
4. **First active-competition campaign** once an appealing Playground episode opens
   (or pick another active tabular comp): fresh `--live` run, approve-first;
   then Phase 4.2's re-entry makes it a multi-day campaign within quota.
5. **Phase 5** when the above is boring: daemon + `attach`, `slayer chat`,
   agentic orchestrator upgrade, docker sandboxing.

## Conventions for a new session

- `AGENTS.md` is the standing rulebook (review-before-commit; ask before live runs;
  docstrings on every function; phase-completion reports with risks/opportunities).
- `docs/plan/README.md` explains question/decision/change/design records — update
  them as you go; keep `00-implementation-plan.md` statuses current.
- Checks before presenting any step: `uv run pytest -q` (169 green as of this
  writing) and `uv run ruff check .`.
- Working tree state: clean; everything through the monitor fixes is committed.
