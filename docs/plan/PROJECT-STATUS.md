# Project Status & Next Steps

> **Read this first in any new session** — it's the living summary of where the
> project stands, what the live evidence says, and what to do next. For the deep
> structure: `docs/architecture.md` (system), `docs/loop.md` (phases),
> `docs/usage.md` (operating). Master plan: `docs/plan/00-implementation-plan.md`.

*Updated: 2026-10-22 (4.2 re-entry rounds; 4.3 OOF blender + researcher A/B)*

## Where we are

The full pipeline is **implemented and live-proven end-to-end**: all 9 engine phases,
the five agents, the CV harness sandbox, resume, monitoring. A real Kaggle submission
has been made and scored. Step 4.2 adds the "act on falling short" mechanism —
implemented and unit-tested, **not yet live-validated**.

| Milestone | Status |
|---|---|
| Phase 0 — infrastructure (config, LLM client, kaggle client, journal/registry) | ✅ |
| Phase 1 — data collection + EDA | ✅ live |
| Phase 2 — planner/researcher/reviewer, mock submission end-to-end | ✅ live |
| Phase 2b — first real Kaggle submission | ✅ submission `57503219`, LB **0.94149** → est. ~2545/3577 (**top 71%**) — target top-20% NOT met |
| Phase 3 — iteration engine, resume, timeout handling | ✅ live |
| Phase 4 — monitoring (4.1 LB + position; 4.2 gap signal + re-entry rounds; 4.3 OOF blender) | ✅ (4.2 live validation in progress; 4.3 pending) |
| Phase 5 — daemon/attach/chat, agentic orchestrator | ⬜ |

## Live evidence (closed playground comp `playground-series-s6e9`)

- ~668k rows; the loop runs **baseline → 5-6 roadmap candidates → reviews →
  (mock or real) submission → monitor** unattended.
- **The gap, quantified** (leaderboard snapshot, 3576 teams): top-20% line = **0.94640**;
  our best LB = 0.94149 → **gap +0.0049 AUC**. All scored candidates span 0.94147–0.94183
  CV (spread 0.00035, below fold std 0.0008) → **gap/spread ≈ 14×**: more of the same
  steps cannot close it. The top ~30% are packed within 0.0006 of each other with a
  cliff below — typical of a structural unlock most serious teams found.
- **CatBoost never landed** (4 attempts, 3 distinct contract errors: category dtype,
  raw strings, a wrapper without `classes_`) — so the run never had a working second
  family, and its "Ensemble" step was a 16-minute in-pipeline refit.
- That run's calls predate cost journaling (status shows $0 — not a live bug).

## What 4.2 does (design 13, D0007)

After MONITOR, a deterministic gate re-enters `REPLAN → ITERATE → REVIEW → SUBMIT →
MONITOR` while the target is unmet and rounds (`max_rounds`, default 2) / budget /
deadline allow. REPLAN shows the planner the full history (failures with error lines),
the **gap vs spread** ratio, and round constraints; it appends 1-4 new steps or stops.
Dry runs estimate the position from CV (`basis: cv`). The LLM budget is now summed from
the journal, so it holds across resumes. Verified offline against the real run's
artifacts: the gate re-enters with `gap +0.00491`, ratio 13.9×.

## Known issues (all documented)

- `Q0003` — Ctrl+C leaves run status `active` (graceful SIGINT pending); orphaned
  harness processes can outlive their budget if the parent is killed mid C fit.
- `Q0004` — competition discovery UX (`slayer competitions` proposed).
- OOF blending only for binary-probability metrics and regression; pre-4.3 candidates have no OOF.
- Monitor leaderboard matching is username heuristic.

## Next steps (recommended order)

1. **Finish the 4.2 live check** — dry-run fork `20261022-001955-…-reentry-dry`
   (resumed from a git worktree at the 4.2 commit). Round 1 planner output already
   observed: it cited the 14× gap/spread ratio and planned a CatBoost contract fix,
   in-estimator target encoding, stacking, and a class-weight ablation.
2. **Live-validate 4.3 + researcher A/B** with one fresh dry run on the same comp
   (`slayer run playground-series-s6e9`): first run with OOF persistence (blend step
   active) and `claude-sonnet-5.5` as researcher; compare against the Q0001 baseline.
3. **`slayer competitions` browse command (Q0004)** — before the first
   active-competition campaign, which is when picking a comp actually matters.
4. **First active-competition campaign** — fresh `--live` run, approve-first; re-entry
   makes it a multi-day campaign ("resume once per day").
5. Graceful SIGINT (Q0003.2); then **Phase 5** — daemon + `attach`, `slayer chat`,
   agentic orchestrator.

## Conventions for a new session

- `AGENTS.md` is the standing rulebook (review-before-commit; ask before live runs;
  docstrings on every function; phase-completion reports with risks/opportunities).
- `docs/plan/README.md` explains question/decision/change/design records — update
  them as you go; keep `00-implementation-plan.md` statuses current.
- Checks before presenting any step: `uv run pytest -q` and `uv run ruff check .`.
