# Project Status & Next Steps

> **Read this first in any new session** — it's the living summary of where the
> project stands, what the live evidence says, and what to do next. For the deep
> structure: `docs/architecture.md` (system), `docs/loop.md` (phases),
> `docs/usage.md` (operating). Master plan: `docs/plan/00-implementation-plan.md`.

*Updated: 2026-10-22 (4.4 — evidence-driven strategy: notebook scout + prompts)*

## Where we are

All 9 phases, re-entry rounds, the OOF blender, and now the **public-notebook scout**
are implemented. The big lesson of the day (full evidence in
[`changes/C0004`](changes/C0004-evidence-driven-strategy.md)): the bottleneck was
**how the agents chose ideas**, not missing capabilities.

| Milestone | Status |
|---|---|
| Phases 0-3 (infra, EDA, planning/baseline/review/submit, iteration, resume) | ✅ live |
| Phase 2b — first real Kaggle submission (LB 0.94149, ≈ top 71%) | ✅ |
| 4.1 monitor · 4.2 re-entry rounds · 4.3 OOF blender + researcher A/B | ✅ live |
| 4.4 notebook scout + evidence-driven prompts + retry hardening | ✅ (live test confirms direction) |
| 4.5 paired-fold submission gate + reviewer re-review fix | ⬜ **next** |
| Phase 5 — daemon/attach/chat, agentic orchestrator | ⬜ |

## Live evidence (`playground-series-s6e9`, closed; target line 0.94640 = top 20%)

| Run | What changed | Best CV | ≈ LB position |
|---|---|---|---|
| `20261022-001955` | gpt-5-mini researcher, generic prompts | 0.94183 (LB 0.94149) | top 71% (real LB) |
| `20261022-031417` | claude-sonnet-5.5 researcher, 2 re-entry rounds, blender | 0.94200 | top 53% (CV-estimate) |
| `…-031417-…-scout-test` | + notebook scout + evidence-driven prompts (1 round) | **0.94426** (2nd step; 1st: 0.94404) | ≈ top 45% |

- The new researcher model: 1 failure in 14 candidates (vs 3 + 1 stuck in 9);
  CatBoost works; $0.56 per night.
- Re-entry and blending work end-to-end but added ~nothing with generic ideas
  (candidates were 0.998-correlated copies). The scout fixed the *idea* source.
- Public notebooks reach CV ≈ 0.946; the top-20% line (0.9464) is reached there with
  the original dataset, which our rules forbid (kept, by decision). Realistic goal for
  this competition with current rules: ≈ top 30-40%.

## Known issues

- **REVISE blocks the best candidate** — a REVISE'd candidate stays `scored`, is re-picked
  by REVIEW repeatedly, and is never submittable until revised (step 4.5).
- The submission gate compares mean CV vs a fixed threshold; paired fold noise is
  ~28× smaller than fold std (step 4.5, code part).
- Q0003 (graceful Ctrl+C), Q0004 (competition browser), `slayer status` prints every
  phase of every resume on one line.

## Next steps (recommended order)

1. **Finish/read the scout-test round** (log: the run's journal; `slayer status
   20261022-031417-playground-series-s6e9-scout-test`).
2. **Step 4.5** — paired-fold gate + REVISE handling (skip already-reviewed candidates;
   let a REVISE'd best candidate be revised before newer reviews).
3. **`slayer competitions` (Q0004)** → pick an active Playground episode → live,
   approve-first campaign; resume once per day.
4. Optional: feed the digest to the researcher too (~$0.10-0.20/run more) if step
   descriptions prove too lossy.

## Conventions for a new session

- `AGENTS.md` is the standing rulebook (review-before-commit; ask before live runs;
  docstrings on every function; phase-completion reports with risks/opportunities).
- Keep prompts **generic for tabular data**; competition specifics come from the data
  profile and the notebook digest (C0004).
- Checks before presenting any step: `uv run pytest -q` and `uv run ruff check .`.
