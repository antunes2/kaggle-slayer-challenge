# Project Status & Next Steps

> **Read this first in any new session** — it's the living summary of where the
> project stands, what the live evidence says, and what to do next. For the deep
> structure: `docs/architecture.md` (system), `docs/loop.md` (phases),
> `docs/usage.md` (operating). Master plan: `docs/plan/00-implementation-plan.md`.

*Updated: 2026-10-22 (4.6 — delivery hardening; v1.0 codebase, limitations documented)*

## Where we are — v1.0 delivered

*Updated: 2026-10-22.* Every planned v1 capability is implemented, tested (233 unit
tests), documented, and live-proven on two competitions. Honest limits are in
[`docs/limitations.md`](../limitations.md); the top-20% goal is **not** reached.

| Milestone | Status |
|---|---|
| Phases 0-3: infra, EDA, planning/baseline/review/submit, iteration, resume | ✅ live |
| 4.1 monitor · 4.2 re-entry rounds · 4.3 OOF blender + researcher A/B | ✅ live |
| 4.4 notebook scout + evidence-driven prompts · 4.5 paired gate + review fix + `competitions` | ✅ live |
| 4.6 graceful Ctrl+C, limitations doc, README · 4.7 external data, dashboard, harness fixes | ✅ (external data not yet live) |
| Phase 5: daemon/attach/chat, agentic orchestrator | ⬜ out of v1 scope |

## Live results

| Competition | Mode | Result |
|---|---|---|
| s6e9 (closed) | first real submission | LB 0.94149 ≈ top 71% (estimated) |
| s6e9 | dry, after C0004 | best CV 0.94426 ≈ top 45% (CV-estimated) |
| **s6e10 (active)** | live, 4 real submissions, 2 rounds, $0.80 | **LB 0.96015, rank 1240/2714 = top 45.7%**; gains came from OOF blends |

## Next steps (optional, post-delivery)

1. Resume the s6e10 campaign daily until 2026-10-31 (fresh quota; first live test of
   external data): `uv run slayer resume 20261022-142647-playground-series-s6e10 --max-rounds 3`
   (raise the number each day).
2. Improvement opportunities, ranked: [`docs/limitations.md` §8](../limitations.md).
3. A regression or multiclass competition to prove generality further.

## Conventions for a new session

- `AGENTS.md` is the standing rulebook (review-before-commit; ask before live runs;
  docstrings on every function; phase-completion reports with risks/opportunities).
- Keep prompts **generic for tabular data**; competition specifics come from the data
  profile and the notebook digest (C0004).
- Checks before presenting any step: `uv run pytest -q` and `uv run ruff check .`.
