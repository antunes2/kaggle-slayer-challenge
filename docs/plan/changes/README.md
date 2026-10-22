# Direction Changes (`changes/`)

One file per meaningful **change of direction** — scope, pattern, or approach shifts
after something was already decided or built. This is the "why did we pivot" trail:
decisions record what we chose at the time; changes record when and why we moved on.

File naming: `C####-slug.md`, numbered sequentially.

## Template

```markdown
# C#### — <title>

- **Date:** YYYY-MM-DD

## What changed
The previous behavior/approach and the new one.

## Why
What we learned or what changed in the world that forced the pivot.

## Plan impact
Which sections of `00-implementation-plan.md` were updated, and which ADRs
(if any) are now superseded.
```

## Index

- [C0001 — Workflow change: no commit before user review](C0001-no-commit-before-review.md)
- [C0002 — Loop phase order: PLANNING moves after ANALYSIS](C0002-loop-phase-reorder.md)
- [C0003 — Phase 2b moves after Phase 3; resume pulled forward](C0003-reorder-2b-and-scope.md)
- [C0004 — Evidence-driven strategy: from a generic playbook to competition evidence](C0004-evidence-driven-strategy.md)
