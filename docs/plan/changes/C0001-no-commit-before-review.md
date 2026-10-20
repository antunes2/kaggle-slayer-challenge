# C0001 — Workflow change: no commit before user review

- **Date:** 2026-10-19

## What changed

Previously: each implementation step was committed first and then reviewed by the user
(commit `57c7d0c` was produced this way).

Now: every step is **implemented → presented for review → committed only after the
user approves**. No exceptions.

## Why

The user wants review to *gate* commits, not follow them. A commit is a checkpoint of
approved state; unreviewed work should never land in history.

## Plan impact

- `docs/plan/README.md` — workflow item 5 updated.
- `00-implementation-plan.md` §15 — step/commit wording updated.
- `AGENTS.md` — created to persist the rule across sessions.
