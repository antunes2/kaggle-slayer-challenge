# Slayer — Project Documentation

This folder is the source of truth for planning, decisions, and design. Documents are
**living**: they are updated as the project evolves, and anything no longer true gets
deleted rather than left to rot.

| Path | Purpose |
|------|---------|
| [`00-implementation-plan.md`](00-implementation-plan.md) | Living master plan — vision, architecture, phases, statuses, decision summary |
| [`questions/`](questions/) | Open questions awaiting a decision (`Q####-slug.md`) |
| [`decisions/`](decisions/) | Settled decisions as ADRs (`D####-slug.md`) |
| [`changes/`](changes/) | Direction changes / pivots (`C####-slug.md`) |
| [`designs/`](designs/) | Feature design docs (`NN-<topic>.md`), written before implementation |

## Workflow

1. **An open question arises** → create `questions/Q####-...` with context, options,
   and a recommendation.
2. **The user settles it** → write `decisions/D####-...` (ADR), mark the question
   `Status: settled → D####`, and update the master plan (§14 table).
3. **Direction changes** after something was decided or built → write
   `changes/C####-...` (what changed, why, plan impact) and update the master plan.
4. **Feature work starts** → write `designs/NN-...` first, then code; docs and code
   land in the same commit.
5. **Each implementation step:** implement → present for review → **commit only after
   the user approves**. Never commit unreviewed work.
