# D0002 — Submission autonomy: approve-first, MVP fully mocked

- **Status:** accepted
- **Date:** 2026-10-19

## Context

Kaggle submissions are **irreversible** and quota-limited (~5/day). The MVP pipeline
will be unstable for a while; burning real quota (or risking a bad submission history)
while debugging is unacceptable.

## Options considered

1. **Approve-first** — every submission requires explicit user approval.
2. **Auto after first 2** — user approves the first two; then auto-submit when CV
   improves and the reviewer approves.
3. **Agentic** — never ask; rely on internal gates only.

## Decision

- `approve-first` is the **default autonomy policy** for real submissions.
- Additionally, until Phase 2b ("first live submission"), **all submissions are
  mocked**: a `SubmissionSink` interface with `MockSink` (records the submission in the
  journal + registry with a simulated LB score) and `KaggleSink` (real API, selected
  via `--dry-run=false`). The entire pipeline — approvals, quota manager, monitoring,
  journaling — is exercised end-to-end with zero quota risk.
- `KaggleSink` goes live at Phase 2b, on a Playground Series competition (D0004).

## Consequences

- The `SubmissionSink` abstraction is required in Phase 2; the dry-run mode becomes a
  first-class citizen that also serves integration tests.
- Trust in the loop is earned before autonomy is granted; Phase 3+ can revisit relaxing
  approvals (option 2) with evidence.
