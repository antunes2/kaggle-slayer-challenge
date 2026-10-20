# D0001 — v1 interaction model: foreground streaming with inline approvals

- **Status:** accepted
- **Date:** 2026-10-19

## Context

The user needs to launch and supervise an autonomous competition loop. Three candidate
interaction models were considered for the first usable version.

## Options considered

1. **Foreground streaming + inline approvals** — one process, no IPC infrastructure;
   agent activity streams live; approvals are prompted inline; `Ctrl+C` = graceful
   pause with checkpointed state. Simplest to build and debug.
2. **Chat-first REPL** — every operation steered by conversation with the orchestrator.
   Most flexible, but costlier (everything through the LLM), weaker structure, harder
   to test.
3. **Background daemon + terminal attach** — most powerful UX (multiple observers,
   detachable), but requires daemonization and a control channel before any feature
   work can start.

## Decision

v1 runs the orchestration loop **in the foreground** (`slayer run`), streaming agent
activity live and prompting approvals inline. Pause = `Ctrl+C` (state checkpointed),
resume via `slayer resume`. Daemon + `attach` + `slayer chat` are deferred to Phase 5.
The chat interaction contract (journal `user.guidance` events consumed at decision
points) is already fixed so nothing blocks that upgrade later.

## Consequences

- No daemon/IPC infrastructure in the MVP; the loop is a single process.
- Approvals block the loop until answered — acceptable in v1, where supervision is
  expected.
- Chat-based steering arrives in Phase 5 without schema rework.
