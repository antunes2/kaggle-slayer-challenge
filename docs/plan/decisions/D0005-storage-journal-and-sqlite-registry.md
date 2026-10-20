# D0005 — Storage: JSONL journal for events + SQLite registry for entity state

- **Status:** accepted
- **Date:** 2026-10-19

## Context

A run needs two different things from storage:

1. **Agent communication (blackboard) + audit trail** — append-only, immutable facts:
  what agents produced, what was decided, what happened. Must be LLM-consumable
  (bounded context windows) and human-greppable.
2. **Candidate registry** — mutable entity state. A candidate has a lifecycle
   (proposed → scored → reviewed → submitted → LB-scored hours later) and is queried
   constantly by loop logic: best CV, only-if-improved submit policy, dedupe,
   lineage.

## Options considered

1. **JSONL only** — great as an event log; poor for entity state (updates require
   tombstones or full re-reads, queries mean scan + parse in Python, memory grows with
   run length).
2. **SQLite only** — great for entity state and queries; poor as an agent-facing
   communication channel (binary, not LLM-readable, not diff-friendly).
3. **Both, with strict roles, event-sourced** — JSONL journal is the source of truth;
   SQLite is a materialized view over it, rebuildable by replay.

## Decision

**Both, event-sourced:**

- `runs/<run_id>/journal.jsonl` — append-only immutable events; the agent blackboard
  and audit trail; **source of truth**.
- `runs/<run_id>/experiments.db` — SQLite (stdlib `sqlite3`, WAL mode, stdlib-only DAO,
  no ORM) holding candidate entity state; every registry mutation is *also* a journal
  event, so the DB can be rebuilt by journal replay.

## Consequences

- Candidate updates are atomic `UPDATE`s; submit policy / reviewer / dedupe queries
  become indexed SQL.
- Agents never consume raw registry dumps: a `registry.summary_for_agent()` renders
  compact markdown tables — deliberate, bounded context (a feature, not a workaround).
- Small redundancy cost (journal event + DB write per mutation) — accepted for
  recoverability and auditability.
- Single writer in v1 (the loop process); WAL is belt-and-suspenders for crash safety.
