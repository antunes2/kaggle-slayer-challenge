# 02 — Journal, Run Store & Candidate Registry

- **Phase:** 0 (step 0.3) · **Status:** implemented · **Date:** 2026-10-19
- **ADR:** [D0005](../decisions/D0005-storage-journal-and-sqlite-registry.md) (event-sourced storage)

## Goal

The run's persistence layer, with the two strict roles D0005 defines:

1. **Journal** (`runs/<run_id>/journal.jsonl`) — append-only immutable events; the agent
   blackboard, audit trail, and **source of truth**.
2. **Candidate registry** (`runs/<run_id>/experiments.db`, SQLite) — candidate entity
   state for queries; a materialized view rebuildable by journal replay.
3. **Run store** — run lifecycle scaffolding: `run.json`, workspace dirs, finding runs.

## Interface

```python
Journal(path)
  .append(agent, type, data=None, *, run=None) -> JournalEvent
  .read(*, type=None, agent=None, limit=None) -> list[JournalEvent]
  .tail(n=20) -> list[JournalEvent]

RunStore(runs_dir)
  .create(competition, defaults: DefaultsConfig) -> Run      # makes dirs + run.json + run.created event
  .get(run_id) / .list() / .update_status(run, status)
  .journal(run) -> Journal
  .registry(run, journal=None) -> CandidateRegistry          # experiments.db wired to the journal

CandidateRegistry(db_path, *, run_id, journal=None)
  .add(ModelCandidate) -> CandidateRecord                   # cand-0001, status=proposed
  .get(id) / .list(status=None)
  .record_cv(id, score, std=None)                           # auto: proposed -> scored
  .set_status(id, status)                                   # proposed|scored|approved|rejected|submitted|retired
  .record_lb(id, lb_score, submission_id)                   # auto: -> submitted
  .best(n=1, *, higher_is_better=True, submitted_only=False, unsubmitted_only=False)
  .rebuild(db_path, *, run_id, journal)                     # replay candidate.* events
```

## Decisions

- **Registry emits its own journal events.** Callers can't forget to journal a mutation;
  rebuildability is guaranteed by construction. Infra events use `agent="system"`.
- **Metric-agnostic ranking.** `best()` takes `higher_is_better` — the metric direction
  is decided by the planner (Phase 2+), the registry stays dumb.
- **Auto-transitions** on the lifecycle: `proposed → scored` (on CV), `→ submitted`
  (on LB record). Explicit `set_status` for reviewer verdicts (approved/rejected).
- **`cand-*` events carry the full record dump** (`candidate.added`), so replay is a
  straight `model_validate` + partial-update handlers.
- **Run ids**: `{YYYYMMDD-HHMMSS}-{slug}[-n]` — sortable, human-readable, collision-suffixed.
- **`run.json` writes are atomic** (temp + rename) — a crash can't leave a half-written
  run metadata file.
- Event types live as free-form strings with an `EventType` constants class — extensible
  without schema migrations; known types: `run.created`, `run.status_changed`,
  `phase.enter`, `agent.call`, `agent.output`, `candidate.added`,
  `candidate.status_changed`, `candidate.scored`, `candidate.lb_scored`,
  `submission.accepted`, `user.guidance`, `budget.warning`.

## Tests

- Journal: append/read round-trip, type/agent filters, limit/tail, schema defaults.
- Run store: dirs + run.json created, `run.created` event, get/list, missing run,
  status persistence, id collision suffixing.
- Registry: add/get, sequential ids, CV auto-score, LB auto-submit, status filters,
  best() in both metric directions, unsubmitted-only, event emission, and the key D0005
  test — **rebuild-from-journal equivalence** (full record list equality after replay).

## Retro

Implemented as designed. Notes:
- Rebuild replay reuses the exact internal mutators the public API uses, so drift
  between write path and replay path is structurally impossible.
- WAL mode + short transactions; v1 is single-writer anyway (the loop process).
- Model placement: `JournalEvent`/`Run`/`CandidateRecord` live next to their components
  (not in a central schemas package) — agent-output schemas (StrategyPlan etc.) will
  get `slayer/schemas/` in Phase 1 per plan §7.
