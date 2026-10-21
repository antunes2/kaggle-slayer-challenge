# Decisions (`decisions/`)

ADR-style records of **settled** decisions. One file per decision, named
`D####-slug.md`, numbered sequentially. A decision is only recorded here once the user
has settled it — open proposals live in [`../questions/`](../questions/).

## Template

```markdown
# D#### — <title>

- **Status:** accepted | superseded by D####
- **Date:** YYYY-MM-DD

## Context
The problem being decided and the forces at play.

## Options considered
Alternatives and their tradeoffs.

## Decision
What we chose and why.

## Consequences
What this implies — including downsides we consciously accept.
```

## Index

- [D0001 — v1 interaction model: foreground streaming with inline approvals](D0001-interaction-model-v1.md)
- [D0002 — Submission autonomy: approve-first, MVP fully mocked](D0002-submission-autonomy-mock-sink.md)
- [D0003 — Orchestrator pattern: deterministic pipeline with LLM decision points](D0003-orchestrator-deterministic-pipeline.md)
- [D0004 — MVP target: Playground Series tabular competition](D0004-mvp-target-playground-competition.md)
- [D0005 — Storage: JSONL journal for events + SQLite registry for entity state](D0005-storage-journal-and-sqlite-registry.md)
- [D0006 — Model routing: fast Gemini Flash tier as default](D0006-model-routing-gemini-flash.md)
