# Feature Designs (`designs/`)

One design doc per significant feature, written **before** implementation and updated
after. Naming: `NN-<topic>.md`, where `NN` orders by implementation sequence
(01 = first feature).

A design doc answers: goal, user-facing interface (CLI/config/schema), data model,
error handling, tests, and open questions. When the implementation reveals design
errors, finish with a short **Retro** section — the doc stays true, not aspirational.

## Index

- [01 — LLM Client (OpenRouter)](01-llm-client.md)
- [02 — Journal, Run Store & Candidate Registry](02-journal-and-registry.md)
- [03 — Kaggle Client & Quota Manager](03-kaggle-client.md)
- [04 — Run Loop Engine, INIT & DATA_COLLECTION](04-run-loop-and-data-collection.md)
- [05 — ANALYSIS: Deterministic Profiler + Data Analyst Agent](05-analysis-and-data-analyst.md)
- [06 — CV Harness & Sandbox Runner](06-cv-harness.md)
- [07 — PLANNING: Planner Agent → StrategyPlan](07-planner.md)
- [08 — BASELINE: Researcher Codegen → First Scored Candidate](08-researcher-and-baseline.md)
- [09 — REVIEW & SUBMIT: Reviewer Gate + Submission Sinks (Mock-first)](09-review-and-submission.md)
- [10 — ITERATION: The Improvement Loop](10-iteration-engine.md)
- [11 — RESUME: Continue Without Redoing](11-resume.md)
- [12 — MONITOR: LB-Score Collection & Position Awareness](12-monitor.md)
- [13 — RE-ENTRY: Bounded Rounds on Target Shortfall](13-reentry-rounds.md)
