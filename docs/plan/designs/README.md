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
