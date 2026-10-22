# kaggle-slayer

A team of LLM agents that autonomously participates in Kaggle competitions — collecting
data, training models, submitting solutions, and monitoring the leaderboard — aiming for
the top 20% with bounded cost and safe defaults.

## Quickstart

```bash
uv sync
cp .env.example .env     # then fill in OPENROUTER_API_KEY and KAGGLE_API_TOKEN
uv run slayer init       # verify the environment
```

## Documentation

Human- and LLM-friendly references:

- [Architecture](docs/architecture.md) — system design, components, safety model
- [Usage](docs/usage.md) — commands, configuration, troubleshooting
- [The run loop](docs/loop.md) — phases, gates, resume semantics
- [Project status](docs/plan/PROJECT-STATUS.md) — where we stand and what's next

Planning, decisions, and design history live in [`docs/plan/`](docs/plan/README.md).
