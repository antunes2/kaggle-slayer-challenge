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

All planning, decisions, and design docs live in [`docs/plan/`](docs/plan/README.md).
Start with the [implementation plan](docs/plan/00-implementation-plan.md).
