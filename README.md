# kaggle-slayer

A CLI that runs a team of LLM agents through Kaggle tabular competitions end to end:
download data → analyze it → read the competition's top public notebooks → plan →
write and cross-validate model pipelines → review them → submit (gated) → read the
leaderboard back → re-plan when short of the target. Bounded cost, safe defaults.

**Design principle:** LLM judgment rides on deterministic scaffolding. Agents propose
ideas and code; *code* computes every score, owns every gate, and journals every event.

## What it achieves (honestly)

- Fully autonomous, resumable campaigns with real Kaggle submissions (live-proven).
- On `playground-series-s6e9`: first real LB ≈ top 71%; best CV after the
  evidence-driven upgrades ≈ top 45%. **The top-20% goal is not reached.** See
  [docs/limitations.md](docs/limitations.md) before relying on it.

## Quickstart

```bash
uv sync
cp .env.example .env              # fill OPENROUTER_API_KEY and KAGGLE_API_TOKEN
uv run slayer init                # checks config, secrets, and both APIs
```

## A typical campaign

```bash
uv run slayer competitions                         # browse active competitions
# accept the rules once in the browser: https://www.kaggle.com/competitions/<ref>/rules
uv run slayer run <ref>                            # dry run first (mock submissions)
uv run slayer run <ref> --live                     # real submissions; asks y/n for each
uv run slayer status <run_id>                      # progress, spend, rounds, candidates
uv run slayer resume <run_id>                      # continue (e.g. once a day for fresh quota)
```

Ctrl+C pauses safely; `resume` continues where it stopped.

## Safety defaults

| Default | Meaning |
|---|---|
| `dry_run: true` | submissions are simulated unless you pass `--live` |
| `autonomy: approve-first` | every real submission needs your y/n (`agentic` skips it) |
| `submissions_per_day: 4` | daily cap, counted from the journal |
| `budget_usd: 10` | LLM spend cap per run, cumulative across resumes |
| `max_rounds: 2` | re-entry rounds when short of the target |

## Documentation

- [Architecture](docs/architecture.md): components, data flow, safety model
- [The run loop](docs/loop.md): phases, gates, resume semantics, events
- [Usage](docs/usage.md): commands, configuration, troubleshooting
- [Limitations & known gaps](docs/limitations.md): what is not guaranteed
- [Project status](docs/plan/PROJECT-STATUS.md): where things stand
- [`docs/plan/`](docs/plan/README.md): decisions, designs, direction changes

## Development

```bash
uv run pytest -q        # unit tests (integration tests are env-gated)
uv run ruff check .
```
