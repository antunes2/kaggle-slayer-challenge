# AGENTS.md

## Workflow

- **Never commit before user review.** Each implementation step: implement → present the
  changes for review → commit only after the user approves.
- **Do not run the live pipeline by default** — it is slow (harness CV takes minutes).
  Validate with unit/integration tests; ask the user before any live `slayer run` and
  say why it is needed.
- Each step is one logical unit of work; code and its docs land in the same commit.
- **Every function gets a short docstring** — basic documentation of what it does; not
  perfect, just present.
- **At every phase completion**, report to the user: what was implemented (with the key
  design decisions), the current risks, and opportunities/new routes worth considering
  (including feature removal) — with reasoned explanations, so the user can steer.
- Project documentation lives in `docs/plan/` — update the master plan (statuses,
  decisions, phase step breakdowns) as part of any step that completes or changes work.
  See `docs/plan/README.md` for the question / decision / change / design conventions.

## Project

- Python ≥3.12 managed with **uv** — always `uv sync` after dependency changes and run
  everything through `uv run <cmd>`.
- Entry point: the `slayer` CLI (typer) → `uv run slayer ...`.
- Config: `config/slayer.yaml` + `config/agents.yaml`; secrets in `.env` (gitignored,
  never printed or committed).

## Checks (run before presenting any step for review)

```bash
uv sync
uv run pytest -q
uv run ruff check .
```
