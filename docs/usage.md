# Using Slayer

> Practical guide for humans and LLM agents. Setup, commands, configuration, and
> how to read what the system is doing.

## Setup

```bash
uv sync                              # python ≥3.12, uv-managed
cp .env.example .env                # fill OPENROUTER_API_KEY and KAGGLE_API_TOKEN
uv run slayer init                  # verifies config, secrets, and both API pings
```

One-time per competition: **accept the competition rules in the browser**
(`https://www.kaggle.com/competitions/<ref>/rules`). There is no API for this; INIT
fails fast with the exact URL if you haven't.

## Commands

```bash
# The main command: run the full agent loop on a competition
uv run slayer run playground-series-s6e9
#   --live / --dry-run          real vs mock submissions (default: mock)
#   --autonomy approve-first|agentic   prompt per submission vs auto (default: approve-first)
#   --budget-usd 10             LLM spend cap for the run
#   --submissions-per-day 4      stay under the kaggle quota
#   --harness-timeout-s 1800    per-candidate CV time budget
#   --max-rounds 2              re-entry rounds on target shortfall (0 disables)
#   --refresh-data              force redownload (data cache is per-competition)
#   --config-dir / --env-file   overrides

# Continue a paused/failed/finished run without redoing completed phases;
# a finished-short run continues its campaign (re-entry gate)
uv run slayer resume <run_id>
#   --max-rounds N              raise/lower the run's round cap (persisted)

# Discover competitions (Q0004) — default: playground, latest deadline first
uv run slayer competitions          # --search TEXT  --category playground|featured|all  --sort-by ...  -n 20

# Observability
uv run slayer status                 # list runs (or one run's detail + LLM spend)
uv run slayer journal <run_id>       # event timeline; --limit N
uv run slayer leaderboard <comp>    # top-10 + our position

uv run slayer init                  # environment check
```

## What a run looks like

```
[phase] init            competition resolved; rules probe (fail-fast if not accepted)
[phase] data_collection  download or cache reuse (~seconds when cached)
[phase] analysis        profiler (deterministic) + data analyst agent
[phase] planning        scout digests top public notebooks; then strategy: CV protocol,
                        roadmap, submission gate, guidance
[phase] baseline        researcher writes pipeline.py #1; harness scores it
[phase] iterate         one candidate per roadmap step; repair-retry; review+submit
[phase] review          reviewer agent gates the best unreviewed candidate
[phase] submit          CV-gate → quota-gate → approval-gate → sink
[phase] monitor         real LB scores; position vs target percentile; gap to target line
[campaign] / [round N]   re-entry gate: target unmet + rounds/budget left → round N:
[phase] replan (round 1) planner extends the roadmap from history + gap (or stops)
[phase] iterate (round 1) … review → submit → monitor, then the gate again
[done] all phases completed  → run status: finished
```

Re-entry also runs on dry runs: with no real LB score, MONITOR estimates the position
from the best **CV** (`basis: cv`), so a mock campaign exercises the full loop. Set
`--max-rounds 0` for a single pass.

Quiet stretches are normal: `cand-000N: evaluating (5 folds, budget 1800s) — silence
during CV is normal` means a sandboxed harness is training models. Failures print the
error's first line and trigger one repair attempt per step.

A run takes minutes to ~an hour depending on dataset size, roadmap depth, and the
harness budget. Everything is resumable — Ctrl+C and resume later is safe (the run
status will read `active` until resumed; that's a known cosmetic gap).

## Configuration

`config/slayer.yaml` (run defaults — every CLI flag overrides these):

```yaml
defaults:
  target_percentile: 20     # the goal MONITOR reports against
  budget_usd: 10.0          # LLM spend cap
  submissions_per_day: 4    # real submission cap (kaggle allows ~5-10)
  autonomy: approve-first   # approve-first | agentic
  dry_run: true            # mock submissions by default — going live is explicit
  refresh_data: false
  harness_timeout_s: 1800   # per-candidate CV budget
  max_rounds: 2             # re-entry rounds on target shortfall (0 disables)
paths:
  runs_dir: runs
llm:
  base_url: https://openrouter.ai/api/v1
  timeout_s: 180
  max_retries: 3
```

`config/agents.yaml` (model routing; OpenRouter ids):

```yaml
defaults:
  model: google/gemini-2.5-flash      # fallback for any agent
  temperature: 0.2
  max_tokens: 16384                    # reasoning models need headroom
orchestrator: {model: google/gemini-2.5-flash-lite}
planner: {}                           # inherits defaults
data_analyst: {model: google/gemini-2.5-flash-lite}
ml_researcher: {model: anthropic/claude-sonnet-5.5}   # upgraded from gpt-5-mini (A/B)
reviewer: {temperature: 0.0}          # adversarial, deterministic
```

Prompts live in `prompts/*.md` — editing them changes agent behavior immediately;
they are version-controlled like code (and most behavioral lessons live in them).
`prompts/planner_extend.md` is the planner's re-entry prompt (same model routing as
`planner`). Keep prompt rules **generic for tabular data** — competition specifics
should arrive through the data profile and the notebook digest (C0004).

## Reading a run

- `slayer status <id>` — phase history, pause reason (paused runs only), LLM spend vs
  budget (journal-summed, across resumes), re-entry rounds + last campaign decision,
  candidates by lifecycle state, latest CV.
- `slayer journal <id>` — every event: agent calls (with tokens/cost), phase
  transitions, review verdicts, submissions, monitor positions, re-entry decisions.
- Artifacts under `runs/<id>/workspace/` — `reports/strategy_plan.md` (what the agents
  decided, including appended re-entry steps), `reports/roadmap_round_N.json` (each
  round's assessment + steps), `eda/profile.md` (the data), `models/<cand>/pipeline.py`
  + `cv_results.json` (what they built and how it scored).

## Troubleshooting

- **403 on download/INIT** — accept the competition rules (URL in the error).
- **HTTP 429 from OpenRouter** — provider rate limit; the client backs off 15/30/45s
  automatically; if a run dies anyway, just `slayer resume` it.
- **`connection error: ... name resolution`** — the internet dropped; the client now
  waits 15 → 30 → 60 → 120 → 240 s before giving up (~7.5 min). Longer outages still
  fail the run — `slayer resume` it.
- **`scout: no public notebook digest available`** — harmless; planning proceeds from
  the data alone (new competitions often have few notebooks).
- **`exhausted 3 attempts`** — a roadmap step failed initial + repair + retry; inspect
  `workspace/models/*/run.log`, fix the prompt or config, resume.
- **Stuck `active` after Ctrl+C** — cosmetic; `slayer resume <id>` picks up cleanly.
- **Orphan harness process** (rare, after killing the CLI mid-CV) —
  `pkill -9 -f slayer.cv.harness`.
