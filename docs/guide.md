# Operator Guide — Running Slayer Step by Step

> A complete runbook for running Slayer yourself, from a fresh machine to a multi-day
> Kaggle campaign. Reference material (every flag, config key, troubleshooting) lives
> in [`usage.md`](usage.md); what Slayer does *not* guarantee is in
> [`limitations.md`](limitations.md).

## 0. What you need

| Need | Why | How |
|---|---|---|
| Linux or macOS, Python ≥ 3.12 | runtime | `python3 --version` |
| [`uv`](https://docs.astral.sh/uv/) | installs and runs everything | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| An OpenRouter account **with credits** | the agents' AI models | create a key at https://openrouter.ai/keys; typical campaign ≈ $1 |
| A Kaggle account allowed to submit | data, submissions, leaderboard | Kaggle requires a verified phone number to submit |
| A Kaggle API token | programmatic access | kaggle.com → your profile → **Settings** → **API** → create a new token |
| Disk + CPU | training runs locally | ~1 GB per competition; training uses all cores |

## 1. One-time setup

```bash
cd kaggle-slayer
uv sync                          # creates .venv and installs all dependencies
cp .env.example .env             # then edit .env:
#   OPENROUTER_API_KEY=sk-or-...
#   KAGGLE_API_TOKEN=...
uv run slayer init               # checks config files, keys, and pings both APIs
```

`slayer init` must end with **Environment ready.** If a line is red, fix that item
(usually a missing/mistyped key) and run it again. `.env` is git-ignored — never
commit or share it.

## 2. Pick a competition

```bash
uv run slayer competitions                   # Playground, latest deadline first
uv run slayer competitions -c all -s "credit" # all categories, search "credit"
```

Pick a **tabular** competition whose deadline is in the future, with a
`train.csv` / `test.csv` / `sample_submission.csv` layout (Playground episodes fit).
Copy its ref (e.g. `playground-series-s6e10`).

**Accept its rules once in the browser** — there is no API for this:
`https://www.kaggle.com/competitions/<ref>/rules`. Slayer stops immediately with
this URL if you forget.

## 3. First run: a dry run (no real submissions)

```bash
uv run slayer run <ref>
```

- Submissions are **simulated** by default — nothing reaches Kaggle, no quota used.
- It still asks `Submit ...? [y/N]` (mock) — answer `y` to exercise the flow.
- Takes ~1-4 h depending on data size. Long silent stretches during
  `evaluating (5 folds ...)` are normal: a model is training.
- It ends with `run finished: <run_id>`. Note the run id.

**Unattended (e.g. overnight) dry run** — skip the y/n questions:

```bash
uv run slayer run <ref> --autonomy agentic
```

`agentic` is safe only together with dry runs; for live runs keep the default so you
approve each real submission.

## 4. Live campaign (real submissions)

```bash
uv run slayer run <ref> --live
```

- Before **every real submission** it asks `Submit cand-000X (...)? [y/N]`:
  `y` = submit, Enter = skip. Stay near the computer until the first question
  (usually 10-30 min in).
- At most **4 real submissions per day** (config `submissions_per_day`); after that it
  says `daily submission quota exhausted` and keeps the candidate for later.
- After the first pass it reads the real leaderboard, and if short of the target
  (top 20%) it re-plans and runs up to 2 more rounds (`max_rounds`), then finishes.
- Real leaderboard scores can take a few minutes to appear; `no LB score yet` is
  normal — the next check (or resume) picks them up.

## 5. Continue the next day (fresh quota)

```bash
uv run slayer status                                    # find your run id
uv run slayer resume <run_id> --max-rounds 3            # +1 round vs yesterday
```

`resume` keeps all history (no repeated work). First it submits the best candidate
it could not submit before (if any), then runs new rounds. Raise `--max-rounds` by one
or two each day (3, 4, 5…) until the deadline. Each extra round costs ≈ $0.40 and
~1 h of compute.

## 6. Stopping safely

- **Ctrl+C** at any time (even at a y/n question) → the run is marked `paused` and the
  training subprocess is stopped. It prints the `slayer resume <run_id>` command.
- If the terminal was killed hard (closed window, power loss): `slayer resume` still
  works; if the computer stays busy, stop leftovers with `pkill -9 -f slayer.cv.harness`.

## 7. Watching and reading results

```bash
uv run slayer status <run_id>          # phases, LLM spend vs budget, rounds, candidates
uv run slayer journal <run_id> -l 40   # last 40 events (verdicts, submissions, decisions)
uv run slayer leaderboard <ref>        # top 10 + your position
uv run slayer dashboard                # every run: candidates, submissions, best CV/LB, position
uv run slayer dashboard --html report.html   # dark interactive report (campaigns + comparison)
```

Files of interest in `runs/<run_id>/workspace/`:

| File | What it tells you |
|---|---|
| `reports/strategy_plan.md` | the plan (roadmap steps, rules for the researcher) |
| `reports/roadmap_round_N.json` | each re-entry round's diagnosis and new steps |
| `reports/notebook_digest.md` | what the top public notebooks do (the scout's evidence) |
| `external/manifest.json` | outside datasets used, with their sources |
| `models/cand-NNNN/pipeline.py` | the code a candidate ran |
| `models/cand-NNNN/cv_results.json` | its fold scores, timing, errors |
| `eda/profile.md` | the dataset profile |

## 8. Everyday knobs

Pass as flags (one run) or edit `config/slayer.yaml` (all future runs):

| Want | Flag | Config key |
|---|---|---|
| spend cap | `--budget-usd 5` | `budget_usd` |
| more/fewer re-entry rounds | `--max-rounds 4` (`0` = single pass) | `max_rounds` |
| real vs simulated submissions | `--live` / `--dry-run` | `dry_run` |
| skip y/n prompts | `--autonomy agentic` | `autonomy` |
| per-model time limit | `--harness-timeout-s 1800` | `harness_timeout_s` |
| daily submission cap | `--submissions-per-day 3` | `submissions_per_day` |
| outside datasets | `--external-data` / `--no-external-data` | `external_data` (`auto` = Playground only) |
| re-download data | `--refresh-data` | `refresh_data` |

**Outside data:** on automatically only for Playground competitions. For any other
competition, read its rules before passing `--external-data` — breaking a
competition's external-data rule can get you disqualified.

**AI models** per agent are in `config/agents.yaml` (OpenRouter model ids). The
researcher (`ml_researcher`) is where most of the spend goes; the planner is the
cheapest upgrade with plausible upside (see `limitations.md` §8).

## 9. When something goes wrong

| Symptom | What to do |
|---|---|
| `403` / rules message at start | accept the competition rules in the browser (§2) |
| `connection error ... name resolution` | internet dropped for > 7 min; reconnect, `slayer resume <run_id>` |
| `HTTP 402` / credit errors from OpenRouter | add credits, then resume |
| `exhausted 3 attempts` | one roadmap step failed 3 times; read `models/*/run.log`, then resume |
| `daily submission quota exhausted` | normal; resume tomorrow |
| run status `active` but nothing running | it was hard-killed; `slayer resume <run_id>` |
| computer stays busy after a crash | `pkill -9 -f slayer.cv.harness` |

More: [`usage.md` → Troubleshooting](usage.md#troubleshooting).

## 10. For developers

```bash
uv run pytest -q          # unit tests (~1 min)
uv run ruff check .       # lint
```

Read `AGENTS.md` and `docs/plan/PROJECT-STATUS.md` before changing code.
