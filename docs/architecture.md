# Slayer Architecture

> Reference for humans and LLM agents. Read this before the code. For history and
> rationale see `docs/plan/` (decisions, designs, changes); this file describes the
> system *as it is*.

## What this is

**Slayer** is a CLI that runs a team of LLM agents through a deterministic pipeline to
participate in Kaggle competitions: collect data → analyze → plan → build & score
models → review → submit → monitor, aiming for top 20% with bounded cost and safe
defaults.

## The one design principle everything follows

**LLM judgment rides on deterministic scaffolding.** Agents never touch raw data,
never compute their own scores, never talk to each other directly, and never perform
irreversible actions ungated. Code (tested, ours) owns protocol, transport, and
bookkeeping; the LLM owns interpretation, ideas, and judgment. Every live incident so
far has reinforced this split (see `docs/plan/designs/` retros).

Consequences you can rely on:

- Scores come only from the CV harness — a pipeline that doesn't run can't be a candidate.
- Every LLM output is schema-validated (pydantic); invalid output fails loudly.
- Every state change is an append-only journal event (the source of truth); the SQLite
  registry is a rebuildable view (ADR D0005).
- Every submission passes three gates: CV-improvement → quota → human approval
  (unless `--autonomy agentic`); mock submissions (the default) never touch Kaggle.

## Component map

```
src/slayer/
├── cli/main.py        # typer app: run, resume, init, status, journal, leaderboard,
│                      # competitions, dashboard
├── dashboard.py       # cross-run summaries → terminal table / self-contained HTML
├── config.py          # slayer.yaml + agents.yaml + .env → pydantic settings
├── llm/client.py      # OpenRouter client: retries (429-aware; long backoff on
│                      # connection blips), JSON extraction,
│                      # token/cost accounting (estimates when provider omits cost)
├── kaggle/            # KaggleClient (thin, defensive SDK wrapper; incl. public
│                      # notebook listing/source), QuotaManager
├── analysis/          # deterministic profiler → profile.{json,md} + markdown renders;
│                      # notebooks (public-notebook scout → capped digest)
├── agents/            # Agent base (prompt + schema + journaling) + the five roles
├── loop/              # engine (phase machine + bounded re-entry rounds), phases (the
│                      # 9 handlers), campaign (re-entry gate + REPLAN), blending
│                      # (OOF blend step), external (public datasets, D0009), data cache
├── cv/                # metric registry, the CV harness (sandboxed subprocess; persists
│                      # OOF predictions), runner (timeout kill, partial results),
│                      # blend (deterministic hill-climb over OOF)
├── submissions.py     # SubmissionSink: MockSink (default) | KaggleSink (real)
├── schemas/           # pydantic contracts for every agent output
└── workspace/         # RunStore (run dirs), Journal (JSONL), CandidateRegistry (SQLite)
prompts/               # system prompts per agent (markdown, version-controlled)
config/                # slayer.yaml (run defaults), agents.yaml (model routing)
runs/<run_id>/         # artifacts (gitignored)
```

## The run loop (short version — deep dive in docs/loop.md)

Nine phases, in order, each a handler in `loop/phases.py`:

```
INIT → DATA_COLLECTION → ANALYSIS → PLANNING → BASELINE → ITERATE → REVIEW → SUBMIT → MONITOR
   then, while the re-entry gate allows (D0007, ≤ max_rounds):
REPLAN → ITERATE → REVIEW → SUBMIT → MONITOR → … → finished
```

- Deterministic gates; LLM decides *content* within a phase (ADR D0003).
- A phase resumes by skipping its own completed work (skip predicates + registry).
- ITERATE contains the improvement loop: one candidate per pending roadmap step,
  repair-retry on failure, review + submit gates after each candidate, LLM budget
  stops, one post-loop repair for unactioned REVISE feedback.
- MONITOR reads the real world back: LB scores for submitted candidates, position
  vs the run's target percentile, and the **score gap** to the target line.
- Re-entry (`loop/campaign.py`): a deterministic gate (target unmet, rounds/budget
  left, deadline allows, planner hasn't stopped) → REPLAN, where the planner extends
  the roadmap from the full experiment history + gap, or stops the campaign.

## Data flow

```
User ──slayer run──► engine ──► phases ──► agents ──(LLM)──► schema-validated outputs
                     │                        │
                     ├──► kaggle client ◄──────┤  (downloads, submits, leaderboards,
                     │                        │   public notebooks → scout digest)
                     └──► harness (subprocess) ─┘  (CV scores, submissions files)

Everything lands in runs/<run_id>/:
├── run.json            # status + config snapshot
├── journal.jsonl       # append-only events (audit trail + agent blackboard)
├── experiments.db      # candidate registry (rebuildable from the journal)
├── workspace/
│   ├── data/           # competition CSVs + manifest.json
│   ├── eda/            # profile.json / profile.md
│   ├── external/       # public datasets (D0009) + manifest.json (sources, files, columns)
│   ├── models/<cand>/  # pipeline.py, cv_config.json, cv_results.json, submission.csv, run.log,
│   │                   # oof.npy + fold_ids.npy (blendable metrics); blends: blend.json
│   └── reports/        # competition.json, data_report.json, feature_plan.json, strategy_plan.json,
│                       # notebook_digest.md, roadmap_round_N.json, leaderboard.csv
```

## The five agents

| Agent | role | reads | writes |
|---|---|---|---|
| data_analyst | interprets the deterministic profile | profile.md, competition.json | DataReport, FeaturePlan |
| planner | strategy for this competition; re-entry roadmap extensions | analyst outputs, competition, public-notebook digest; history + gap (REPLAN) | StrategyPlan; RoadmapExtension |
| ml_researcher | writes pipeline.py plugins | strategy + feedback from registry/reviews | PipelineProposal (code) |
| reviewer | adversarial gate before submissions | candidate code + CV + guidance | ReviewVerdict (approve/reject/revise) |
| orchestrator | reserved (Phase 5) | — | — |

Model routing is per-agent in `config/agents.yaml` (OpenRouter ids). The researcher
is the highest-leverage model (A/B in `docs/plan/questions/Q0001`). How the agents
*choose* ideas matters more than model tier — see `docs/plan/changes/C0004`.

## Safety model

- **Mock by default**: `dry_run: true` routes submissions to MockSink — zero quota risk.
- **Approve-first default**: real submissions prompt y/n in the terminal; `agentic` skips.
- **Quota**: `QuotaManager` counts real submissions per UTC day vs the configured cap.
- **LLM budget**: cumulative cost summed from the journal's `agent.call` events (real or
  catalog-estimated) — survives resumes — stops ITERATE and blocks re-entry.
- **Harness sandbox**: per-candidate subprocess, process-group SIGKILL on timeout,
  SIGALRM self-deadline (best-effort), probe-fold early abort for pathological candidates.
- **Untrusted external text**: the public-notebook digest is built by code, never
  executed, and framed to the planner as evidence, not instructions; ideas that need
  external data are tagged `[EXTERNAL]`.
- **External data** (D0009): only public Kaggle datasets that top notebooks load;
  `auto` = Playground only (rules can't be read by API); external rows join training
  folds only, never validation.
- **Secrets**: `.env` only (gitignored); never printed.

## Known limitations (honest)

- Full inventory: [`docs/limitations.md`](limitations.md).
- Ctrl+C pauses the run and kills the running harness; a *hard* kill of the CLI can
  still orphan a harness (SIGALRM cannot interrupt long C fits — Q0003).
- MONITOR matches leaderboard entries by username/display-name heuristics.
- OOF blending covers binary-probability metrics and regression only; candidates
  scored before design 14 have no OOF and are never blended.
- Single foreground process; daemon + attach + chat are Phase 5.
