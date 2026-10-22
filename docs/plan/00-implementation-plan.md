# Kaggle Slayer — Implementation Plan

> **Status:** v2.2 (all 9 phases live-proven; first real submission scored; step 4.2 bounded re-entry rounds implemented, live validation pending) · **Updated:** 2026-10-22 · **Owner:** Gabriel
>
> This is the living master plan. It is updated as the project evolves. Feature-level
> design docs live alongside it as numbered files (see §15).

---

## 1. Vision

**Slayer** is a CLI-first system where a team of LLM agents autonomously participates in
Kaggle competitions — collecting data, training models, submitting solutions, and
monitoring the leaderboard — aiming for **top 20%** with bounded cost and safe defaults.

The user's mental model: *run one command, watch a team of AI specialists work, intervene
only when you want to.*

## 2. Goals & Success Metrics

| # | Goal | Success metric |
|---|------|----------------|
| G1 | Autonomously complete a full Kaggle loop (data → model → submit → monitor) | End-to-end run on a real competition with ≥1 accepted submission |
| G2 | Reach top 20% on target competitions | Final leaderboard percentile ≤ 20 on ≥1 competition |
| G3 | Safe & cheap by default | No submission quota waste; per-run USD budget enforced |
| G4 | Observable & interruptible | Every decision is in a journal; runs can be paused/resumed |
| G5 | Simple UX | One command to start; zero commands to let it finish |

## 3. Non-Goals (v1)

- No GUI / web dashboard (terminal only).
- No deep learning training (tabular-first: sklearn / LightGBM / CatBoost cover most of the top-20% space for tabular comps).
- No multi-machine scaling — single machine, single process (v1).
- No auto-selection of *which* competition to enter (user picks; auto-suggestion is v2+).

## 4. Core Concepts

| Concept | Definition |
|---------|------------|
| **Orchestrator** | The team-lead agent. Owns the loop, routes work to agents, reports to the user. The *only* agent the user talks to. |
| **Agent** | An LLM persona (planner, data analyst, ML researcher, reviewer) with a system prompt, a model, a schema-validated output, and access to the workspace. |
| **Run** | One participation attempt on one competition. Has an id, state, workspace, journal, and budgets. |
| **Workspace** | Directory per run holding data, generated code, reports, models, submissions. |
| **Journal** | Append-only JSONL event log — the blackboard agents read/write and the audit trail the user reviews. |
| **Candidate** | A concrete model pipeline (feature code + model code + params + local CV score) eligible for submission. |
| **Phase gate** | A deterministic checkpoint in the loop (e.g., "baseline done"). Progress is code-controlled; *content* within a phase is LLM-driven. |

## 5. System Architecture

### 5.1 Diagram

```
            ┌─────────────────────────────────────────────┐
            │                 User (CLI)                   │
            │  slayer run / status / chat / pause / ...   │
            └───────────────────────┬─────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │   ORCHESTRATOR    │   team lead
                          │   (loop owner)    │   decision-maker
                          └─────────┬─────────┘
        ┌──────────┬──────────┬──────┴──────┬──────────────┐
        ▼          ▼          ▼             ▼              ▼
    PLANNER   DATA ANALYST  ML RESEARCHER  REVIEWER   (future roles)
        │          │          │             │
        └──────────┴─────┬────┴─────────────┘
                         ▼
           Shared Workspace + Journal (blackboard)
                         │
          ┌──────────────┴───────────────┐
          ▼                              ▼
    Kaggle API client              LLM client (OpenRouter)
    download · submit · LB         per-agent model routing
          │                        + structured outputs
          ▼
    Local execution sandbox
    (generated Python runs in a subprocess; Docker later)
```

### 5.2 The orchestration loop

Deterministic phase gates, LLM-driven content within each phase.
Phase order per [C0002](changes/C0002-loop-phase-reorder.md) — planning happens
**after** EDA so the planner sees the DataReport:

```
INIT ──► DATA_COLLECTION ──► ANALYSIS ──► PLANNING ──► BASELINE ──► ITERATE ◄───┐
            (download)         (EDA)      (strategy)   (first CV)     │         │
                                                                     ▼         │
                                                          REVIEW ─► SUBMIT ─► MONITOR
                                                                     (until: target % reached,
                                                                      deadline, or budget out)
```

Loop responsibilities per phase:

1. **INIT** — create run + workspace, fetch competition metadata (title, metric, deadline, quota), write `reports/competition.json`.
2. **DATA_COLLECTION** — download competition data via Kaggle API, safe-unzip, write sha256 manifest.
3. **ANALYSIS** — deterministic profiler computes the dataset statistics; the data analyst agent interprets them → `DataReport` + `FeaturePlan` (schema-validated, per [design 05](designs/05-analysis-and-data-analyst.md)).
4. **PLANNING** — planner agent reads comp metadata **+ DataReport** → `StrategyPlan` (schema-validated): problem type, metric, CV protocol, modelling roadmap, submission policy.
5. **BASELINE** — ML researcher writes a minimal pipeline + CV harness → first `Candidate` with local CV score.
6. **ITERATE** — researcher proposes experiments (features, models, tuning, ensembling); each runs locally through the standard CV harness; results tracked.
7. **REVIEW** — reviewer agent inspects best candidate(s) for leakage, CV/LB mismatch risk, quota value → `ReviewVerdict` (approve / reject / revise).
8. **SUBMIT** — approved candidates go to Kaggle (respecting daily quota + only-if-CV-improves rule); scores recorded.
9. **MONITOR** — leaderboard deltas + new public info feed back into planning; orchestrator decides to iterate or hold.

**Design stance:** v1 implements this as a *deterministic pipeline with LLM decision
points* — predictable, testable, cheap. Evolving the orchestrator itself into a fully
agentic tool-using agent (dynamic replanning) is a later milestone (§11, Phase 5).

### 5.3 Agent team

| Agent | Prompt file | Produces | Notes |
|-------|-------------|----------|-------|
| Orchestrator | `prompts/orchestrator.md` | phase transitions, user updates, submit/hold decisions | cheap-ish model, low temperature |
| Planner | `prompts/planner.md` | `StrategyPlan` | needs strong reasoning model |
| Data Analyst | `prompts/data_analyst.md` | `DataReport`, `FeaturePlan` | interprets deterministic profiler output (see design 05) |
| ML Researcher | `prompts/ml_researcher.md` | `ModelCandidate`, `CVResult` | writes feature/training code |
| Reviewer | `prompts/reviewer.md` | `ReviewVerdict` | adversarial: hunts leakage/overfit |

Agents never talk to each other directly; all communication happens through the
**workspace** (files) and the **journal** (events), routed by the orchestrator
(blackboard pattern).

### 5.4 Blackboard: workspace & journal

```
runs/<run_id>/
├── run.json            # metadata, state, budgets, phase
├── journal.jsonl       # append-only events (audit trail + agent memory)
├── workspace/
│   ├── data/           # raw competition downloads
│   ├── eda/            # EDA notebooks/scripts + figures
│   ├── features/       # generated feature pipelines (code)
│   ├── models/         # training code, params, pickles
│   ├── submissions/    # submission CSVs + Kaggle scores
│   └── reports/        # schema-validated agent outputs (strategy, data report, reviews)
├── experiments.db      # candidate registry: params, CV, LB, status
└── logs/               # agent-level + execution logs
```

Journal event sketch: `{"ts": ..., "run": ..., "agent": ..., "type": "...", "data": {...}}`
— types like `phase.enter`, `agent.output`, `candidate.scored`, `submission.accepted`,
`user.guidance`, `budget.warning`.

Per D0005: the journal is the **source of truth** and the agents' communication
channel; `experiments.db` is a materialized view over it, rebuildable by journal replay.

## 6. User Interaction Model

### 6.1 CLI surface (typer)

```
slayer init                          # verify .env, ping OpenRouter + Kaggle auth
slayer competitions [list] [--filter]# browse active competitions
slayer run <competition>             # start the orchestration loop (main command)
    --target top-20                  # goal percentile (default 20)
    --autonomy approve-first|agentic # approval policy (default approve-first)
    --budget-usd 10                  # LLM spend cap for this run
    --submissions-per-day 4          # stay under Kaggle quota
    --dry-run                        # route submissions to MockSink (no real API calls)
    --fresh / --resume <run_id>
slayer status [run_id]               # phase, budget, best CV/LB, next action
slayer journal [run_id] [--follow]   # timeline view; -f tails like logs
slayer leaderboard <competition>    # current standings + our position
slayer pause <run_id>                # graceful pause (checkpoint & stop)
slayer resume <run_id>               # continue from checkpoint
slayer chat [run_id]                 # (v2) REPL: talk to the orchestrator mid-run
```

### 6.2 Live experience (v1)

The run command runs **in the foreground** and streams the team's activity:
agent-by-agent action lines, phase transitions, CV scores, budget ticker.
`Ctrl+C` = graceful pause (state checkpointed, resumable). No daemon in v1 — this
keeps IPC simple; daemonization + `attach` is Phase 5.

### 6.3 Approvals & autonomy

Default is **human-in-the-loop for irreversible/irreplaceable actions**:

| Action | `approve-first` (default) | `agentic` |
|--------|--------------------------|---------|
| Download data, EDA, local CV | automatic | automatic |
| Spend > 50% of run budget | warn + approve | warn only |
| **Kaggle submission** | **approve each one** (first 2 always; then only if CV improved) | auto if CV improved & reviewer approved |
| Select/enter a competition | user already chose via `slayer run` | same |

### 6.4 Chat with the orchestrator (v2 sketch)

`slayer chat` opens a REPL where the user talks *only* to the orchestrator, which can
inject guidance into the team ("stop tuning LightGBM, try feature X") without stopping
the loop. Behind the scenes: guidance events land in the journal; the orchestrator
incorporates them at the next decision point. (Needs the daemon + control channel from
Phase 5; the *interaction contract* is already defined by the journal schema.)

## 7. Data Model & Schemas

All agent outputs are schema-validated pydantic models (the `schemas/` folder becomes the
single source of truth, re-exported as a package module):

- `StrategyPlan` — problem type, metric, CV protocol, roadmap[], submission_policy
- `DataReport` — schema profiles, missingness, target analysis, risks
- `FeaturePlan` — features[] with rationale + expected value
- `ModelCandidate` — feature_code_ref, model_type, params, cv_protocol
- `CVResult` — per-fold scores, aggregate, std, timing
- `ReviewVerdict` — decision: approve|reject|revise, reasoning, risk_flags[]
- `SubmissionRecord` — file, cv_score, lb_score, timestamp, quota state
- `RunState` — phase, history of transitions, budget state, targets

## 8. Integrations

### 8.1 LLM — OpenRouter

- Thin client: chat completions + **structured outputs** (JSON schema / tool mode), retries with backoff, token/USD accounting per agent per run.
- Model routing per agent from `config/agents.yaml` (e.g., strong reasoning model for planner/researcher, cheaper model for reviewer/analyst).
- Every call logged to the journal with cost.

### 8.2 Kaggle API

Wrapped in `KaggleClient` (kaggle package + direct REST where needed):

- `competitions.list` — browse/filter
- `competition.download_files` — data collection
- `competitions.submissions` — submit + poll status
- `competitions.leaderboard_view` — monitoring (respecting Kaggle's rate expectations; poll on a slow cadence, e.g., hourly)
- Quota manager: count submissions/day, refuse to exceed configured max.

**Submission sinks.** Submission goes through a `SubmissionSink` interface with two
implementations: `MockSink` (records the submission in the journal + experiments DB with
a simulated LB score; default for the MVP slice and integration tests) and
`KaggleSink` (real API). Selected via `--dry-run` flag / config. This lets the entire
pipeline — including approvals, quota logic, and monitoring — be exercised end-to-end
with zero quota risk before going live.

### 8.3 Local execution sandbox

- v1: generated Python runs in a subprocess of the project venv with a hard timeout, captured stdout/stderr, and a working directory scoped to the run's workspace.
- v2+: containerized execution (docker) for stronger isolation of generated code.

## 9. Safety, Budgets, ToS

- **Budget:** per-run USD cap enforced in the LLM client; near-limit warnings in journal + CLI.
- **Submissions:** never exceed daily quota; dedupe identical submissions; only submit on CV improvement (or explicit user say-so).
- **Secrets:** `.env` only (already gitignored); never printed/logged.
- **Kaggle rules:** use only the official kaggle API; no scraping; respect per-competition automation rules noted in the plan/ToS review during PLANNING phase.

## 10. Proposed Repository Layout

```
src/slayer/
├── __init__.py            # exports main()
├── cli/                   # typer app, commands, live display
├── config.py              # pydantic-settings: .env + config/*.yaml
├── llm/                   # OpenRouter client, structured outputs, cost tracker
├── kaggle/                # KaggleClient, quota manager
├── agents/                # Agent base + orchestrator/planner/analyst/researcher/reviewer
├── loop/                  # phase gates, state machine, budgets
├── workspace/             # run directory management, journal
└── schemas/               # pydantic models (re-exported from schemas/)
prompts/                   # agent system prompts (markdown)
config/                    # slayer.yaml (global), agents.yaml (model routing)
schemas/                   # JSON schema exports (docs/API contracts)
docs/plan/                 # this plan + design docs
tests/{unit,integration}/
runs/                      # run artifacts (gitignored)
```

## 11. Implementation Phases

| Phase | Scope | Key deliverables | Acceptance criteria | Status |
|-------|-------|------------------|---------------------|--------|
| **0. Foundation** | infra only | config loading; OpenRouter client w/ JSON mode + cost tracking; KaggleClient wrapper; journal + run store; `slayer init` | `slayer init` pings both APIs green; journal round-trip unit-tested | ✅ |
| **1. Data collection & EDA** | INIT → DATA_COLLECTION → ANALYSIS | loop engine + `slayer run <comp>`; download + safe unzip + manifest; profiler + data analyst → `DataReport` | Real comp: data on disk, human-readable EDA report in `reports/` | ✅ |
| **2. Baseline & first submission (mocked)** | PLANNING → BASELINE → REVIEW → SUBMIT | CV harness; planner `StrategyPlan`; researcher pipeline codegen; reviewer gate; `SubmissionSink` abstraction (**MockSink** for MVP); submission approval flow | End-to-end pipeline completes with submission recorded via MockSink; approval + quota logic exercised; zero real quota spent | ✅ |
| **2b. First live submission** | go live *(re-sequenced after Phase 3, C0003)* | fresh `--live` run on the closed playground (user-confirmed submission under approve-first) | Real submission accepted by Kaggle; score recorded in journal | ✅ (submission `57503219`, LB 0.94149) |
| **3. Iteration engine & resilience** | the heart | experiment loop (propose → run → review → submit per roadmap step, with repair-retry); budget enforcement; `slayer resume`; harness timeout handling (Q0002) | ≥5 sequential iterations with measurable CV improvement; interrupted run resumes without redoing completed phases | ✅ |
| **4. Monitoring & leaderboard** | MONITOR + ops | MONITOR phase (LB-score collection + percentile-vs-target awareness); `leaderboard` command; deadline awareness; iterate re-entry decision | Real LB scores recorded in the registry; position vs target reported from a live run | ✅ (4.2 live validation pending) |
| **5. Autonomy & polish** | UX depth | daemon + `attach`; `slayer chat` REPL; richer live view; agentic orchestrator upgrade (dynamic replanning); cost report | User can steer a live run via chat; top-20% achieved on target comp | ⬜ |

Phases 0–2 form the **MVP slice**: every architectural piece exists but shallow (all
submissions mocked). Phase 2b is the go-live moment. Phases 3–5 deepen the loop.

**Phase 0 step breakdown** — each step is one commit + user review:

| Step | Scope | Status |
|------|-------|--------|
| 0.1 | Scaffolding: repo layout, config system (pydantic-settings + YAML), `slayer init` (config + secrets checks), docs structure + ADRs D0001–D0005 | ✅ |
| 0.2 | OpenRouter LLM client: chat + structured outputs + cost tracking + `slayer init` LLM ping | ✅ ([design](designs/01-llm-client.md)) |
| 0.3 | Journal (JSONL) + run store + candidate registry (SQLite, per D0005) | ✅ ([design](designs/02-journal-and-registry.md)) |
| 0.4 | KaggleClient wrapper + quota manager + `slayer init` Kaggle ping | ✅ ([design](designs/03-kaggle-client.md)) |
| 0.5 | Phase 0 acceptance: `slayer init` green end-to-end; unit tests green | ✅ |

**Phase 1 step breakdown** (order per [C0002](changes/C0002-loop-phase-reorder.md)):

| Step | Scope | Status |
|------|-------|--------|
| 1.1 | Loop engine (phase machine, pause/fail semantics) + `slayer run` command + INIT (competition context, rules probe) + DATA_COLLECTION (download, safe unzip, sha256 manifest) | ✅ |
| 1.2 | ANALYSIS: deterministic profiler + data analyst agent → `DataReport` + `FeaturePlan` | ✅ ([design](designs/05-analysis-and-data-analyst.md)) |

**Phase 2 step breakdown:**

| Step | Scope | Status |
|------|-------|--------|
| 2.1 | CV harness (fixed skeleton, generated plugin) + sandbox runner + metric registry + `CVResult` | ✅ ([design](designs/06-cv-harness.md)) |
| 2.1b | Ops polish pulled forward from the Phase 1 review: per-competition data cache (skip re-downloads), `slayer status`/`slayer journal` commands, docstring backfill across src | ✅ |
| 2.2 | PLANNING: planner agent → `StrategyPlan` (problem type, CV protocol, roadmap, submission policy) | ✅ ([design](designs/07-planner.md)) |
| 2.3 | BASELINE: researcher agent pipeline codegen + registry wiring → first scored candidate | ✅ ([design](designs/08-researcher-and-baseline.md)) |
| 2.4 | REVIEW + SUBMIT (mock): reviewer gate, `SubmissionSink` (MockSink), approve-first flow | ✅ ([design](designs/09-review-and-submission.md)) |

**Phase 3 step breakdown** (2b re-sequenced after Phase 3 per [C0003](changes/C0003-reorder-2b-and-scope.md)):

| Step | Scope | Status |
|------|-------|--------|
| 3.1 | Iteration engine core: per-roadmap-step propose → evaluate → review → submit loop, repair-retry on failed candidates with error feedback, LLM budget enforcement | ✅ ([design](designs/10-iteration-engine.md); live loop validation pending user go-ahead) |
| 3.1b | Gap corrections from live validation: post-iterate revise-retry (one bounded repair on an unactioned REVISE verdict) + LLM cost visibility in `slayer status` | ✅ |
| 3.2 | `slayer resume <run>`: resume paused/failed runs without redoing completed phases (registry-driven skip logic) | ✅ ([design](designs/11-resume.md)) |
| 3.3 | Harness timeout handling per [Q0002](questions/Q0002-harness-timeout-handling.md): configurable budget, budget-aware researcher prompt, probe fold + incremental fold reporting + harness self-deadline (Q0003.1) | ✅ |

**Phase 4 step breakdown:**

| Step | Scope | Status |
|------|-------|--------|
| 4.1 | MONITOR phase: real LB-score collection for submitted candidates (registry `record_submission`), percentile-vs-target position reporting; `slayer leaderboard` command | ✅ ([design](designs/12-monitor.md)) |
| 4.1b | Robustness fixes from the live resume: bounded step retries (max 3 attempts/step across resumes), CatBoost/XGBoost prompt contracts, cost estimation fallback (budget enforcement was inert), iteration terminal narration | ✅ |
| 4.2 | Deadline awareness + bounded iterate re-entry on target shortfall (D0007): score-gap signal + CV-basis position proxy in MONITOR, deterministic re-entry gate, REPLAN (planner roadmap extension from full history), journal-based cumulative LLM budget, `--max-rounds` | ✅ ([design](designs/13-reentry-rounds.md); live validation in progress) |
| 4.3 | OOF persistence in the harness + deterministic hill-climb blender (code-owned blend step at the end of ITERATE); researcher model A/B (`ml_researcher` → claude-sonnet-5.5) | ✅ ([design](designs/14-oof-blender.md); live validation pending) |

## 12. Testing Strategy

- **Unit** — agents with a fake LLM (recorded/ canned responses), Kaggle client mocked, phase machine transitions, budget/quota logic.
- **Integration** — real OpenRouter (tiny models, env-gated); real Kaggle metadata calls (read-only).
- **Smoke / live** — full loop against a playground competition (e.g., `playground-series-s5e*` style) with `--budget-usd 1`, run manually.
- Generated code (features/training) is validated by the CV harness itself — a pipeline that doesn't run can't be a candidate.

## 13. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM cost runaway | per-run USD cap enforced client-side; cheap-model routing; budget events in journal |
| Hallucinated/leaky pipelines | reviewer gate before submission; standard CV harness (agents can't fake scores); leakage checklist in reviewer prompt |
| Wasting submission quota | quota manager + only-if-CV-improved rule + dedupe |
| Long-running state corruption | checkpoint-on-phase-transition; journal replay to rebuild state; resume validation |
| Kaggle rule/ToS friction | official API only; PLANNING phase reads comp rules & flags automation constraints |
| Model write-code quality | baseline template library (known-good skeletons per problem type) the researcher adapts instead of writing from scratch |

## 14. Decisions

Decisions are recorded as ADRs in [`decisions/`](decisions/); open questions live in
[`questions/`](questions/) — formats and workflow in [`README.md`](README.md).

| # | Area | Decision | ADR |
|---|------|----------|-----|
| D1 | Interaction model v1 | Foreground streaming + inline approvals; daemon + chat in Phase 5 | [D0001](decisions/D0001-interaction-model-v1.md) |
| D2 | Submission autonomy | `approve-first` default; all MVP submissions mocked via `SubmissionSink`; live at Phase 2b | [D0002](decisions/D0002-submission-autonomy-mock-sink.md) |
| D3 | Orchestrator pattern | Deterministic pipeline with LLM decision points; agentic upgrade in Phase 5 | [D0003](decisions/D0003-orchestrator-deterministic-pipeline.md) |
| D4 | MVP target | Playground Series tabular competition (Phases 0–2b); real comp from Phase 3 | [D0004](decisions/D0004-mvp-target-playground-competition.md) |
| D5 | Storage | Event-sourced: JSONL journal (events + agent blackboard + source of truth) + SQLite candidate registry (entity state, rebuildable from journal) | [D0005](decisions/D0005-storage-journal-and-sqlite-registry.md) |
| D6 | Model routing | Fast Gemini Flash tier default (D0006); free reasoning model as per-agent fallback | [D0006](decisions/D0006-model-routing-gemini-flash.md) |
| D7 | Shortfall handling | Automatic bounded re-entry rounds (gate → REPLAN → ITERATE…MONITOR), ≤ `max_rounds` | [D0007](decisions/D0007-bounded-reentry-rounds.md) |

**Open questions:** [Q0003](questions/Q0003-interruption-safety.md) (graceful SIGINT
still open), [Q0004](questions/Q0004-competition-discovery-ux.md) (discovery UX).

## 15. Documentation Conventions (`docs/plan/`)

Full structure and workflow: [`docs/plan/README.md`](README.md).

- `00-implementation-plan.md` — this living master plan.
- `questions/Q####-*.md` — open questions (context, options, recommendation).
- `decisions/D####-*.md` — ADRs for settled decisions.
- `changes/C####-*.md` — direction changes / pivots.
- `designs/NN-*.md` — feature design docs, written before implementation, retro after.

Every implementation step: implement → present for review → **commit only after user
approval** (code + its docs together). See [C0001](changes/C0001-no-commit-before-review.md).
