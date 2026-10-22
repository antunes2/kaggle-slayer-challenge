# The Run Loop, In Detail

> Phase-by-phase reference: what each phase reads, writes, and skips. For the big
> picture see `docs/architecture.md`; for design history, `docs/plan/designs/`.

## Phase machine

`PHASE_ORDER` (in `loop/engine.py`) — deterministic; the engine never improvises:

```
init → data_collection → analysis → planning → baseline → iterate → review → submit → monitor
```

- Entering a phase journals `phase.enter` and streams `[phase] <name>`.
- A phase that raises → run status `failed`, exception re-raised to the CLI.
- All phases complete → `finished`.
- `resume` re-enters the same machine; phases skip their own completed work (below).

## Phase by phase

### INIT
Resolves the competition (paginated ref search, search fallback for closed comps),
probes data access (`list_files` — fails fast with the rules URL if unaccepted),
writes `reports/competition.json` (metric, deadline, quota). Never skipped (cheap).

### DATA_COLLECTION
Downloads the competition zip **or reuses the per-competition cache**
(`runs/_cache/<slug>/`, validated by remote file list + zip sha256). Safe-unzips
(zip-slip guarded), writes `data/manifest.json` (sizes + sha256 per file).

### ANALYSIS — skip if `data_report.json` + `feature_plan.json` exist
1. **Profiler** (deterministic, `analysis/profiler.py`): target detection from
   sample_submission, problem-type guess, per-column stats/missingness/cardinality,
   class balance or distribution, numeric correlations (string-binary targets encoded).
   Writes `eda/profile.{json,md}`.
2. **Data analyst agent**: interprets the profile → `DataReport` + `FeaturePlan`
   (schema-validated). The analyst hunts dupes, constants, skew, cardinality,
   missingness-as-signal, and states what the profile *cannot* show.

### PLANNING — skip if `strategy_plan.json` exists
Planner agent → `StrategyPlan`: confirmed problem type, CV shape (`k_folds`, seed,
stratified), ordered **roadmap** (step 1 = simplest baseline; later steps: feature
groups → alternative family → tuning → blending), `min_cv_improvement` submission
gate, and `guidance[]` — hard constraints (imbalance handling, leakage rules, metric
discipline) that flow verbatim into every researcher prompt. **The metric and its
direction are resolved by code** (`cv.build_cv_protocol` via the metric registry) —
the planner never decides physics. Unsupported metric → loud failure here.

### BASELINE — skip if roadmap step 1 has a non-failed candidate
Researcher writes `pipeline.py` for roadmap step 1 → syntax-checked (fail fast) →
harness. Failure raises (first candidate is fatal to a fresh run). More than 3
attempts total → `exhausted 3 attempts`.

### ITERATE — the improvement loop (inside one phase)
For each pending roadmap step (priority order, **max 3 candidate attempts per step
across all resumes**, counted from the registry):
1. Researcher proposes (with **iteration context**: registry summary, last reviewer
   verdict + reasoning, last failure's log tail, CV time budget).
2. Evaluate through the shared `_evaluate_proposal` (register → write code →
   compile-check → harness → `record_cv` or mark `failed`).
3. On failure: **one repair attempt** with the error fed back; second failure
   exhausts the step.
4. After each scored candidate: `review_phase` + `submit_phase` run.
5. After the loop: one bounded repair for an unactioned REVISE verdict, if any.

Budget: before every proposal, cumulative LLM cost ≥ `run.budget_usd` →
`budget.warning` + stop iterating.

### REVIEW
Reviewer agent judges the best scored-but-unreviewed candidate (metric-direction
aware): its pipeline code, CV results, and the strategy guidance. Verdicts:
approve → `approved`; reject → `rejected`; revise → stays `scored`, full reasoning
journaled (this is the fuel the next proposal's iteration context consumes).
The reviewer hunts leakage, fit-on-test, metric mishandling, hardcoded
dataset statistics, intent-vs-implementation gaps, and improvements within fold noise.

### SUBMIT — three gates, in order
1. **CV gate**: `cv(candidate) - cv(best_submitted)` (direction-aware) ≥
   `min_cv_improvement`; first submission always allowed.
2. **Quota gate**: real submissions only — `QuotaManager` counts journal
   `submission.accepted` events per UTC day vs the configured cap; mock submissions
   never consume quota.
3. **Approval gate**: `approve-first` (default) prompts y/n in the terminal;
   `agentic` skips.
Then the sink: **MockSink** (default) records a simulated LB (cv + deterministic
noise) and `submission.mocked`; **KaggleSink** submits for real, records the
kaggle submission id (`record_submission`) and `submission.accepted`.

### MONITOR
Reads the world back:
1. Poll `kaggle.submissions(competition)`; for each registry candidate that is
   `submitted` with no LB score yet, match by **submission id** *or* the
   `slayer cand-NNNN:` marker in the submission description (pre-record_submission
   runs), then `record_lb`.
2. Fetch the leaderboard (paginated; SDK print noise suppressed), find our entry
   (username, then substring fallback), compute `percentile = rank / total * 100`,
   journal `monitor.position`, and report met / NOT met vs `target_percentile`.
   Not found → honest message (display name vs username, or beyond fetched pages).

## Resume semantics

Every expensive phase has a skip predicate; pending roadmap steps derive from the
registry (`params.roadmap_step` + attempt cap). Resuming a finished run re-enters
MONITOR (fresh leaderboard data). A failed baseline re-proposes — resume is the
crash-repair path — until the 3-attempt cap.

## Candidate lifecycle

```
proposed → scored → approved → submitted → (lb recorded by MONITOR)
   │          │
   │          └── rejected / stays scored with REVISE feedback (one bounded repair)
   └── failed (harness error/timeout/syntax — counts toward the step's 3 attempts)
```

## Journal event catalog

`run.created`, `run.status_changed`, `phase.enter`, `loop.paused`,
`competition.context`, `data.collected`, `agent.call` (tokens + cost, with
`cost_estimated` flag), `agent.output`, `candidate.added`,
`candidate.status_changed` (carries `submission_id` for real submissions),
`candidate.scored`, `candidate.lb_scored`, `submission.accepted` (real — quota),
`submission.mocked`, `budget.warning`, `iterate.skipped`, `monitor.position`.
