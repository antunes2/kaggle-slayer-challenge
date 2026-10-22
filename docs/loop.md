# The Run Loop, In Detail

> Phase-by-phase reference: what each phase reads, writes, and skips. For the big
> picture see `docs/architecture.md`; for design history, `docs/plan/designs/`.

## Phase machine

`PHASE_ORDER` (in `loop/engine.py`) — deterministic; the engine never improvises:

```
init → data_collection → analysis → planning → baseline → iterate → review → submit → monitor
```

then, while the re-entry gate allows (`ROUND_ORDER`, D0007):

```
replan → iterate → review → submit → monitor      (round N; phase.enter carries "round": N)
```

- Entering a phase journals `phase.enter` and streams `[phase] <name>`.
- A phase that raises → run status `failed`, exception re-raised to the CLI.
- REPLAN raising `CampaignStop` (planner stopped) ends the campaign → `finished`.
- All phases complete and the gate says no → `finished`.
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
First the **notebook scout** (design 15, no LLM): the competition's most-voted public
notebooks are read via the official API and reduced to a capped digest
(`reports/notebook_digest.md`, ~4k tokens, `[EXTERNAL]` lines tagged, refreshed when
older than 12 h, `scout.digest` journaled; fail-soft). Then the planner agent →
`StrategyPlan`: confirmed problem type, CV shape (`k_folds`, seed, stratified), ordered
**roadmap** (step 1 = simplest baseline; then hypotheses ordered by evidence —
representation changes first, see C0004), `min_cv_improvement` submission gate (sized
to *paired* fold noise), and `guidance[]` — hard constraints (imbalance handling, leakage rules, metric
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
5. After the loop: one bounded repair for the **best-scoring** candidate with an
   unactioned REVISE verdict (unactioned = no candidate has `revision_of` = it); the
   fix is registered with `parent_id`/`revision_of` lineage (skipped when the budget
   stopped the loop).
6. **Blend step** (design 14, no LLM — runs even after a budget stop): the
   reviewer-approved candidates with OOF predictions are hill-climbed (Caruana greedy
   selection with replacement, mean per-fold metric on the shared folds). A blend of ≥ 2
   members that beats the best single is registered as an `approved` `Blend` candidate
   (weighted-average submission), then SUBMIT's gates run. Each distinct pool is
   evaluated once (`blend.evaluated`).

Budget: before every proposal, cumulative LLM cost ≥ `run.budget_usd` →
`budget.warning` + stop iterating. Cumulative = journal sum of `agent.call.cost_usd`
(max with the live client), so spend from earlier resumes counts.

### REVIEW
Reviewer agent judges the best scored candidate **without a verdict yet**
(metric-direction aware; a REVISE'd candidate is never re-reviewed — it is fixed by the
revision step instead): its pipeline code, CV results, and the strategy guidance. Verdicts:
approve → `approved`; reject → `rejected`; revise → stays `scored`, full reasoning
journaled (this is the fuel the next proposal's iteration context consumes).
The reviewer hunts leakage, fit-on-test, metric mishandling, hardcoded
dataset statistics, and intent-vs-implementation gaps. Small gains are flagged, not
revised — the submission gate owns the minimum-improvement decision.

### SUBMIT — three gates, in order
1. **CV gate**: passes when `cv(candidate) - cv(best_submitted)` (direction-aware) ≥
   `min_cv_improvement`, **or** when the paired per-fold gain is consistent (positive
   on all folds but at most one, mean > 2× the std of the fold differences) — all
   candidates share folds, so paired noise is far below fold std. First submission
   always allowed.
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
2. Download the **full public leaderboard CSV** (authoritative: every team, with
   member usernames) — the paginated view API can't reach deep fields and the SDK's
   page-token printing is suppressed. Find our entry by username in the member list.
   Not listed → compute an **estimated position**: `rank = entries strictly better + 1`,
   `percentile = rank / (total + 1)`, tie band reported. The estimate's `basis` is
   `lb` (best real LB score — e.g. **late submissions to a closed comp are scored but
   never join the LB**) or `cv` (no real LB score yet — dry runs, pending scoring).
3. **Gap signal**: `target_score` = score at rank `max(1, floor(total × target%))`;
   `gap` = direction-aware shortfall of our best score (positive = short). Journal
   `monitor.position` (`basis`, rank, percentile, met, best/target score, gap); report
   met / NOT met. The CSV snapshot persists as `workspace/reports/leaderboard.csv`.

### Re-entry gate (after every MONITOR)
`decide_reentry` journals `reentry.decision` and stops on the first failing check:
planner previously stopped → round cap (`rounds_done >= max_rounds`) → no position
signal → target met → LLM budget spent → deadline in the future but closer than one
harness budget (a passed deadline does not block). Otherwise: re-enter round N.

### REPLAN (round phases only)
The planner (re-entry prompt `prompts/planner_extend.md`) reads the current strategy,
the full experiment history (every candidate; failures with their first error line),
the position + gap with the gap/spread ratio, the round constraints (budget left,
time budget, observed fold time, deadline), and the notebook digest (refreshed if stale). Output `RoadmapExtension`: an assessment,
1-4 new steps + optional guidance — merged by code (unique titles, priorities after
the current max, guidance dedupe) into `strategy_plan.json`; the raw extension lands in
`reports/roadmap_round_N.json`; `roadmap.extended` is journaled. Or `stop` with a
reason → `CampaignStop`, final for the run. ITERATE then picks the new steps up as
ordinary pending steps (same caps, repair, review, submit gates).

## Resume semantics

Every expensive phase has a skip predicate; pending roadmap steps derive from the
registry (`params.roadmap_step` + attempt cap). Resuming a finished run re-enters
MONITOR (fresh leaderboard data) and then the re-entry gate — a finished-short run
continues its campaign until a stop condition holds (`slayer resume --max-rounds N`
raises the cap). A crash mid-round resumes naturally: the round's steps are already
merged into `strategy_plan.json`, so the linear pass's ITERATE finishes them. A failed
baseline re-proposes — resume is the crash-repair path — until the 3-attempt cap.

## Candidate lifecycle

```
proposed → scored → approved → submitted → (lb recorded by MONITOR)
   │          │         ▲
   │          │         └── Blend candidates enter here directly (all members approved)
   │          └── rejected / stays scored with REVISE feedback (one bounded repair)
   └── failed (harness error/timeout/syntax — counts toward the step's 3 attempts)
```

## Journal event catalog

`run.created`, `run.status_changed`, `run.settings_changed`, `phase.enter`,
`loop.paused`, `competition.context`, `data.collected`, `agent.call` (tokens + cost,
with `cost_estimated` flag), `agent.output`, `candidate.added`,
`candidate.status_changed` (carries `submission_id` for real submissions),
`candidate.scored`, `candidate.lb_scored`, `submission.accepted` (real — quota),
`submission.mocked`, `budget.warning`, `iterate.skipped`, `monitor.position`,
`reentry.decision`, `roadmap.extended`, `blend.evaluated`, `scout.digest`.
