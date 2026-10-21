# 07 — PLANNING: Planner Agent → StrategyPlan

- **Phase:** 2 (step 2.2) · **Status:** implemented · **Date:** 2026-10-20
- **Feeds:** design 06 (CV harness consumes the plan's protocol); BASELINE (step 2.3) executes roadmap step 1

## Goal

Turn the analysis artifacts into the per-competition strategy the researcher executes:
confirmed problem type, CV protocol, an ordered modelling roadmap, a submission gate,
and hard guidance for the researcher.

## Key design decision: the LLM plans, code decides physics

The planner **never** decides the evaluation metric or its direction — those are fixed
by the competition. `build_cv_protocol()` (in `slayer/cv`) merges the planner's chosen
CV shape with the competition metric resolved through the metric registry:

```python
StrategyPlan(
    problem_type, target_column,
    cv: CVSettings(k_folds, seed, stratified),   # planner-chosen
    roadmap: [RoadmapStep(title, description, model_family, feature_ideas, priority)],
    min_cv_improvement: float,                    # submission gate delta
    guidance: [str],                              # constraints/warnings for the researcher
    summary,
)
build_cv_protocol(competition_metric, plan.cv) -> CVProtocol   # deterministic merge
```

If the competition's metric isn't in the registry, PLANNING fails loudly (MetricError)
before any modelling effort is spent — the supported list is in the planner prompt.

## Why these fields

- `cv.k_folds` is planner-chosen because fold count is a *cost/quality tradeoff* that
  depends on dataset size (the planner sees train rows via the analyst report).
- `min_cv_improvement` operationalizes the plan's submission policy (D0002:
  only-if-CV-improved) — a small delta (e.g. 0.0005 for AUC) prevents wasting quota on
  noise.
- `roadmap[0]` is by convention the simplest possible baseline — the BASELINE phase
  (step 2.3) executes it first, later steps drive ITERATE.
- `guidance` carries the nature-of-the-challenge instructions (imbalance handling,
  leakage warnings, metric-specific notes like "optimize ranking, not accuracy") into
  every researcher prompt.

## Interface

```python
PlannerAgent(llm, config, journal, run_id)
  .plan(competition_json, data_report_json, feature_plan_json) -> StrategyPlan

planning_phase(ctx)  # writes reports/strategy_plan.{json,md}, journals, reports
```

## Tests

- Agent: prompt contains all three artifacts; `agent.call` journaled; canned output flows.
- Schema: empty roadmap rejected; invalid problem_type rejected.
- Phase: happy path writes plan + markdown; unsupported metric → MetricError; missing
  LLM → AnalysisError; protocol direction correct for AUC vs RMSE.
- CLI: fake LLM dispatches by response_model; `strategy_plan.json` on disk end-to-end.

## Retro

Live-validated on the real run (free reasoning model): the planner produced a 6-step
roadmap (LightGBM anchor → feature groups → CatBoost benchmark → tuning → blend) with
rigorous guidance — ranking-metric discipline for ROC AUC, fold-derived class weights
for the 17.5% imbalance, fold-safe encoders, SMOTE-inside-folds warning, train/test
shift audits, and submission restraint given the 10/day quota. This is the strongest
evidence yet that "LLM judgment on top of deterministic scaffolding" works on the free
tier.

Operational note (process, not design): the free reasoning model takes 1.5–3 minutes
per structured call — foreground runs need generous timeouts (or background execution
+ polling). The validation backlog also surfaced a UX truth: without resume, any
interruption costs a full re-run. `slayer resume` (Phase 4) is now clearly the highest
value ops item.
