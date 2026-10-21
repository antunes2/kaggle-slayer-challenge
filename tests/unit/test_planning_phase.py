import json

import pytest

from slayer.analysis import AnalysisError
from slayer.config import DefaultsConfig, load_agents_config
from slayer.cv import MetricError, build_cv_protocol
from slayer.llm import LLMResponse
from slayer.loop import RunContext
from slayer.loop.phases import planning_phase
from slayer.schemas import CVSettings, RoadmapStep, StrategyPlan
from slayer.workspace import EventType, RunStore

CANNED = StrategyPlan(
    problem_type="binary_classification",
    target_column="Will_Buy_EV",
    cv=CVSettings(k_folds=3, seed=7, stratified=True),
    roadmap=[
        RoadmapStep(
            title="Baseline",
            description="LightGBM raw",
            model_family="lightgbm",
            priority=1,
        )
    ],
    min_cv_improvement=0.001,
    guidance=["stratified folds"],
    summary="Go fast.",
)


class FakeLLM:
    """Returns the canned plan for any structured request."""

    def complete_structured(self, messages, model, response_model, **kwargs):
        """Return the canned strategy plan."""
        return LLMResponse(content="{}", model=model), CANNED


def _make_ctx(tmp_path, llm, metric: str = "Roc Auc Score") -> RunContext:
    """Create a run with analysis artifacts on disk."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    reports = run.workspace("reports")
    (reports / "competition.json").write_text(
        json.dumps({"ref": "comp-x", "evaluation_metric": metric}), encoding="utf-8"
    )
    (reports / "data_report.json").write_text(
        json.dumps({"problem_type": "binary_classification", "target_column": "T"}),
        encoding="utf-8",
    )
    (reports / "feature_plan.json").write_text(json.dumps({"ideas": []}), encoding="utf-8")
    return RunContext(
        run=run,
        journal=store.journal(run),
        registry=store.registry(run),
        kaggle=None,
        store=store,
        llm=llm,
        agent_configs=load_agents_config(),
    )


def test_planning_phase_writes_strategy(tmp_path) -> None:
    """The phase writes strategy_plan.{json,md} and journals the planner output."""
    ctx = _make_ctx(tmp_path, FakeLLM())
    messages: list[str] = []
    ctx.reporter = messages.append

    planning_phase(ctx)

    reports = ctx.run.workspace("reports")
    plan = json.loads((reports / "strategy_plan.json").read_text(encoding="utf-8"))
    assert plan["cv"]["k_folds"] == 3
    assert plan["cv"]["stratified"] is True
    assert (reports / "strategy_plan.md").is_file()

    outputs = ctx.journal.read(type=EventType.AGENT_OUTPUT)
    assert outputs[0].agent == "planner"
    assert outputs[0].data["roadmap_steps"] == 1
    assert any("strategy:" in m for m in messages)


def test_planning_phase_fails_on_unsupported_metric(tmp_path) -> None:
    """A competition metric outside the registry fails planning loudly."""
    ctx = _make_ctx(tmp_path, FakeLLM(), metric="Some Exotic Metric")
    with pytest.raises(MetricError, match="unsupported metric"):
        planning_phase(ctx)


def test_planning_phase_requires_llm(tmp_path) -> None:
    """Without an LLM client the phase cannot run."""
    ctx = _make_ctx(tmp_path, llm=None)
    with pytest.raises(AnalysisError, match="LLM client"):
        planning_phase(ctx)


def test_build_cv_protocol_resolves_direction() -> None:
    """Direction comes from the metric registry, not the planner."""
    auc = build_cv_protocol("Roc Auc Score", CVSettings(k_folds=3))
    assert auc.higher_is_better is True
    assert auc.k_folds == 3
    assert auc.metric == "Roc Auc Score"

    rmse = build_cv_protocol("Root Mean Squared Error", CVSettings())
    assert rmse.higher_is_better is False
