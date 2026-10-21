from pathlib import Path

import pytest
from pydantic import ValidationError

from slayer.agents import AgentError, PlannerAgent
from slayer.config import AgentModelConfig
from slayer.llm import LLMResponse
from slayer.schemas import CVSettings, RoadmapStep, StrategyPlan
from slayer.workspace import EventType, Journal

CANNED = StrategyPlan(
    problem_type="binary_classification",
    target_column="Target",
    cv=CVSettings(k_folds=3, seed=42, stratified=True),
    roadmap=[
        RoadmapStep(
            title="Baseline LightGBM",
            description="LightGBM on raw features",
            model_family="lightgbm",
            feature_ideas=[],
            priority=1,
        ),
        RoadmapStep(
            title="Charging bundle features",
            description="Add the charging-access feature bundle",
            model_family="lightgbm",
            feature_ideas=["Charging-access bundle"],
            priority=2,
        ),
    ],
    min_cv_improvement=0.0005,
    guidance=["Use stratified folds", "Do not use the id column"],
    summary="LightGBM-first plan for an imbalanced binary problem.",
)


class FakeLLM:
    """Records calls and returns the canned plan."""

    def __init__(self) -> None:
        self.calls: list[tuple[list, str, type]] = []

    def complete_structured(self, messages, model, response_model, **kwargs):
        """Return the canned plan and record the request."""
        self.calls.append((messages, model, response_model))
        return LLMResponse(content="{}", model=model), CANNED


def test_plan_returns_strategy_and_journals(tmp_path: Path) -> None:
    """The planner produces a validated plan and journals its call."""
    llm = FakeLLM()
    journal = Journal(tmp_path / "journal.jsonl")
    agent = PlannerAgent(
        llm=llm,
        config=AgentModelConfig(model="vendor/model"),
        journal=journal,
        run_id="run-1",
    )
    plan = agent.plan(
        competition_json='{"ref": "c", "evaluation_metric": "Roc Auc Score"}',
        data_report_json='{"problem_type": "binary_classification"}',
        feature_plan_json='{"ideas": []}',
    )

    assert plan.problem_type == "binary_classification"
    assert plan.roadmap[0].title == "Baseline LightGBM"

    messages = llm.calls[0][0]
    assert messages[0]["role"] == "system"
    assert "strategist" in messages[0]["content"].lower()
    user_content = messages[1]["content"]
    assert "Roc Auc Score" in user_content
    assert "binary_classification" in user_content
    assert "matching exactly this schema" in user_content

    calls = journal.read(type=EventType.AGENT_CALL)
    assert len(calls) == 1
    assert calls[0].agent == "planner"


def test_strategy_plan_schema_constraints() -> None:
    """Roadmap must be non-empty and problem_type must be a known literal."""
    with pytest.raises(ValidationError):
        StrategyPlan(
            problem_type="weird",
            target_column="T",
            cv=CVSettings(),
            roadmap=[],
            summary="s",
        )


def test_missing_prompt_raises(tmp_path: Path) -> None:
    """A planner without a prompt file cannot be constructed."""
    with pytest.raises(AgentError, match="prompt file missing"):
        PlannerAgent(
            llm=FakeLLM(),
            config=AgentModelConfig(model="m"),
            journal=Journal(tmp_path / "j.jsonl"),
            run_id="r",
            prompts_dir=tmp_path,
        )


def test_render_strategy_plan_md() -> None:
    """The markdown render includes roadmap and guidance."""
    from slayer.agents.planner import render_strategy_plan_md

    md = render_strategy_plan_md(CANNED)
    assert "Baseline LightGBM" in md
    assert "3-fold" in md or "folds: 3" in md
    assert "stratified folds" in md
