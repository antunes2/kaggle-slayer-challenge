import json

import pytest

from slayer.analysis import AnalysisError
from slayer.config import DefaultsConfig, load_agents_config
from slayer.llm import LLMResponse
from slayer.loop import RunContext
from slayer.loop.phases import analysis_phase
from slayer.schemas import DataAnalystOutput, DataReport, FeatureIdea, FeaturePlan
from slayer.workspace import EventType, RunStore

CANNED = DataAnalystOutput(
    report=DataReport(
        problem_type="binary_classification",
        target_column="Target",
        summary="ok",
        key_findings=["f1", "f2"],
        data_quality_issues=["missing City"],
        risks=["small data"],
    ),
    feature_plan=FeaturePlan(
        ideas=[
            FeatureIdea(name="i1", rationale="r", expected_value="high", based_on_columns=["Age"])
        ]
    ),
)


class FakeLLM:
    def complete_structured(self, messages, model, response_model, **kwargs):
        return LLMResponse(content="{}", model=model), CANNED


def _make_ctx(tmp_path, llm) -> RunContext:
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    data_dir = run.workspace("data")
    (data_dir / "train.csv").write_text(
        "id,Age,City,Target\n0,25,X,1\n1,35,Y,0\n2,45,X,1\n", encoding="utf-8"
    )
    (data_dir / "test.csv").write_text("id,Age,City\n5,60,Z\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,Target\n5,0.5\n", encoding="utf-8")
    (run.workspace("reports") / "competition.json").write_text(
        '{"ref": "comp-x", "evaluation_metric": "AUC"}', encoding="utf-8"
    )
    return RunContext(
        run=run,
        journal=store.journal(run),
        registry=store.registry(run),
        kaggle=None,
        store=store,
        llm=llm,
        agent_configs=load_agents_config(),
    )


def test_analysis_phase_end_to_end(tmp_path) -> None:
    ctx = _make_ctx(tmp_path, FakeLLM())
    messages: list[str] = []
    ctx.reporter = messages.append

    analysis_phase(ctx)

    reports = ctx.run.workspace("reports")
    report = json.loads((reports / "data_report.json").read_text(encoding="utf-8"))
    assert report["problem_type"] == "binary_classification"
    plan = json.loads((reports / "feature_plan.json").read_text(encoding="utf-8"))
    assert plan["ideas"][0]["name"] == "i1"
    report_md = (reports / "data_report.md").read_text(encoding="utf-8")
    assert "Key findings" in report_md

    assert (ctx.run.workspace("eda") / "profile.md").is_file()

    agent_calls = ctx.journal.read(type=EventType.AGENT_CALL)
    assert len(agent_calls) == 1
    assert agent_calls[0].agent == "data_analyst"
    outputs = ctx.journal.read(type=EventType.AGENT_OUTPUT)
    assert outputs[0].data["problem_type"] == "binary_classification"
    assert outputs[0].data["feature_ideas"] == 1

    assert any("profiled train" in m for m in messages)
    assert any("data report" in m for m in messages)


def test_analysis_phase_requires_llm(tmp_path) -> None:
    ctx = _make_ctx(tmp_path, llm=None)
    with pytest.raises(AnalysisError, match="LLM client"):
        analysis_phase(ctx)
