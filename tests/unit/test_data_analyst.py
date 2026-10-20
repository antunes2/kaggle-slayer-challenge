from pathlib import Path

import pytest

from slayer.agents import Agent, AgentError, DataAnalystAgent
from slayer.config import AgentModelConfig
from slayer.llm import LLMResponse
from slayer.schemas import DataAnalystOutput, DataReport, FeatureIdea, FeaturePlan
from slayer.workspace import EventType, Journal

CANNED = DataAnalystOutput(
    report=DataReport(
        problem_type="binary_classification",
        target_column="Target",
        summary="Balanced binary tabular dataset.",
        key_findings=["Age correlates with target"],
        data_quality_issues=[],
        risks=["small dataset"],
    ),
    feature_plan=FeaturePlan(
        ideas=[
            FeatureIdea(
                name="age_bucket",
                rationale="Age shows signal",
                expected_value="medium",
                based_on_columns=["Age"],
            )
        ]
    ),
)


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[list, str, type]] = []

    def complete_structured(self, messages, model, response_model, **kwargs):
        self.calls.append((messages, model, response_model))
        return LLMResponse(content="{}", model=model), CANNED


def _journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal.jsonl")


def test_analyze_returns_output_and_journals_call(tmp_path) -> None:
    llm = FakeLLM()
    journal = _journal(tmp_path)
    agent = DataAnalystAgent(
        llm=llm,
        config=AgentModelConfig(model="vendor/model", temperature=0.2),
        journal=journal,
        run_id="run-1",
    )
    output = agent.analyze(profile_md="# profile", competition_json='{"ref": "c"}')

    assert output.report.problem_type == "binary_classification"
    assert output.feature_plan.ideas[0].name == "age_bucket"

    assert len(llm.calls) == 1
    messages, model, response_model = llm.calls[0]
    assert model == "vendor/model"
    assert response_model is DataAnalystOutput
    assert messages[0]["role"] == "system"
    assert "data analyst" in messages[0]["content"].lower()
    assert "# profile" in messages[1]["content"]
    assert '{"ref": "c"}' in messages[1]["content"]
    assert "problem_type" in messages[1]["content"]
    assert "matching exactly this schema" in messages[1]["content"]

    calls = journal.read(type=EventType.AGENT_CALL)
    assert len(calls) == 1
    assert calls[0].agent == "data_analyst"
    assert calls[0].data["model"] == "vendor/model"
    assert calls[0].run == "run-1"


def test_agent_missing_prompt_raises(tmp_path) -> None:
    llm = FakeLLM()
    with pytest.raises(AgentError, match="prompt file missing"):
        DataAnalystAgent(
            llm=llm,
            config=AgentModelConfig(model="m"),
            journal=_journal(tmp_path),
            run_id="run-1",
            prompts_dir=tmp_path,
        )


def test_agent_name_and_prompt_file(tmp_path) -> None:
    llm = FakeLLM()
    agent = DataAnalystAgent(
        llm=llm,
        config=AgentModelConfig(model="m"),
        journal=_journal(tmp_path),
        run_id="run-1",
    )
    assert agent.name == "data_analyst"
    assert agent.prompt_file == "data_analyst"
    assert Agent.name == "agent"
