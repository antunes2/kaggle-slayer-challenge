from pathlib import Path

from slayer.agents import MLResearcherAgent
from slayer.config import AgentModelConfig
from slayer.llm import LLMResponse
from slayer.schemas import PipelineProposal, RoadmapStep
from slayer.workspace import EventType, Journal

CANNED = PipelineProposal(
    model_family="lightgbm",
    description="Baseline LightGBM with class weights",
    pipeline_code=(
        "from lightgbm import LGBMClassifier\n\n"
        "def build_model():\n"
        "    return LGBMClassifier(n_estimators=300)\n"
    ),
    feature_ideas_used=[],
    notes=None,
)

STEP = RoadmapStep(
    title="Raw-feature LightGBM anchor",
    description="LightGBM on raw features",
    model_family="lightgbm",
    priority=1,
)


class FakeLLM:
    """Records requests and returns the canned proposal."""

    def __init__(self) -> None:
        self.calls: list[tuple[list, str, type]] = []

    def complete_structured(self, messages, model, response_model, **kwargs):
        """Return the canned proposal and record the request."""
        self.calls.append((messages, model, response_model))
        return LLMResponse(content="{}", model=model), CANNED


def test_propose_returns_code_and_journals(tmp_path: Path) -> None:
    """The researcher proposal contains code and journals the call."""
    llm = FakeLLM()
    journal = Journal(tmp_path / "journal.jsonl")
    agent = MLResearcherAgent(
        llm=llm,
        config=AgentModelConfig(model="vendor/model"),
        journal=journal,
        run_id="run-1",
    )
    proposal = agent.propose(
        profile_md="# Dataset profile\n- Age: int64",
        data_report_json='{"problem_type": "binary_classification"}',
        feature_plan_json='{"ideas": []}',
        guidance=["stratified folds", "optimize ranking not accuracy"],
        roadmap_step=STEP,
    )

    assert proposal.model_family == "lightgbm"
    assert "build_model" in proposal.pipeline_code

    user_content = llm.calls[0][0][1]["content"]
    assert "Age" in user_content
    assert "Raw-feature LightGBM anchor" in user_content
    assert "optimize ranking not accuracy" in user_content
    assert "matching exactly this schema" in user_content

    calls = journal.read(type=EventType.AGENT_CALL)
    assert len(calls) == 1
    assert calls[0].agent == "ml_researcher"
