import json

import pytest

from slayer.config import DefaultsConfig, load_agents_config
from slayer.cv import CandidateRunError
from slayer.llm import LLMResponse
from slayer.loop import RunContext
from slayer.loop.phases import baseline_phase
from slayer.schemas import PipelineProposal
from slayer.workspace import RunStore

GOOD_CODE = (
    "from sklearn.linear_model import LogisticRegression\n\n"
    "def build_model():\n"
    "    return LogisticRegression()\n"
)

STRATEGY = {
    "problem_type": "binary_classification",
    "target_column": "Target",
    "cv": {"k_folds": 2, "seed": 42, "stratified": True},
    "roadmap": [
        {
            "title": "Baseline",
            "description": "logistic regression",
            "model_family": "linear",
            "feature_ideas": [],
            "priority": 1,
        }
    ],
    "min_cv_improvement": 0.0005,
    "guidance": ["stratified folds"],
    "summary": "go",
}

DATA = {
    "train.csv": "id,x,Target\n0,1,Yes\n1,2,Yes\n2,3,No\n3,4,No\n4,1.5,Yes\n5,3.5,No\n",
    "test.csv": "id,x\n6,2.2\n",
    "sample_submission.csv": "id,Target\n6,0.5\n",
}


class FakeLLM:
    """Returns a configurable canned pipeline proposal."""

    def __init__(self, proposal: PipelineProposal) -> None:
        self._proposal = proposal

    def complete_structured(self, messages, model, response_model, **kwargs):
        """Return the canned proposal."""
        return LLMResponse(content="{}", model=model), self._proposal


def _proposal(code: str) -> PipelineProposal:
    """Build a canned proposal with the given code."""
    return PipelineProposal(
        model_family="linear",
        description="test candidate",
        pipeline_code=code,
    )


def _make_ctx(tmp_path, llm) -> RunContext:
    """Create a run with all baseline prerequisites on disk."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    data_dir = run.workspace("data")
    for name, content in DATA.items():
        (data_dir / name).write_text(content, encoding="utf-8")
    reports = run.workspace("reports")
    (reports / "competition.json").write_text(
        json.dumps({"ref": "comp-x", "evaluation_metric": "Roc Auc Score"}), encoding="utf-8"
    )
    (reports / "data_report.json").write_text('{"problem_type": "x"}', encoding="utf-8")
    (reports / "feature_plan.json").write_text('{"ideas": []}', encoding="utf-8")
    (reports / "strategy_plan.json").write_text(json.dumps(STRATEGY), encoding="utf-8")
    (run.workspace("eda") / "profile.md").write_text("# profile\n- x: int64", encoding="utf-8")
    return RunContext(
        run=run,
        journal=store.journal(run),
        registry=store.registry(run),
        kaggle=None,
        store=store,
        llm=llm,
        agent_configs=load_agents_config(),
    )


def test_baseline_phase_scores_candidate(tmp_path) -> None:
    """A good proposal runs through the harness and records a scored candidate."""
    ctx = _make_ctx(tmp_path, FakeLLM(_proposal(GOOD_CODE)))
    messages: list[str] = []
    ctx.reporter = messages.append

    baseline_phase(ctx)

    records = ctx.registry.list()
    assert len(records) == 1
    assert records[0].status == "scored"
    assert records[0].cv_score is not None
    candidate_dir = ctx.run.workspace("models") / records[0].id
    assert (candidate_dir / "pipeline.py").is_file()
    assert (candidate_dir / "submission.csv").is_file()
    assert (candidate_dir / "cv_results.json").is_file()
    assert any("scored:" in m for m in messages)


def test_baseline_phase_syntax_error_fails_fast(tmp_path) -> None:
    """A syntactically broken proposal fails before spawning the harness."""
    ctx = _make_ctx(tmp_path, FakeLLM(_proposal("def broken(:\n    pass\n")))
    candidate_ids_before = len(ctx.registry.list())

    with pytest.raises(CandidateRunError, match="syntax error"):
        baseline_phase(ctx)

    records = ctx.registry.list()
    assert len(records) == candidate_ids_before + 1
    assert records[-1].status == "failed"
    assert not (ctx.run.workspace("models") / records[-1].id / "submission.csv").exists()


def test_baseline_phase_harness_failure_marks_failed(tmp_path) -> None:
    """A pipeline that crashes inside the harness is recorded as failed."""
    ctx = _make_ctx(
        tmp_path,
        FakeLLM(_proposal("def build_model():\n    raise RuntimeError('nope')\n")),
    )

    with pytest.raises(CandidateRunError, match="failed the harness"):
        baseline_phase(ctx)

    assert ctx.registry.list()[-1].status == "failed"


def test_baseline_phase_strips_code_fences(tmp_path) -> None:
    """Markdown-fenced code is unwrapped before being written."""
    fenced = f"```python\n{GOOD_CODE}```"
    ctx = _make_ctx(tmp_path, FakeLLM(_proposal(fenced)))

    baseline_phase(ctx)

    record = ctx.registry.list()[-1]
    written = (ctx.run.workspace("models") / record.id / "pipeline.py").read_text(
        encoding="utf-8"
    )
    assert not written.startswith("```")
    assert "build_model" in written
