import json

from slayer.config import DefaultsConfig, load_agents_config
from slayer.llm import CostReport, LLMResponse
from slayer.loop import RunContext
from slayer.loop.phases import iterate_phase
from slayer.schemas import PipelineProposal, ReviewVerdict
from slayer.submissions import MockSink
from slayer.workspace import EventType, ModelCandidate, RunStore

STRATEGY = {
    "problem_type": "binary_classification",
    "target_column": "Target",
    "cv": {"k_folds": 2, "seed": 42, "stratified": True},
    "roadmap": [
        {
            "title": "Baseline",
            "description": "lr baseline",
            "model_family": "linear",
            "feature_ideas": [],
            "priority": 1,
        },
        {
            "title": "Boosted trees",
            "description": "lightgbm with feature ideas",
            "model_family": "lightgbm",
            "feature_ideas": ["i1"],
            "priority": 2,
        },
    ],
    "min_cv_improvement": 0.0005,
    "guidance": ["stratified folds"],
    "summary": "go",
}

GOOD_CODE = (
    "from sklearn.linear_model import LogisticRegression\n\n"
    "def build_model():\n"
    "    return LogisticRegression()\n"
)

BROKEN_CODE = "def broken(:\n    pass\n"

APPROVE = ReviewVerdict(decision="approve", reasoning="solid", risk_flags=[])

DATA = {
    "train.csv": "id,x,Target\n0,1,Yes\n1,2,Yes\n2,3,No\n3,4,No\n4,1.5,Yes\n5,3.5,No\n",
    "test.csv": "id,x\n6,2.2\n",
    "sample_submission.csv": "id,Target\n6,0.5\n",
}


def _proposal(code: str) -> PipelineProposal:
    """Build a canned proposal with the given code."""
    return PipelineProposal(model_family="linear", description="candidate", pipeline_code=code)


class FakeIterateLLM:
    """Serves a proposal sequence and a fixed verdict, with a configurable cost."""

    def __init__(self, proposals, verdict=APPROVE, cost_usd: float = 0.0) -> None:
        self._proposals = list(proposals)
        self._verdict = verdict
        self._cost = CostReport(total_usd=cost_usd)
        self.proposal_calls: list[dict] = []

    @property
    def cost(self) -> CostReport:
        """Reported accumulated cost."""
        return self._cost

    def complete_structured(self, messages, model, response_model, **kwargs):
        """Serve canned proposals or the verdict."""
        self.proposal_calls.append({})
        from slayer.schemas import PipelineProposal as PP
        from slayer.schemas import ReviewVerdict as RV

        if response_model is PP:
            return LLMResponse(content="{}", model=model), self._proposals.pop(0)
        assert response_model is RV
        return LLMResponse(content="{}", model=model), self._verdict


def _make_ctx(tmp_path, llm) -> RunContext:
    """Create a run with step 1 already baselined and step 2 pending."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig(autonomy="agentic", dry_run=True))
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

    registry = store.registry(run)
    seed = registry.add(
        ModelCandidate(model_type="linear", params={"roadmap_step": "Baseline", "priority": 1})
    )
    registry.record_cv(seed.id, 0.0, 0.001)
    registry.set_status(seed.id, "approved")
    registry.record_lb(seed.id, 0.0, "mock-seed")
    return RunContext(
        run=run,
        journal=store.journal(run),
        registry=registry,
        kaggle=None,
        store=store,
        llm=llm,
        agent_configs=load_agents_config(),
        sink=MockSink(),
        approver=lambda prompt: True,
    )


def test_iterate_runs_pending_step_and_submits(tmp_path) -> None:
    """The pending roadmap step produces a scored, approved, mock-submitted candidate."""
    ctx = _make_ctx(tmp_path, FakeIterateLLM([_proposal(GOOD_CODE)]))
    messages: list[str] = []
    ctx.reporter = messages.append

    iterate_phase(ctx)

    records = ctx.registry.list()
    assert len(records) == 2
    new_record = [r for r in records if r.params.get("roadmap_step") == "Boosted trees"][0]
    assert new_record.status == "submitted"
    assert new_record.cv_score is not None
    assert new_record.lb_score is not None

    mocked = ctx.journal.read(type=EventType.SUBMISSION_MOCKED)
    assert len(mocked) == 1
    assert any("iterating step 2" in m for m in messages)
    assert any("scored:" in m for m in messages)


def test_iterate_repairs_failed_attempt(tmp_path) -> None:
    """A broken first attempt triggers one repair with the error fed back."""
    llm = FakeIterateLLM([_proposal(BROKEN_CODE), _proposal(GOOD_CODE)])
    ctx = _make_ctx(tmp_path, llm)

    iterate_phase(ctx)

    failed = [r for r in ctx.registry.list() if r.status == "failed"]
    assert len(failed) == 1
    final = [r for r in ctx.registry.list() if r.params.get("roadmap_step") == "Boosted trees"
             and r.status == "submitted"]
    assert len(final) == 1


def test_iterate_skips_step_after_double_failure(tmp_path) -> None:
    """Two failures for the same step exhaust repair attempts and move on."""
    llm = FakeIterateLLM([_proposal(BROKEN_CODE), _proposal(BROKEN_CODE)])
    ctx = _make_ctx(tmp_path, llm)
    messages: list[str] = []
    ctx.reporter = messages.append

    iterate_phase(ctx)

    failed = [r for r in ctx.registry.list() if r.status == "failed"]
    assert len(failed) == 2
    assert any("failed again" in m for m in messages)


def test_iterate_stops_on_budget(tmp_path) -> None:
    """An exhausted LLM budget stops iteration before any proposal."""
    llm = FakeIterateLLM([_proposal(GOOD_CODE)], cost_usd=99.0)
    ctx = _make_ctx(tmp_path, llm)

    iterate_phase(ctx)

    assert len(ctx.registry.list()) == 1
    assert ctx.journal.read(type=EventType.BUDGET_WARNING)


def test_iterate_no_pending_steps_makes_no_calls(tmp_path) -> None:
    """When every roadmap step already has a candidate, iterate does nothing."""
    llm = FakeIterateLLM([])
    ctx = _make_ctx(tmp_path, llm)
    seed_two = ctx.registry.add(
        ModelCandidate(model_type="lightgbm", params={"roadmap_step": "Boosted trees",
                                                       "priority": 2})
    )
    ctx.registry.record_cv(seed_two.id, 0.94, 0.001)
    messages: list[str] = []
    ctx.reporter = messages.append

    iterate_phase(ctx)

    assert llm.proposal_calls == []
    assert any("nothing to iterate" in m for m in messages)
