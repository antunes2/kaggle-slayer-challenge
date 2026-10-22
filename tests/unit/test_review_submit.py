import json

from slayer.config import DefaultsConfig, load_agents_config
from slayer.kaggle import QuotaManager
from slayer.llm import LLMResponse
from slayer.loop import RunContext
from slayer.loop.phases import review_phase, submit_phase
from slayer.schemas import ReviewVerdict
from slayer.submissions import KaggleSink, MockSink
from slayer.workspace import EventType, ModelCandidate, RunStore

STRATEGY = {
    "problem_type": "binary_classification",
    "target_column": "Target",
    "cv": {"k_folds": 2, "seed": 42, "stratified": True},
    "roadmap": [
        {
            "title": "Baseline",
            "description": "lr",
            "model_family": "linear",
            "feature_ideas": [],
            "priority": 1,
        }
    ],
    "min_cv_improvement": 0.001,
    "guidance": ["stratified folds"],
    "summary": "go",
}

PIPELINE = (
    "from sklearn.linear_model import LogisticRegression\n\n"
    "def build_model():\n"
    "    return LogisticRegression()\n"
)


class FakeReviewLLM:
    """Returns a configurable canned verdict."""

    def __init__(self, verdict: ReviewVerdict) -> None:
        self._verdict = verdict

    def complete_structured(self, messages, model, response_model, **kwargs):
        """Return the canned verdict."""
        return LLMResponse(content="{}", model=model), self._verdict


class FakeKaggleTransport:
    """Fake kaggle client for the real sink."""

    def __init__(self) -> None:
        self.submitted: list[str] = []

    def submit(self, competition, file_path, message=""):
        """Record and fake a submission id."""
        from slayer.kaggle import SubmissionRef

        self.submitted.append(str(file_path))
        return SubmissionRef(id="kaggle-777", message="ok")


def _make_ctx(tmp_path, verdict: ReviewVerdict, autonomy: str = "agentic",
              dry_run: bool = True) -> RunContext:
    """Create a run with a scored candidate and review prerequisites."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig(autonomy=autonomy, dry_run=dry_run))
    reports = run.workspace("reports")
    (reports / "competition.json").write_text(
        json.dumps({"ref": "comp-x", "evaluation_metric": "Roc Auc Score"}), encoding="utf-8"
    )
    (reports / "data_report.json").write_text('{"problem_type": "x"}', encoding="utf-8")
    (reports / "strategy_plan.json").write_text(json.dumps(STRATEGY), encoding="utf-8")

    registry = store.registry(run)
    record = registry.add(ModelCandidate(model_type="lightgbm"))
    registry.record_cv(record.id, 0.941, 0.001)
    candidate_dir = run.workspace("models") / record.id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "pipeline.py").write_text(PIPELINE, encoding="utf-8")
    (candidate_dir / "cv_results.json").write_text('{"status": "ok"}', encoding="utf-8")
    (candidate_dir / "submission.csv").write_text("id,Target\n6,0.5\n", encoding="utf-8")
    return RunContext(
        run=run,
        journal=store.journal(run),
        registry=registry,
        kaggle=None,
        store=store,
        llm=FakeReviewLLM(verdict),
        agent_configs=load_agents_config(),
        sink=MockSink() if dry_run else KaggleSink(FakeKaggleTransport()),
        approver=lambda prompt: True,
    )


APPROVE = ReviewVerdict(decision="approve", reasoning="solid baseline", risk_flags=[])
REJECT = ReviewVerdict(decision="reject", reasoning="leakage risk", risk_flags=["leak"])


def test_review_approve_sets_status(tmp_path) -> None:
    """An approve verdict marks the candidate approved."""
    ctx = _make_ctx(tmp_path, APPROVE)
    review_phase(ctx)
    assert ctx.registry.list()[0].status == "approved"
    outputs = ctx.journal.read(type=EventType.AGENT_OUTPUT)
    assert outputs[-1].agent == "reviewer"
    assert outputs[-1].data["decision"] == "approve"


def test_review_reject_sets_status(tmp_path) -> None:
    """A reject verdict marks the candidate rejected."""
    ctx = _make_ctx(tmp_path, REJECT)
    review_phase(ctx)
    assert ctx.registry.list()[0].status == "rejected"


def test_review_revise_keeps_scored(tmp_path) -> None:
    """A revise verdict leaves the candidate scored with feedback journaled."""
    ctx = _make_ctx(tmp_path, ReviewVerdict(decision="revise", reasoning="fix weights"))
    review_phase(ctx)
    assert ctx.registry.list()[0].status == "scored"


def test_review_without_candidates_noop(tmp_path) -> None:
    """With nothing to review the phase passes silently."""
    ctx = _make_ctx(tmp_path, APPROVE)
    ctx.registry.set_status(ctx.registry.list()[0].id, "rejected")
    review_phase(ctx)


def test_submit_mock_records_submission_without_quota(tmp_path) -> None:
    """Mock submission records lb + status and consumes no quota."""
    ctx = _make_ctx(tmp_path, APPROVE)
    review_phase(ctx)
    submit_phase(ctx)

    record = ctx.registry.list()[0]
    assert record.status == "submitted"
    assert record.lb_score is not None
    assert ctx.journal.read(type=EventType.SUBMISSION_MOCKED)
    assert ctx.journal.read(type=EventType.SUBMISSION_ACCEPTED) == []


def test_submit_cv_gate_blocks_small_improvement(tmp_path) -> None:
    """A second candidate below the improvement gate is not submitted."""
    ctx = _make_ctx(tmp_path, APPROVE)
    review_phase(ctx)
    submit_phase(ctx)
    first = ctx.registry.list()[0]

    better = ctx.registry.add(ModelCandidate(model_type="catboost"))
    ctx.registry.record_cv(better.id, first.cv_score + 0.0001)
    ctx.registry.set_status(better.id, "approved")
    submit_phase(ctx)

    still_scored = [r for r in ctx.registry.list() if r.id == better.id][0]
    assert still_scored.status == "approved"
    assert still_scored.lb_score is None


def test_submit_cv_gate_allows_improvement(tmp_path) -> None:
    """A candidate clearing the gate gets submitted."""
    ctx = _make_ctx(tmp_path, APPROVE)
    review_phase(ctx)
    submit_phase(ctx)
    first = ctx.registry.list()[0]

    better = ctx.registry.add(ModelCandidate(model_type="catboost"))
    ctx.registry.record_cv(better.id, first.cv_score + 0.01)
    ctx.registry.set_status(better.id, "approved")
    submit_phase(ctx)

    assert ctx.registry.get(better.id).status == "submitted"


def test_submit_approve_first_requires_approval(tmp_path) -> None:
    """With approve-first, a declining approver blocks the submission."""
    ctx = _make_ctx(tmp_path, APPROVE, autonomy="approve-first")
    ctx.approver = lambda prompt: False
    review_phase(ctx)
    submit_phase(ctx)

    record = ctx.registry.list()[0]
    assert record.status == "approved"
    assert record.lb_score is None


def test_submit_kaggle_sink_consumes_quota(tmp_path) -> None:
    """The real sink journals submission.accepted and the quota manager sees it."""
    ctx = _make_ctx(tmp_path, APPROVE, dry_run=False)
    review_phase(ctx)
    submit_phase(ctx)

    assert ctx.journal.read(type=EventType.SUBMISSION_ACCEPTED)
    assert ctx.registry.list()[0].status == "submitted"
    quota = QuotaManager(ctx.journal, ctx.run.submissions_per_day)
    assert quota.used_today() == 1


REVISE = ReviewVerdict(decision="revise", reasoning="type digits as categorical", risk_flags=[])


def _write_folds(ctx, candidate_id: str, scores: list[float]) -> None:
    """Write a cv_results.json with the given fold scores."""
    folder = ctx.run.workspace("models") / candidate_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cv_results.json").write_text(json.dumps(
        {"status": "ok", "folds": [{"fold": i, "score": s} for i, s in enumerate(scores)]}
    ), encoding="utf-8")


def _submitted_base_and_challenger(tmp_path, challenger_folds):
    """A submitted base candidate and an approved challenger with given fold scores."""
    ctx = _make_ctx(tmp_path, APPROVE)
    review_phase(ctx)
    submit_phase(ctx)
    base = ctx.registry.list()[0]
    base_folds = [0.940, 0.941, 0.942, 0.943, 0.944]
    _write_folds(ctx, base.id, base_folds)
    challenger = ctx.registry.add(ModelCandidate(model_type="catboost"))
    mean = sum(challenger_folds) / len(challenger_folds)
    ctx.registry.record_cv(challenger.id, base.cv_score + (mean - sum(base_folds) / 5))
    ctx.registry.set_status(challenger.id, "approved")
    _write_folds(ctx, challenger.id, challenger_folds)
    return ctx, challenger


def test_submit_paired_gate_accepts_consistent_small_gain(tmp_path) -> None:
    """+0.0002 on every fold (below the 0.001 gate) is a real, paired improvement."""
    ctx, challenger = _submitted_base_and_challenger(
        tmp_path, [0.9402, 0.94121, 0.94219, 0.9432, 0.94418]
    )
    messages: list[str] = []
    ctx.reporter = messages.append
    submit_phase(ctx)
    assert ctx.registry.get(challenger.id).status == "submitted"
    assert any("paired gain" in m and "5/5 folds" in m for m in messages)


def test_submit_paired_gate_rejects_inconsistent_gain(tmp_path) -> None:
    """A small mean gain that flips sign across folds stays blocked."""
    ctx, challenger = _submitted_base_and_challenger(
        tmp_path, [0.9410, 0.9400, 0.9430, 0.9420, 0.9450]
    )
    submit_phase(ctx)
    assert ctx.registry.get(challenger.id).status == "approved"


def test_review_skips_candidates_that_already_have_a_verdict(tmp_path) -> None:
    """A REVISE'd candidate is not re-reviewed; the next unreviewed one is."""
    ctx = _make_ctx(tmp_path, REVISE)
    review_phase(ctx)
    first = ctx.registry.list()[0]
    second = ctx.registry.add(ModelCandidate(model_type="catboost"))
    ctx.registry.record_cv(second.id, first.cv_score - 0.001)
    (ctx.run.workspace("models") / second.id).mkdir(parents=True, exist_ok=True)
    review_phase(ctx)
    verdicts = ctx.journal.read(type=EventType.AGENT_OUTPUT, agent="reviewer")
    assert [v.data["candidate_id"] for v in verdicts] == [first.id, second.id]
    messages: list[str] = []
    ctx.reporter = messages.append
    review_phase(ctx)
    assert any("nothing to review" in m for m in messages)


def test_pending_revision_targets_best_unrevised_candidate(tmp_path) -> None:
    """The best REVISE'd candidate is pending even if a later verdict approved another."""
    from slayer.loop.phases import _pending_revision

    ctx = _make_ctx(tmp_path, REVISE)
    review_phase(ctx)
    best = ctx.registry.list()[0]
    other = ctx.registry.add(ModelCandidate(model_type="catboost"))
    ctx.registry.record_cv(other.id, best.cv_score - 0.001)
    ctx.journal.append(agent="reviewer", type=EventType.AGENT_OUTPUT,
                       data={"candidate_id": other.id, "decision": "approve"})
    record, verdict = _pending_revision(ctx)
    assert record.id == best.id and verdict["decision"] == "revise"

    fix = ctx.registry.add(ModelCandidate(model_type="lightgbm",
                                          params={"revision_of": best.id}))
    assert fix and _pending_revision(ctx) is None
