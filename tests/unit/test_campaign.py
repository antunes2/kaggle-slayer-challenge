import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from slayer.config import DefaultsConfig, load_agents_config
from slayer.llm import CostReport, LLMResponse
from slayer.loop import PHASE_ORDER, CampaignStop, LoopEngine, Phase, ReentryDecision, RunContext
from slayer.loop.campaign import (
    decide_reentry,
    merge_extension,
    render_history_md,
    replan_phase,
    rounds_done,
)
from slayer.loop.phases import _budget_out
from slayer.schemas import RoadmapExtension, RoadmapStep, StrategyPlan
from slayer.workspace import EventType, ModelCandidate, RunStatus, RunStore, llm_spend

STRATEGY = {
    "problem_type": "binary_classification",
    "target_column": "Target",
    "cv": {"k_folds": 2, "seed": 42, "stratified": True},
    "roadmap": [
        {"title": "Baseline", "description": "lgbm", "model_family": "lightgbm",
         "feature_ideas": [], "priority": 1},
        {"title": "CatBoost", "description": "cb", "model_family": "catboost",
         "feature_ideas": [], "priority": 2},
    ],
    "min_cv_improvement": 0.0005,
    "guidance": ["stratified folds"],
    "summary": "go",
}


def _step(title: str) -> RoadmapStep:
    """A minimal roadmap step."""
    return RoadmapStep(title=title, description="d", model_family="lightgbm")


class FakeReplanLLM:
    """Serves a canned RoadmapExtension and records the user messages it was sent."""

    def __init__(self, extension: RoadmapExtension, cost_usd: float = 0.0) -> None:
        self._extension = extension
        self._cost = CostReport(total_usd=cost_usd)
        self.messages: list[str] = []

    @property
    def cost(self) -> CostReport:
        """Reported accumulated cost."""
        return self._cost

    def complete_structured(self, messages, model, response_model, **kwargs):
        """Return the canned extension (only RoadmapExtension calls are expected)."""
        assert response_model is RoadmapExtension
        self.messages.append(messages[-1]["content"] if isinstance(messages[-1], dict)
                             else messages[-1].content)
        return LLMResponse(content="{}", model=model, cost_usd=0.01), self._extension


def _make_ctx(tmp_path, llm=None, deadline: str | None = None, **defaults) -> RunContext:
    """A run with competition + strategy artifacts and one scored, one failed candidate."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig(**defaults))
    reports = run.workspace("reports")
    (reports / "competition.json").write_text(
        json.dumps({"ref": "comp-x", "evaluation_metric": "Roc Auc Score",
                    "deadline": deadline}),
        encoding="utf-8",
    )
    (reports / "strategy_plan.json").write_text(json.dumps(STRATEGY), encoding="utf-8")
    registry = store.registry(run)
    ok = registry.add(ModelCandidate(model_type="lightgbm",
                                     params={"roadmap_step": "Baseline", "priority": 1}))
    registry.record_cv(ok.id, 0.9415, 0.0008)
    bad = registry.add(ModelCandidate(model_type="catboost",
                                      params={"roadmap_step": "CatBoost", "priority": 2}))
    registry.set_status(bad.id, "failed")
    bad_dir = run.workspace("models") / bad.id
    bad_dir.mkdir(parents=True)
    (bad_dir / "cv_results.json").write_text(
        json.dumps({"status": "error", "error": "CatBoostError: bad cat_features\ntrace"}),
        encoding="utf-8",
    )
    return RunContext(
        run=run,
        journal=store.journal(run),
        registry=registry,
        kaggle=None,
        store=store,
        llm=llm,
        agent_configs=load_agents_config(),
    )


def _position(ctx, met: bool = False, percentile: float = 71.0, gap: float = 0.005) -> None:
    """Journal a MONITOR position."""
    ctx.journal.append(
        agent="loop",
        type=EventType.MONITOR_POSITION,
        data={"basis": "lb", "rank": 2545, "total": 3576, "percentile": percentile,
              "target_percentile": 20, "met": met, "best_score": 0.9415,
              "target_score": 0.9464, "gap": gap},
        run=ctx.run.run_id,
    )


def _spend(ctx, usd: float) -> None:
    """Journal one priced agent call."""
    ctx.journal.append(agent="planner", type=EventType.AGENT_CALL,
                       data={"cost_usd": usd}, run=ctx.run.run_id)


# --- budget -----------------------------------------------------------------


def test_llm_spend_sums_journal_costs(tmp_path) -> None:
    """Priced agent calls are summed; unpriced ones are ignored."""
    ctx = _make_ctx(tmp_path)
    _spend(ctx, 1.25)
    _spend(ctx, 0.5)
    ctx.journal.append(agent="x", type=EventType.AGENT_CALL, data={"cost_usd": None})
    assert llm_spend(ctx.journal) == pytest.approx(1.75)


def test_budget_counts_journal_spend_from_earlier_processes(tmp_path) -> None:
    """A fresh client (cost 0) still stops when earlier resumes spent the budget."""
    ctx = _make_ctx(tmp_path, llm=FakeReplanLLM(None, cost_usd=0.0), budget_usd=2.0)
    _spend(ctx, 2.5)
    assert _budget_out(ctx) is True
    assert ctx.journal.read(type=EventType.BUDGET_WARNING)


# --- the gate ---------------------------------------------------------------


def test_gate_reenters_on_shortfall_and_journals(tmp_path) -> None:
    """Target not met with rounds/budget left → re-enter round 1."""
    ctx = _make_ctx(tmp_path)
    _position(ctx)
    decision = decide_reentry(ctx)
    assert decision.reenter is True
    assert decision.round == 1
    assert "gap +0.00500" in decision.reason
    event = ctx.journal.read(type=EventType.REENTRY_DECISION)[-1].data
    assert event["reenter"] is True and event["percentile"] == 71.0


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        (lambda ctx: None, "no position signal"),
        (lambda ctx: _position(ctx, met=True, percentile=15.0), "target met"),
        (lambda ctx: (_position(ctx), _spend(ctx, 50.0)), "LLM budget exhausted"),
    ],
)
def test_gate_stop_reasons(tmp_path, setup, expected) -> None:
    """Each failing check yields a no-re-entry decision with its reason."""
    ctx = _make_ctx(tmp_path)
    setup(ctx)
    decision = decide_reentry(ctx)
    assert decision.reenter is False
    assert expected in decision.reason


def test_gate_respects_round_cap(tmp_path) -> None:
    """max_rounds=0 disables re-entry."""
    ctx = _make_ctx(tmp_path, max_rounds=0)
    _position(ctx)
    decision = decide_reentry(ctx)
    assert decision.reenter is False
    assert "round cap" in decision.reason


def test_gate_stops_after_planner_stop(tmp_path) -> None:
    """A journaled planner stop is final for the run."""
    ctx = _make_ctx(tmp_path)
    _position(ctx)
    ctx.journal.append(agent="planner", type=EventType.ROADMAP_EXTENDED,
                       data={"round": 1, "steps_added": 0, "stop": True,
                             "stop_reason": "nothing left"})
    decision = decide_reentry(ctx)
    assert decision.reenter is False
    assert "nothing left" in decision.reason


def test_gate_deadline_too_close_vs_passed(tmp_path) -> None:
    """A deadline closer than one harness budget stops; a passed deadline does not."""
    now = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    soon = (now + timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")
    ctx = _make_ctx(tmp_path / "a", deadline=soon, harness_timeout_s=600)
    _position(ctx)
    assert "deadline too close" in decide_reentry(ctx, now=now).reason

    ctx = _make_ctx(tmp_path / "b", deadline="2026-09-30 23:59:00")
    _position(ctx)
    assert decide_reentry(ctx, now=now).reenter is True


# --- merge + replan ---------------------------------------------------------


def test_schema_requires_steps_unless_stop() -> None:
    """An extension with no steps must declare stop."""
    with pytest.raises(ValidationError):
        RoadmapExtension(assessment="x", steps=[])
    assert RoadmapExtension(assessment="x", stop=True, stop_reason="done").stop


def test_merge_makes_titles_unique_and_renumbers() -> None:
    """Colliding titles get a round suffix; priorities continue; guidance dedupes."""
    plan = StrategyPlan.model_validate(STRATEGY)
    extension = RoadmapExtension(
        assessment="a",
        steps=[_step("CatBoost"), _step("Seed bagging"), _step("Seed bagging")],
        new_guidance=["stratified folds", "use TargetEncoder inside the Pipeline"],
    )
    merged, titles = merge_extension(plan, extension, round_number=1)
    assert titles == ["CatBoost (round 1)", "Seed bagging", "Seed bagging (round 1)"]
    assert [s.priority for s in merged.roadmap] == [1, 2, 3, 4, 5]
    assert merged.guidance == ["stratified folds", "use TargetEncoder inside the Pipeline"]


def test_history_md_includes_failure_lines(tmp_path) -> None:
    """Failed candidates show their first error line to the planner."""
    ctx = _make_ctx(tmp_path)
    history = render_history_md(ctx)
    assert "CatBoostError: bad cat_features" in history
    assert "0.94150 ± 0.00080" in history


def test_replan_extends_roadmap(tmp_path) -> None:
    """REPLAN merges new steps into the strategy, writes the round file, and journals."""
    extension = RoadmapExtension(assessment="gap is 10x spread",
                                 steps=[_step("Numerics as categoricals")])
    llm = FakeReplanLLM(extension)
    ctx = _make_ctx(tmp_path, llm=llm)
    _position(ctx)

    replan_phase(ctx)

    plan = StrategyPlan.model_validate_json(
        (ctx.run.workspace("reports") / "strategy_plan.json").read_text())
    assert plan.roadmap[-1].title == "Numerics as categoricals"
    assert plan.roadmap[-1].priority == 3
    assert (ctx.run.workspace("reports") / "roadmap_round_1.json").exists()
    assert rounds_done(ctx.journal) == 1
    sent = llm.messages[0]
    assert "score at the target line: 0.9464" in sent
    assert "CatBoostError" in sent
    assert "round 1 of at most 2" in sent


def test_replan_stop_raises_campaign_stop(tmp_path) -> None:
    """A planner stop journals the reason and raises CampaignStop; strategy unchanged."""
    extension = RoadmapExtension(assessment="exhausted", stop=True, stop_reason="no ideas")
    ctx = _make_ctx(tmp_path, llm=FakeReplanLLM(extension))
    _position(ctx)
    with pytest.raises(CampaignStop, match="no ideas"):
        replan_phase(ctx)
    plan = json.loads((ctx.run.workspace("reports") / "strategy_plan.json").read_text())
    assert len(plan["roadmap"]) == 2
    assert rounds_done(ctx.journal) == 0


# --- engine -----------------------------------------------------------------


def _recording_handlers(ctx, calls: list[str], replan=None) -> dict:
    """No-op handlers for every phase that record their invocation order."""
    def make(phase):
        """Build a recorder for one phase."""
        return lambda c: calls.append(phase.value)

    handlers = {phase: make(phase) for phase in PHASE_ORDER}
    handlers[Phase.REPLAN] = replan or make(Phase.REPLAN)
    return handlers


def test_engine_runs_rounds_until_decider_stops(tmp_path) -> None:
    """Two re-entries then a stop: round phases run twice, journaled with round numbers."""
    ctx = _make_ctx(tmp_path, max_rounds=5)
    calls: list[str] = []
    answers = iter([ReentryDecision(True, "short", 1), ReentryDecision(True, "short", 2),
                    ReentryDecision(False, "target met", 2)])
    final = LoopEngine(ctx, handlers=_recording_handlers(ctx, calls),
                       decider=lambda c: next(answers)).start()

    assert final.status == RunStatus.FINISHED
    assert calls.count("replan") == 2
    assert calls.count("monitor") == 3
    rounds = [e.data.get("round") for e in ctx.journal.read(type=EventType.PHASE_ENTER)
              if e.data.get("phase") == "replan"]
    assert rounds == [1, 2]


def test_engine_hard_caps_rounds_per_invocation(tmp_path) -> None:
    """A decider that always says yes is still bounded by max_rounds."""
    ctx = _make_ctx(tmp_path, max_rounds=2)
    calls: list[str] = []
    LoopEngine(ctx, handlers=_recording_handlers(ctx, calls),
               decider=lambda c: ReentryDecision(True, "short", 1)).start()
    assert calls.count("replan") == 2


def test_engine_campaign_stop_finishes_run(tmp_path) -> None:
    """CampaignStop from REPLAN ends the campaign cleanly (finished, not failed)."""
    ctx = _make_ctx(tmp_path)
    calls: list[str] = []

    def stopper(c):
        """Simulate the planner stopping."""
        raise CampaignStop("no ideas")

    final = LoopEngine(ctx, handlers=_recording_handlers(ctx, calls, replan=stopper),
                       decider=lambda c: ReentryDecision(True, "short", 1)).start()
    assert final.status == RunStatus.FINISHED
    assert calls.count("iterate") == 1


def test_engine_without_decider_keeps_linear_behavior(tmp_path) -> None:
    """Custom handler maps without a decider never re-enter."""
    ctx = _make_ctx(tmp_path)
    calls: list[str] = []
    LoopEngine(ctx, handlers=_recording_handlers(ctx, calls)).start()
    assert "replan" not in calls


def test_engine_with_real_gate_and_replan(tmp_path) -> None:
    """Real gate + real REPLAN: one round runs, then the round cap stops the campaign."""
    extension = RoadmapExtension(assessment="a", steps=[_step("Seed bagging")])
    ctx = _make_ctx(tmp_path, llm=FakeReplanLLM(extension), max_rounds=1)
    calls: list[str] = []
    handlers = _recording_handlers(ctx, calls, replan=replan_phase)
    handlers[Phase.MONITOR] = lambda c: (calls.append("monitor"), _position(c))
    messages: list[str] = []
    ctx.reporter = messages.append

    final = LoopEngine(ctx, handlers=handlers, decider=decide_reentry).start()

    assert final.status == RunStatus.FINISHED
    assert calls.count("monitor") == 2
    assert rounds_done(ctx.journal) == 1
    decisions = [e.data for e in ctx.journal.read(type=EventType.REENTRY_DECISION)]
    assert [d["reenter"] for d in decisions] == [True, False]
    assert any("round cap reached" in m for m in messages)


def test_replan_passes_cached_notebook_digest_as_untrusted_evidence(tmp_path) -> None:
    """A fresh digest in reports/ reaches the planner, framed as untrusted data."""
    extension = RoadmapExtension(assessment="a", steps=[_step("Digit features")])
    llm = FakeReplanLLM(extension)
    ctx = _make_ctx(tmp_path, llm=llm)
    _position(ctx)
    (ctx.run.workspace("reports") / "notebook_digest.md").write_text(
        "# Public notebook digest\ndf['d'] = df['x'] // 10 % 10\n", encoding="utf-8"
    )
    replan_phase(ctx)
    sent = llm.messages[0]
    assert "UNTRUSTED evidence" in sent
    assert "df['d'] = df['x'] // 10 % 10" in sent


def test_replan_tells_planner_when_no_external_data_exists(tmp_path) -> None:
    """Without a manifest the planner is told external data is unavailable."""
    llm = FakeReplanLLM(RoadmapExtension(assessment="a", steps=[_step("X")]))
    ctx = _make_ctx(tmp_path, llm=llm)
    _position(ctx)
    replan_phase(ctx)
    assert "External data: NONE available" in llm.messages[0]
