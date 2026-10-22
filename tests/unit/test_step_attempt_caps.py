import json

import pytest

from slayer.config import DefaultsConfig, load_agents_config
from slayer.cv import CandidateRunError
from slayer.loop import RunContext
from slayer.loop.phases import MAX_STEP_ATTEMPTS, baseline_phase, iterate_phase
from slayer.schemas import PipelineProposal
from slayer.workspace import ModelCandidate, RunStore
from tests.unit.test_baseline_phase import GOOD_CODE as BASELINE_CODE
from tests.unit.test_baseline_phase import FakeLLM
from tests.unit.test_iterate_phase import GOOD_CODE, FakeIterateLLM, _make_ctx, _proposal

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
    "min_cv_improvement": 0.0005,
    "guidance": ["stratified folds"],
    "summary": "go",
}


def test_max_step_attempts_constant() -> None:
    """The retry cap is 3 (initial + repair + one resume retry)."""
    assert MAX_STEP_ATTEMPTS == 3


def test_iterate_retries_failed_step_only_under_cap(tmp_path) -> None:
    """A step with 2 failed candidates gets exactly one more attempt, then is capped."""
    llm = FakeIterateLLM([_proposal(GOOD_CODE)])
    ctx = _make_ctx(tmp_path, llm)
    for _ in range(2):
        record = ctx.registry.add(
            ModelCandidate(
                model_type="catboost", params={"roadmap_step": "Boosted trees", "priority": 2}
            )
        )
        ctx.registry.set_status(record.id, "failed")

    iterate_phase(ctx)

    assert llm.proposal_calls.count(PipelineProposal) == 1
    new_records = [
        r
        for r in ctx.registry.list()
        if r.params.get("roadmap_step") == "Boosted trees" and r.status != "failed"
    ]
    assert len(new_records) == 1


def test_iterate_skips_step_at_attempt_cap(tmp_path) -> None:
    """A step with MAX_STEP_ATTEMPTS failed candidates is never retried again."""
    llm = FakeIterateLLM([])
    ctx = _make_ctx(tmp_path, llm)
    for _ in range(MAX_STEP_ATTEMPTS):
        record = ctx.registry.add(
            ModelCandidate(
                model_type="catboost", params={"roadmap_step": "Boosted trees", "priority": 2}
            )
        )
        ctx.registry.set_status(record.id, "failed")
    messages: list[str] = []
    ctx.reporter = messages.append

    iterate_phase(ctx)

    assert llm.proposal_calls.count(PipelineProposal) == 0
    assert any("no pending roadmap steps" in m for m in messages)


def test_baseline_exhausted_raises(tmp_path) -> None:
    """Baseline with MAX_STEP_ATTEMPTS failures refuses to retry blindly."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    reports = run.workspace("reports")
    (reports / "competition.json").write_text(
        json.dumps({"ref": "comp-x", "evaluation_metric": "Roc Auc Score"}), encoding="utf-8"
    )
    (reports / "data_report.json").write_text("{}", encoding="utf-8")
    (reports / "feature_plan.json").write_text("{}", encoding="utf-8")
    (reports / "strategy_plan.json").write_text(json.dumps(STRATEGY), encoding="utf-8")
    (run.workspace("eda") / "profile.md").write_text("# p", encoding="utf-8")
    registry = store.registry(run)
    for _ in range(MAX_STEP_ATTEMPTS):
        record = registry.add(
            ModelCandidate(model_type="linear", params={"roadmap_step": "Baseline", "priority": 1})
        )
        registry.set_status(record.id, "failed")
    ctx = RunContext(
        run=run,
        journal=store.journal(run),
        registry=registry,
        kaggle=None,
        store=store,
        llm=FakeLLM(_proposal(BASELINE_CODE)),
        agent_configs=load_agents_config(),
    )

    with pytest.raises(CandidateRunError, match="exhausted 3 attempts"):
        baseline_phase(ctx)
