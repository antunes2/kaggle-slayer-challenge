import pytest

from slayer.workspace import (
    CandidateRegistry,
    CandidateStatus,
    EventType,
    Journal,
    ModelCandidate,
)


@pytest.fixture()
def journal(tmp_path):
    return Journal(tmp_path / "journal.jsonl")


@pytest.fixture()
def registry(tmp_path, journal):
    return CandidateRegistry(tmp_path / "experiments.db", run_id="run-1", journal=journal)


def test_add_and_get(registry) -> None:
    record = registry.add(
        ModelCandidate(model_type="lightgbm", params={"n_estimators": 500}, notes="baseline")
    )
    assert record.id == "cand-0001"
    assert record.status == CandidateStatus.PROPOSED
    assert record.run_id == "run-1"
    fetched = registry.get("cand-0001")
    assert fetched.model_type == "lightgbm"
    assert fetched.params == {"n_estimators": 500}
    assert fetched.notes == "baseline"


def test_sequential_ids(registry) -> None:
    registry.add(ModelCandidate(model_type="lightgbm"))
    registry.add(ModelCandidate(model_type="catboost"))
    third = registry.add(ModelCandidate(model_type="xgboost"))
    assert third.id == "cand-0003"


def test_lineage(registry) -> None:
    parent = registry.add(ModelCandidate(model_type="lightgbm"))
    child = registry.add(ModelCandidate(model_type="lightgbm", parent_id=parent.id))
    assert registry.get(child.id).parent_id == "cand-0001"


def test_record_cv_auto_scores_and_emits(registry, journal) -> None:
    record = registry.add(ModelCandidate(model_type="lightgbm"))
    updated = registry.record_cv(record.id, 0.84321, 0.0021)

    assert updated.status == CandidateStatus.SCORED
    assert updated.cv_score == pytest.approx(0.84321)
    assert updated.cv_std == pytest.approx(0.0021)
    assert journal.read(type=EventType.CANDIDATE_CV)[0].data["cv_score"] == 0.84321
    statuses = journal.read(type=EventType.CANDIDATE_STATUS)
    assert statuses[0].data == {"id": record.id, "from": "proposed", "to": "scored"}


def test_record_lb_marks_submitted(registry, journal) -> None:
    record = registry.add(ModelCandidate(model_type="lightgbm"))
    registry.set_status(record.id, CandidateStatus.APPROVED)
    updated = registry.record_lb(record.id, 0.84001, "sub-77")

    assert updated.status == CandidateStatus.SUBMITTED
    assert updated.lb_score == pytest.approx(0.84001)
    assert updated.submission_id == "sub-77"
    assert len(journal.read(type=EventType.CANDIDATE_LB)) == 1
    assert len(journal.read(type=EventType.CANDIDATE_STATUS)) == 2


def test_set_status(registry) -> None:
    record = registry.add(ModelCandidate(model_type="catboost"))
    updated = registry.set_status(record.id, CandidateStatus.REJECTED)
    assert updated.status == CandidateStatus.REJECTED
    assert registry.list(status=CandidateStatus.REJECTED)[0].id == record.id


def test_best_higher_is_better(registry) -> None:
    for cv in (0.80, 0.91, 0.85):
        record = registry.add(ModelCandidate(model_type="lightgbm"))
        registry.record_cv(record.id, cv)
    best = registry.best(2)
    assert [r.cv_score for r in best] == [0.91, 0.85]


def test_best_lower_is_better(registry) -> None:
    for cv in (0.20, 0.09, 0.15):
        record = registry.add(ModelCandidate(model_type="lightgbm"))
        registry.record_cv(record.id, cv)
    best = registry.best(2, higher_is_better=False)
    assert [r.cv_score for r in best] == [0.09, 0.15]


def test_best_skips_unscored(registry) -> None:
    registry.add(ModelCandidate(model_type="lightgbm"))
    scored = registry.add(ModelCandidate(model_type="catboost"))
    registry.record_cv(scored.id, 0.7)
    assert [r.id for r in registry.best()] == [scored.id]


def test_best_submission_filters(registry) -> None:
    a = registry.add(ModelCandidate(model_type="a"))
    b = registry.add(ModelCandidate(model_type="b"))
    registry.record_cv(a.id, 0.9)
    registry.record_cv(b.id, 0.8)
    registry.record_lb(a.id, 0.88, "sub-1")

    assert [r.id for r in registry.best(submitted_only=True)] == [a.id]
    assert [r.id for r in registry.best(unsubmitted_only=True)] == [b.id]


def test_summary_for_agent(registry) -> None:
    assert registry.summary_for_agent() == "No candidates registered yet."
    record = registry.add(ModelCandidate(model_type="lightgbm", notes="baseline"))
    registry.record_cv(record.id, 0.84321)
    summary = registry.summary_for_agent()
    assert "cand-0001" in summary
    assert "lightgbm" in summary
    assert "0.84321" in summary


def test_rebuild_from_journal(tmp_path, registry, journal) -> None:
    registry.add(ModelCandidate(model_type="lightgbm", params={"lr": 0.05}, notes="v1"))
    registry.record_cv("cand-0001", 0.84, 0.01)
    registry.set_status("cand-0001", CandidateStatus.APPROVED)
    registry.add(ModelCandidate(model_type="catboost", parent_id="cand-0001"))
    registry.record_cv("cand-0002", 0.87)
    registry.record_lb("cand-0001", 0.85, "sub-9")

    rebuilt = CandidateRegistry.rebuild(
        tmp_path / "experiments-rebuilt.db", run_id="run-1", journal=journal
    )
    original_records = [r.model_dump() for r in registry.list()]
    rebuilt_records = [r.model_dump() for r in rebuilt.list()]
    assert rebuilt_records == original_records


def test_no_journal_means_no_events(tmp_path) -> None:
    silent = CandidateRegistry(tmp_path / "silent.db", run_id="run-2", journal=None)
    record = silent.add(ModelCandidate(model_type="lightgbm"))
    silent.record_cv(record.id, 0.9)
    assert silent.get(record.id).cv_score == 0.9


def test_missing_candidate_raises(registry) -> None:
    with pytest.raises(KeyError, match="not found"):
        registry.get("cand-9999")
