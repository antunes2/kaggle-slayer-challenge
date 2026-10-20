from slayer.config import DefaultsConfig
from slayer.workspace import EventType, ModelCandidate, RunStatus, RunStore

WORKSPACE_DIRS = ("data", "eda", "features", "models", "submissions", "reports", "logs")


def _defaults() -> DefaultsConfig:
    return DefaultsConfig()


def test_create_makes_dirs_run_json_and_event(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    run = store.create("playground-series-s5e10", _defaults())

    assert (tmp_path / "runs" / run.run_id).is_dir()
    for name in WORKSPACE_DIRS:
        assert run.workspace(name).is_dir()
    assert (run.root / "run.json").is_file()
    assert run.journal_path.is_file()
    assert run.status == RunStatus.CREATED
    assert run.target_percentile == 20
    assert run.autonomy == "approve-first"

    events = store.journal(run).read(type=EventType.RUN_CREATED)
    assert len(events) == 1
    assert events[0].data["competition"] == "playground-series-s5e10"
    assert events[0].run == run.run_id


def test_run_id_contains_stamp_and_slug(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    run = store.create("Playground Series S5E10!!", _defaults())
    assert "playground-series-s5e10" in run.run_id
    assert run.competition == "Playground Series S5E10!!"


def test_get_and_list(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    run_a = store.create("comp-a", _defaults())
    run_b = store.create("comp-b", _defaults())

    assert store.get(run_a.run_id).competition == "comp-a"
    listed = store.list()
    assert [r.run_id for r in listed] == [run_a.run_id, run_b.run_id]


def test_get_missing_raises(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    try:
        store.get("nope")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "not found" in str(exc)


def test_update_status_persists_and_journals(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-a", _defaults())
    updated = store.update_status(run, RunStatus.ACTIVE)

    assert updated.status == RunStatus.ACTIVE
    assert store.get(run.run_id).status == RunStatus.ACTIVE
    events = store.journal(run).read(type=EventType.RUN_STATUS)
    assert events[0].data == {"from": "created", "to": "active"}


def test_duplicate_ids_disambiguated(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    run_one = store.create("comp", _defaults())
    run_two = store.create("comp", _defaults())
    assert run_one.run_id != run_two.run_id
    assert run_one.root.is_dir() and run_two.root.is_dir()


def test_registry_wiring(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-a", _defaults())
    registry = store.registry(run)

    record = registry.add(ModelCandidate(model_type="lightgbm"))
    assert record.id == "cand-0001"
    assert run.registry_path.is_file()
    events = store.journal(run).read(type=EventType.CANDIDATE_ADDED)
    assert len(events) == 1
