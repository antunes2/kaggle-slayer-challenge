
from slayer.config import DefaultsConfig
from slayer.loop import RunContext
from slayer.loop.phases import monitor_phase
from slayer.workspace import EventType, ModelCandidate, RunStore


class FakeMonitorKaggle:
    """Fake kaggle client for monitoring: submissions list + leaderboard + ping."""

    def __init__(self, submissions=(), entries=(), username="gantunes2") -> None:
        self._submissions = list(submissions)
        self._entries = list(entries)
        self._username = username

    def submissions(self, competition: str):
        """Return the canned submissions list."""
        return self._submissions

    def full_leaderboard(self, competition: str, target_dir=None):
        """Return the canned full leaderboard rows."""
        return self._entries

    def ping(self) -> str:
        """Return the canned username."""
        return self._username


class _Sub:
    """Fake SubmissionInfo."""

    def __init__(
        self, ref: str, status: str, public_score: str | None, description: str | None = None
    ) -> None:
        self.ref = ref
        self.status = status
        self.public_score = public_score
        self.description = description


class _Entry:
    """Fake FullLeaderboardRow (float scores, like the real model)."""

    def __init__(self, rank: int, team_name: str, score: float, usernames=()) -> None:
        self.rank = rank
        self.team_name = team_name
        self.score = score
        self.member_usernames = list(usernames)


def _make_ctx(tmp_path, kaggle, target: int = 20):
    """Create a run with a candidate submitted to Kaggle, awaiting its LB score."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig(target_percentile=target))
    registry = store.registry(run)
    record = registry.add(ModelCandidate(model_type="lightgbm"))
    registry.record_cv(record.id, 0.941, 0.001)
    registry.set_status(record.id, "approved")
    registry.record_submission(record.id, "57503219")
    return RunContext(
        run=run,
        journal=store.journal(run),
        registry=registry,
        kaggle=kaggle,
        store=store,
    )


def test_monitor_records_pending_lb_score(tmp_path) -> None:
    """A submitted candidate with a real LB score gets it recorded."""
    kaggle = FakeMonitorKaggle(
        submissions=[_Sub("57503219", "COMPLETE", "0.94212")],
        entries=[_Entry(1, "someone-else", 0.94999)],
    )
    ctx = _make_ctx(tmp_path, kaggle)
    messages: list[str] = []
    ctx.reporter = messages.append

    monitor_phase(ctx)

    record = ctx.registry.list()[0]
    assert record.lb_score == 0.94212
    assert record.submission_id == "57503219"
    assert ctx.journal.read(type=EventType.CANDIDATE_LB)
    assert any("LB score 0.94212" in m for m in messages)
    assert any("not listed — best LB score 0.94212" in m for m in messages)
    assert any("would rank ~2/2" in m for m in messages)
    position = ctx.journal.read(type=EventType.MONITOR_POSITION)[-1].data
    assert position["basis"] == "lb"
    assert position["estimated"] is True


def test_monitor_skips_unscored_and_mock_submissions(tmp_path) -> None:
    """Pending kaggle scores and mock submission ids are left alone."""
    kaggle = FakeMonitorKaggle(submissions=[_Sub("57503219", "PENDING", None)])
    ctx = _make_ctx(tmp_path, kaggle)
    messages: list[str] = []
    ctx.reporter = messages.append

    registry = ctx.registry
    mock = registry.add(ModelCandidate(model_type="catboost"))
    registry.record_cv(mock.id, 0.95, 0.001)
    registry.record_lb(mock.id, 0.94, "mock-cand-0002")

    monitor_phase(ctx)

    assert ctx.registry.list()[0].lb_score is None
    assert ctx.registry.get(mock.id).lb_score == 0.94
    assert any("no LB score yet" in m for m in messages)


def test_monitor_reports_position_vs_target(tmp_path) -> None:
    """Our leaderboard entry is located and compared against the target percentile."""
    kaggle = FakeMonitorKaggle(
        submissions=[_Sub("57503219", "COMPLETE", "0.94212")],
        entries=[
            _Entry(1, "rival", 0.95),
            _Entry(2, "Gabriel A", 0.94212, usernames=["gantunes2"]),
        ],
    )
    ctx = _make_ctx(tmp_path, kaggle, target=20)
    messages: list[str] = []
    ctx.reporter = messages.append

    monitor_phase(ctx)

    assert any("rank 2/2" in m for m in messages)
    assert any("NOT met" in m for m in messages)
    position = ctx.journal.read(type=EventType.MONITOR_POSITION)
    assert position[0].data == {
        "our_entry": True,
        "basis": "listed",
        "rank": 2,
        "total": 2,
        "percentile": 100.0,
        "target_percentile": 20,
        "met": False,
        "best_score": 0.94212,
        "target_score": 0.95,
        "gap": 0.00788,
    }


def test_monitor_position_met_when_within_target(tmp_path) -> None:
    """Rank inside the target percentile reports MET."""
    entries = [_Entry(1, "Gabriel A", 0.95, usernames=["gantunes2"])]
    entries += [_Entry(i, f"team-{i}", 0.90) for i in range(2, 7)]
    kaggle = FakeMonitorKaggle(entries=entries)
    ctx = _make_ctx(tmp_path, kaggle, target=20)
    messages: list[str] = []
    ctx.reporter = messages.append

    monitor_phase(ctx)

    assert any("MET" in m for m in messages)
    position = ctx.journal.read(type=EventType.MONITOR_POSITION)
    assert position[0].data["met"] is True


def test_monitor_requires_kaggle_client(tmp_path) -> None:
    """Without a kaggle client the phase fails loudly."""
    ctx = _make_ctx(tmp_path, kaggle=None)
    try:
        monitor_phase(ctx)
        raise AssertionError("expected AnalysisError")
    except Exception as exc:
        assert "kaggle client" in str(exc)


def test_monitor_matches_submissions_by_description(tmp_path) -> None:
    """Candidates submitted before record_submission existed are matched via description."""
    kaggle = FakeMonitorKaggle(
        submissions=[
            _Sub(
                "57503219",
                "COMPLETE",
                "0.94183",
                description="slayer cand-0001: LightGBM CV=0.94172",
            )
        ],
        entries=[],
    )
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    registry = store.registry(run)
    record = registry.add(ModelCandidate(model_type="lightgbm"))
    registry.record_cv(record.id, 0.94172, 0.001)
    registry.set_status(record.id, "submitted")
    ctx = RunContext(
        run=run,
        journal=store.journal(run),
        registry=registry,
        kaggle=kaggle,
        store=store,
    )
    messages: list[str] = []
    ctx.reporter = messages.append

    monitor_phase(ctx)

    updated = ctx.registry.get(record.id)
    assert updated.lb_score == 0.94183
    assert updated.submission_id == "57503219"
    assert any("LB score 0.94183" in m for m in messages)


def _field(n: int = 10) -> list:
    """A descending-score leaderboard of n teams: 0.950, 0.949, ..."""
    return [_Entry(i + 1, f"team-{i}", round(0.950 - i * 0.001, 3)) for i in range(n)]


def test_monitor_reports_gap_to_target_line(tmp_path) -> None:
    """The target line is the score at rank floor(total * target%); gap is direction-aware."""
    kaggle = FakeMonitorKaggle(
        submissions=[_Sub("57503219", "COMPLETE", "0.9425")], entries=_field(10)
    )
    ctx = _make_ctx(tmp_path, kaggle, target=20)
    messages: list[str] = []
    ctx.reporter = messages.append

    monitor_phase(ctx)

    position = ctx.journal.read(type=EventType.MONITOR_POSITION)[-1].data
    assert position["target_score"] == 0.949
    assert position["gap"] == 0.0065
    assert position["met"] is False
    assert any("target line 0.94900, gap +0.00650" in m for m in messages)


def test_monitor_uses_cv_proxy_without_real_lb(tmp_path) -> None:
    """Dry runs (mock LB only) estimate the position from best CV, flagged basis=cv."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    registry = store.registry(run)
    record = registry.add(ModelCandidate(model_type="lightgbm"))
    registry.record_cv(record.id, 0.9455, 0.001)
    registry.record_lb(record.id, 0.9999, "mock-cand-0001")
    ctx = RunContext(
        run=run,
        journal=store.journal(run),
        registry=registry,
        kaggle=FakeMonitorKaggle(entries=_field(10)),
        store=store,
    )
    messages: list[str] = []
    ctx.reporter = messages.append

    monitor_phase(ctx)

    position = ctx.journal.read(type=EventType.MONITOR_POSITION)[-1].data
    assert position["basis"] == "cv"
    assert position["best_score"] == 0.9455
    assert position["rank"] == 6
    assert any("best CV (no real LB score yet)" in m for m in messages)
