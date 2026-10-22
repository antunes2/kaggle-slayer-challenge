
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

    def leaderboard(self, competition: str, page_size: int = 50):
        """Return the canned leaderboard entries."""
        return self._entries

    def ping(self) -> str:
        """Return the canned username."""
        return self._username


class _Sub:
    """Fake SubmissionInfo."""

    def __init__(self, ref: str, status: str, public_score: str | None) -> None:
        self.ref = ref
        self.status = status
        self.public_score = public_score


class _Entry:
    """Fake LeaderboardEntry."""

    def __init__(self, rank: int, team_name: str, score: str) -> None:
        self.rank = rank
        self.team_name = team_name
        self.score = score


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
        entries=[_Entry(1, "someone-else", "0.94999")],
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
    assert any("not found on the fetched page" in m for m in messages)


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
        entries=[_Entry(1, "rival", "0.95"), _Entry(2, "gantunes2", "0.94212")],
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
        "rank": 2,
        "total": 2,
        "percentile": 100.0,
        "target_percentile": 20,
        "met": False,
    }


def test_monitor_position_met_when_within_target(tmp_path) -> None:
    """Rank inside the target percentile reports MET."""
    entries = [_Entry(1, "gantunes2", "0.95")]
    entries += [_Entry(i, f"team-{i}", "0.90") for i in range(2, 7)]
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
