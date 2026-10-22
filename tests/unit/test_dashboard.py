import yaml
from typer.testing import CliRunner

from slayer.cli.main import app
from slayer.config import DefaultsConfig
from slayer.dashboard import (
    aggregate_competitions,
    collect,
    render_html,
    render_terminal,
)
from slayer.workspace import EventType, ModelCandidate, RunStore

runner = CliRunner()


def _seed(tmp_path):
    """One live run with a real LB, a mock submission, a failure, and a position."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig(dry_run=False))
    registry = store.registry(run)
    a = registry.add(ModelCandidate(model_type="lightgbm"))
    registry.record_cv(a.id, 0.951, 0.001)
    registry.record_lb(a.id, 0.949, "57500001")
    b = registry.add(ModelCandidate(model_type="xgboost"))
    registry.record_cv(b.id, 0.953, 0.001)
    registry.record_lb(b.id, 0.999, "mock-cand-0002")
    c = registry.add(ModelCandidate(model_type="catboost"))
    registry.set_status(c.id, "failed")
    journal = store.journal(run)
    journal.append(agent="submission", type=EventType.SUBMISSION_ACCEPTED, data={})
    journal.append(agent="submission", type=EventType.SUBMISSION_MOCKED, data={})
    journal.append(agent="planner", type=EventType.AGENT_CALL, data={"cost_usd": 0.25})
    journal.append(agent="planner", type=EventType.ROADMAP_EXTENDED, data={"steps_added": 2})
    journal.append(agent="loop", type=EventType.MONITOR_POSITION,
                   data={"rank": 10, "total": 100, "percentile": 10.0, "basis": "listed"})
    return store


def test_summary_counts_and_scores(tmp_path) -> None:
    """Real vs mock scores are separated; counts, rounds, spend, and position derive."""
    (s,) = collect(_seed(tmp_path))
    assert (s.candidates, s.scored, s.failed) == (3, 2, 1)
    assert (s.real_submissions, s.mock_submissions, s.rounds) == (1, 1, 1)
    assert s.best_cv == 0.953 and s.best_lb == 0.949
    assert s.position == "10/100 (top 10.0%)" and s.spend_usd == 0.25
    assert (s.rank, s.total, s.target_percentile) == (10, 100, 20)
    assert [p.lb for p in s.points] == [0.949, None, None]


def test_renderers(tmp_path) -> None:
    """Terminal table has the run row; HTML is self-contained with an SVG chart."""
    summaries = collect(_seed(tmp_path))
    text = render_terminal(summaries)
    assert "comp-x" in text and "1r/1m" in text and "0.94900" in text
    page = render_html(summaries, "2026-10-22T00:00:00+00:00")
    assert page.startswith("<!doctype html>") and "<svg" in page and "http" not in page
    assert 'data-filter="comp-x"' in page and "class=\"chip" in page
    assert "Score trajectory" in page and "data-series=\"cv\"" in page
    assert "10.00%" in page and "target top 20%" in page


def test_cli_dashboard_writes_html(tmp_path) -> None:
    """`slayer dashboard --html` prints the table and writes the report."""
    _seed(tmp_path)
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    runs = str(tmp_path / "runs")
    (cfg / "slayer.yaml").write_text(yaml.safe_dump({"paths": {"runs_dir": runs}}))
    out = tmp_path / "report.html"
    result = runner.invoke(app, ["dashboard", "--config-dir", str(cfg), "--html", str(out)])
    assert result.exit_code == 0 and "comp-x" in result.output
    assert out.exists() and "<svg" in out.read_text()

def test_competition_aggregation_and_compare_view(tmp_path) -> None:
    """Per-competition rollups and the head-to-head comparison view derive from runs."""
    store = _seed(tmp_path)
    second = store.create("comp-y", DefaultsConfig(dry_run=True))
    registry = store.registry(second)
    candidate = registry.add(ModelCandidate(model_type="catboost"))
    registry.record_cv(candidate.id, 0.80, 0.001)
    journal = store.journal(second)
    journal.append(agent="loop", type=EventType.MONITOR_POSITION,
                   data={"rank": 5, "total": 100, "percentile": 5.0, "basis": "listed"})
    journal.append(agent="planner", type=EventType.ROADMAP_EXTENDED,
                   data={"steps_added": 3})

    summaries = collect(store)
    competitions = aggregate_competitions(summaries)
    assert [c.competition for c in competitions] == ["comp-y", "comp-x"]
    x = next(c for c in competitions if c.competition == "comp-x")
    y = next(c for c in competitions if c.competition == "comp-y")
    assert (x.campaigns, x.best_cv, x.best_percentile, x.spend_usd) == (1, 0.953, 10.0, 0.25)
    assert (y.campaigns, y.dry, y.best_percentile, y.rounds) == (1, 1, 5.0, 1)

    page = render_html(summaries, "2026-10-22T00:00:00+00:00")
    assert 'id="view-compare"' in page and "Competition head-to-head" in page
    assert 'data-view="compare"' in page and "Best finish vs LLM spend" in page
    assert "Head-to-head table" in page and "cmp-point" in page
