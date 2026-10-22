import yaml
from typer.testing import CliRunner

from slayer.cli.main import app
from slayer.config import DefaultsConfig
from slayer.dashboard import collect, render_html, render_terminal
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
    assert [p.lb for p in s.points] == [0.949, None, None]


def test_renderers(tmp_path) -> None:
    """Terminal table has the run row; HTML is self-contained with an SVG chart."""
    summaries = collect(_seed(tmp_path))
    text = render_terminal(summaries)
    assert "comp-x" in text and "1r/1m" in text and "0.94900" in text
    page = render_html(summaries, "2026-10-22T00:00:00+00:00")
    assert page.startswith("<!doctype html>") and "<svg" in page and "http" not in page


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
