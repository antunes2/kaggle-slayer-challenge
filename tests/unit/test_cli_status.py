import yaml
from typer.testing import CliRunner

from slayer.cli.main import app
from slayer.config import DefaultsConfig
from slayer.workspace import EventType, RunStore

runner = CliRunner()


def _config_dir(tmp_path):
    """Create a config dir pointing runs at a tmp directory."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "slayer.yaml").write_text(
        yaml.safe_dump({"paths": {"runs_dir": str(tmp_path / "runs")}}), encoding="utf-8"
    )
    return config_dir


def _make_run(tmp_path, competition: str = "comp-x"):
    """Create a real run with a phase event on disk."""
    store = RunStore(tmp_path / "runs")
    run = store.create(competition, DefaultsConfig())
    store.journal(run).append(
        agent="loop", type=EventType.PHASE_ENTER, data={"phase": "init"}, run=run.run_id
    )
    return run


def test_status_lists_runs(tmp_path) -> None:
    _make_run(tmp_path, "comp-a")
    _make_run(tmp_path, "comp-b")
    result = runner.invoke(app, ["status", "--config-dir", str(_config_dir(tmp_path))])
    assert result.exit_code == 0
    assert "comp-a" in result.output
    assert "comp-b" in result.output


def test_status_shows_run_details(tmp_path) -> None:
    run = _make_run(tmp_path, "comp-x")
    result = runner.invoke(app, ["status", run.run_id, "--config-dir", str(_config_dir(tmp_path))])
    assert result.exit_code == 0
    assert "comp-x" in result.output
    assert "init" in result.output
    assert "workspace:" in result.output


def test_status_unknown_run_fails(tmp_path) -> None:
    result = runner.invoke(app, ["status", "nope", "--config-dir", str(_config_dir(tmp_path))])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_status_shows_llm_spend(tmp_path) -> None:
    """Accumulated agent-call costs from the journal render against the budget."""
    run = _make_run(tmp_path, "comp-x")
    store = RunStore(tmp_path / "runs")
    journal = store.journal(run)
    journal.append(
        agent="planner", type=EventType.AGENT_CALL,
        data={"cost_usd": 0.0042, "model": "m1", "prompt_tokens": 100, "completion_tokens": 50},
        run=run.run_id,
    )
    journal.append(
        agent="reviewer", type=EventType.AGENT_CALL,
        data={"cost_usd": 0.0011, "model": "m1", "prompt_tokens": 80, "completion_tokens": 40},
        run=run.run_id,
    )
    journal.append(
        agent="data_analyst", type=EventType.AGENT_CALL,
        data={"cost_usd": None, "model": "free", "prompt_tokens": 10, "completion_tokens": 5},
        run=run.run_id,
    )

    result = runner.invoke(app, ["status", run.run_id, "--config-dir", str(_config_dir(tmp_path))])
    assert result.exit_code == 0
    assert "llm spend:   $0.0053 / $10.00 budget (3 calls)" in result.output


def test_status_zero_spend_shows_free_note(tmp_path) -> None:
    """With no priced calls the spend line notes the free model."""
    run = _make_run(tmp_path, "comp-x")
    result = runner.invoke(app, ["status", run.run_id, "--config-dir", str(_config_dir(tmp_path))])
    assert result.exit_code == 0
    assert "llm spend:   $0.0000" in result.output
    assert "free/unpriced model" in result.output


def test_status_no_runs(tmp_path) -> None:
    result = runner.invoke(app, ["status", "--config-dir", str(_config_dir(tmp_path))])
    assert result.exit_code == 0
    assert "No runs yet" in result.output


def test_status_shows_rounds_and_hides_stale_pause(tmp_path) -> None:
    """Finished runs show campaign rounds/decision and no leftover 'paused' line."""
    run = _make_run(tmp_path, "comp-x")
    store = RunStore(tmp_path / "runs")
    journal = store.journal(run)
    journal.append(agent="loop", type=EventType.LOOP_PAUSED, data={"reason": "old pause"})
    journal.append(agent="planner", type=EventType.ROADMAP_EXTENDED,
                   data={"round": 1, "steps_added": 2})
    journal.append(agent="loop", type=EventType.REENTRY_DECISION,
                   data={"reenter": False, "reason": "round cap reached (1/2)"})
    store.update_status(run, "finished")

    result = runner.invoke(app, ["status", run.run_id, "--config-dir", str(_config_dir(tmp_path))])
    assert result.exit_code == 0
    assert "old pause" not in result.output
    assert "rounds:      1 / 2 re-entry rounds" in result.output
    assert "campaign:    stopped — round cap reached (1/2)" in result.output


def test_journal_shows_events(tmp_path) -> None:
    run = _make_run(tmp_path, "comp-x")
    result = runner.invoke(
        app, ["journal", run.run_id, "--config-dir", str(_config_dir(tmp_path))]
    )
    assert result.exit_code == 0
    assert "phase.enter" in result.output
    assert "init" in result.output


def test_journal_unknown_run_fails(tmp_path) -> None:
    result = runner.invoke(app, ["journal", "nope", "--config-dir", str(_config_dir(tmp_path))])
    assert result.exit_code == 1
    assert "not found" in result.output
