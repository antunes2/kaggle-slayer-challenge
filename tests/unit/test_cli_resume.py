import importlib
import json

import yaml
from typer.testing import CliRunner

from slayer.cli.main import app
from slayer.config import DefaultsConfig
from slayer.workspace import ModelCandidate, RunStatus, RunStore
from tests.unit.test_cli import _FakeLLM, _FakeRunKaggle

cli_main = importlib.import_module("slayer.cli.main")

runner = CliRunner()

STRATEGY = {
    "problem_type": "binary_classification",
    "target_column": "target",
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
    "guidance": ["stratified"],
    "summary": "go",
}


def _prepare(tmp_path, monkeypatch):
    """Create a paused run with analysis/planning artifacts and a scored baseline."""
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(cli_main, "KaggleClient", _FakeRunKaggle)
    monkeypatch.setattr(cli_main, "LLMClient", _FakeLLM)

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "slayer.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {"runs_dir": str(tmp_path / "runs")},
                "defaults": {"dry_run": True, "autonomy": "agentic"},
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "agents.yaml").write_text(yaml.safe_dump({"defaults": {}}), encoding="utf-8")

    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig(autonomy="agentic", dry_run=True))
    store.update_status(run, RunStatus.PAUSED)

    reports = run.workspace("reports")
    (reports / "competition.json").write_text(
        json.dumps({"ref": "comp-x", "evaluation_metric": "Roc Auc Score"}), encoding="utf-8"
    )
    (reports / "data_report.json").write_text('{"problem_type": "x"}', encoding="utf-8")
    (reports / "feature_plan.json").write_text('{"ideas": []}', encoding="utf-8")
    (reports / "strategy_plan.json").write_text(json.dumps(STRATEGY), encoding="utf-8")

    eda = run.workspace("eda")
    eda.mkdir(parents=True, exist_ok=True)
    (eda / "profile.md").write_text("# profile", encoding="utf-8")

    registry = store.registry(run)
    record = registry.add(
        ModelCandidate(
            model_type="linear",
            params={"roadmap_step": "Baseline", "priority": 1},
        )
    )
    registry.record_cv(record.id, 0.8, 0.01)
    return config_dir, run


def test_resume_skips_completed_phases_and_finishes(tmp_path, monkeypatch) -> None:
    """Resume skips analysis/planning/baseline and still reviews + submits."""
    config_dir, run = _prepare(tmp_path, monkeypatch)
    env = tmp_path / ".env"
    env.write_text(
        "KAGGLE_API_TOKEN=tok123456789012345\nOPENROUTER_API_KEY=key123456789012345\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["resume", run.run_id, "--config-dir", str(config_dir), "--env-file", str(env)],
    )
    assert result.exit_code == 0
    assert f"resuming run {run.run_id}" in result.output
    assert "analysis artifacts exist — skipping" in result.output
    assert "strategy plan exists — skipping" in result.output
    assert "baseline for 'Baseline' already exists" in result.output
    assert "review" in result.output
    assert "submitted" in result.output

    store = RunStore(tmp_path / "runs")
    final = store.get(run.run_id)
    records = store.registry(final).list()
    assert records[0].status == "submitted"


def test_resume_max_rounds_is_persisted(tmp_path, monkeypatch) -> None:
    """--max-rounds on resume updates run.json and journals the change."""
    config_dir, run = _prepare(tmp_path, monkeypatch)
    env = tmp_path / ".env"
    env.write_text(
        "KAGGLE_API_TOKEN=tok123456789012345\nOPENROUTER_API_KEY=key123456789012345\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["resume", run.run_id, "--max-rounds", "5", "--config-dir", str(config_dir),
         "--env-file", str(env)],
    )
    assert result.exit_code == 0
    store = RunStore(tmp_path / "runs")
    assert store.get(run.run_id).max_rounds == 5
    changes = store.journal(run).read(type="run.settings_changed")
    assert changes[-1].data == {"max_rounds": {"from": 2, "to": 5}}


def test_resume_rejects_out_of_range_max_rounds(tmp_path, monkeypatch) -> None:
    """An invalid round cap exits before touching the run."""
    config_dir, run = _prepare(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["resume", run.run_id, "--max-rounds", "99", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "between 0 and 10" in result.output


def test_resume_unknown_run_fails(tmp_path, monkeypatch) -> None:
    """A bad run id exits with an error."""
    config_dir, _ = _prepare(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["resume", "nope", "--config-dir", str(config_dir)]
    )
    assert result.exit_code == 1
    assert "not found" in result.output
