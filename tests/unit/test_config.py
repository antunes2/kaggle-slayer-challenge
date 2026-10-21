from pathlib import Path

import yaml
from pydantic import ValidationError

from slayer.config import (
    AGENT_NAMES,
    DEFAULT_MODEL,
    load_agents_config,
    load_secrets,
    load_slayer_config,
)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_slayer_config_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = load_slayer_config(tmp_path / "slayer.yaml")
    assert cfg.defaults.target_percentile == 20
    assert cfg.defaults.budget_usd == 10.0
    assert cfg.defaults.submissions_per_day == 4
    assert cfg.defaults.autonomy == "approve-first"
    assert cfg.defaults.dry_run is True
    assert cfg.paths.runs_dir == Path("runs")
    assert cfg.llm.base_url == "https://openrouter.ai/api/v1"


def test_slayer_config_loads_values(tmp_path: Path) -> None:
    path = tmp_path / "slayer.yaml"
    _write_yaml(
        path,
        {"defaults": {"budget_usd": 5.5, "autonomy": "agentic"}, "paths": {"runs_dir": "other"}},
    )
    cfg = load_slayer_config(path)
    assert cfg.defaults.budget_usd == 5.5
    assert cfg.defaults.autonomy == "agentic"
    assert cfg.paths.runs_dir == Path("other")
    assert cfg.defaults.target_percentile == 20


def test_slayer_config_rejects_invalid_autonomy(tmp_path: Path) -> None:
    path = tmp_path / "slayer.yaml"
    _write_yaml(path, {"defaults": {"autonomy": "chaos"}})
    try:
        load_slayer_config(path)
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_agents_config_missing_file_uses_default_model(tmp_path: Path) -> None:
    agents = load_agents_config(tmp_path / "agents.yaml")
    for name in AGENT_NAMES:
        assert getattr(agents, name).model == DEFAULT_MODEL


def test_agents_config_merges_defaults_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "agents.yaml"
    _write_yaml(
        path,
        {
            "defaults": {"model": "vendor/strong", "temperature": 0.3, "max_tokens": 1024},
            "reviewer": {"temperature": 0.0},
        },
    )
    agents = load_agents_config(path)
    assert agents.reviewer.model == "vendor/strong"
    assert agents.reviewer.temperature == 0.0
    assert agents.planner.model == "vendor/strong"
    assert agents.planner.temperature == 0.3
    assert agents.planner.max_tokens == 1024


def test_agents_config_rejects_unknown_agent(tmp_path: Path) -> None:
    path = tmp_path / "agents.yaml"
    _write_yaml(path, {"researcher": {"model": "x"}})
    try:
        load_agents_config(path)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "unknown agent" in str(exc)


def test_secrets_loaded_from_env_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "OPENROUTER_API_KEY=key1234567890\nKAGGLE_API_TOKEN=token1234567890\n", encoding="utf-8"
    )
    secrets = load_secrets(env)
    assert secrets.openrouter_api_key == "key1234567890"
    assert secrets.kaggle_api_token == "token1234567890"


def test_secrets_empty_when_env_file_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    secrets = load_secrets(tmp_path / "nope.env")
    assert secrets.openrouter_api_key == ""
    assert secrets.kaggle_api_token == ""
