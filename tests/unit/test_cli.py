from typer.testing import CliRunner

from slayer.cli.main import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "slayer" in result.output


def test_init_fails_when_secrets_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path / "cfg"), "--env-file", str(tmp_path / "none.env")],
    )
    assert result.exit_code == 1
    assert "MISSING" in result.output


def test_init_passes_with_secrets(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "OPENROUTER_API_KEY=key123456789012345\nKAGGLE_API_TOKEN=tok1234567890123456\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path / "cfg"), "--env-file", str(env)],
    )
    assert result.exit_code == 0
    assert "Environment ready" in result.output


def test_init_uses_real_config_dir() -> None:
    result = runner.invoke(app, ["init"])
    assert "config/slayer.yaml: loaded" in result.output
    assert "agents.yaml: loaded" in result.output
