import importlib

from typer.testing import CliRunner

from slayer.cli.main import app

cli_main = importlib.import_module("slayer.cli.main")

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
    monkeypatch.setattr(cli_main, "_llm_ping", lambda *a, **kw: (True, "ok (mocked)"))
    monkeypatch.setattr(cli_main, "_kaggle_ping", lambda *a, **kw: (True, "ok (user=mocked)"))
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
    assert "openrouter: ok (mocked)" in result.output
    assert "kaggle:     ok (user=mocked)" in result.output


def test_init_fails_when_llm_ping_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(cli_main, "_llm_ping", lambda *a, **kw: (False, "failed (mocked)"))
    monkeypatch.setattr(cli_main, "_kaggle_ping", lambda *a, **kw: (True, "ok (mocked)"))
    env = tmp_path / ".env"
    env.write_text(
        "OPENROUTER_API_KEY=key123456789012345\nKAGGLE_API_TOKEN=tok1234567890123456\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path / "cfg"), "--env-file", str(env)],
    )
    assert result.exit_code == 1
    assert "openrouter: failed (mocked)" in result.output


def test_init_fails_when_kaggle_ping_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(cli_main, "_llm_ping", lambda *a, **kw: (True, "ok (mocked)"))
    monkeypatch.setattr(cli_main, "_kaggle_ping", lambda *a, **kw: (False, "failed (mocked)"))
    env = tmp_path / ".env"
    env.write_text(
        "OPENROUTER_API_KEY=key123456789012345\nKAGGLE_API_TOKEN=tok1234567890123456\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["init", "--config-dir", str(tmp_path / "cfg"), "--env-file", str(env)],
    )
    assert result.exit_code == 1
    assert "kaggle:     failed (mocked)" in result.output


def test_init_uses_real_config_dir(monkeypatch) -> None:
    monkeypatch.setattr(cli_main, "_llm_ping", lambda *a, **kw: (True, "ok (mocked)"))
    monkeypatch.setattr(cli_main, "_kaggle_ping", lambda *a, **kw: (True, "ok (mocked)"))
    result = runner.invoke(app, ["init"])
    assert "config/slayer.yaml: loaded" in result.output
    assert "agents.yaml: loaded" in result.output


class _FakeRunKaggle:
    @classmethod
    def from_token(cls, token: str) -> "_FakeRunKaggle":
        return cls()

    def find_competition(self, ref: str, **kwargs):
        from slayer.kaggle import CompetitionInfo

        return CompetitionInfo(
            ref=ref,
            title="Comp X",
            evaluation_metric="AUC",
            deadline="2026-11-01",
            max_daily_submissions=5,
        )

    def download_files(self, competition: str, target_dir, force: bool = False):
        import zipfile

        zip_path = target_dir / f"{competition}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("train.csv", "id,target\n1,0\n")
        return zip_path


def test_run_command_end_to_end(tmp_path, monkeypatch) -> None:
    import yaml

    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(cli_main, "KaggleClient", _FakeRunKaggle)

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "slayer.yaml").write_text(
        yaml.safe_dump({"paths": {"runs_dir": str(tmp_path / "runs")}}), encoding="utf-8"
    )
    env = tmp_path / ".env"
    env.write_text("KAGGLE_API_TOKEN=tok123456789012345\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["run", "comp-x", "--config-dir", str(config_dir), "--env-file", str(env)],
    )
    assert result.exit_code == 0
    assert "run created" in result.output
    assert "paused" in result.output

    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "workspace" / "data" / "train.csv").is_file()
    assert (runs[0] / "journal.jsonl").is_file()


def test_run_command_requires_kaggle_token(tmp_path, monkeypatch) -> None:
    import yaml

    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "slayer.yaml").write_text(
        yaml.safe_dump({"paths": {"runs_dir": str(tmp_path / "runs")}}), encoding="utf-8"
    )
    result = runner.invoke(
        app,
        ["run", "comp-x", "--config-dir", str(config_dir), "--env-file", str(tmp_path / "no.env")],
    )
    assert result.exit_code == 1
    assert "KAGGLE_API_TOKEN missing" in result.output
