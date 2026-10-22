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

    def list_files(self, competition: str):
        from types import SimpleNamespace

        return [SimpleNamespace(name="train.csv", total_bytes=10)]

    def ping(self) -> str:
        """Fake authenticated username."""
        return "test-user"

    def submissions(self, competition: str):
        """No prior submissions."""
        return []

    def full_leaderboard(self, competition: str, target_dir=None):
        """Empty leaderboard."""
        return []

    def download_files(self, competition: str, target_dir, force: bool = False):
        import zipfile

        zip_path = target_dir / f"{competition}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "train.csv",
                "id,x,target\n0,5,0\n1,6,1\n2,7,0\n3,8,1\n4,5.5,0\n5,6.5,1\n6,7.5,0\n7,8.5,1\n",
            )
            zf.writestr("test.csv", "id,x\n8,1.2\n9,3.3\n")
            zf.writestr("sample_submission.csv", "id,target\n8,0.5\n9,0.5\n")
        return zip_path


class _FakeLLM:
    """Fake LLM that dispatches canned outputs by response model."""

    def __init__(self, api_key: str = "", **kwargs) -> None:
        pass

    @staticmethod
    def _analyst_output():
        """Canned data analyst output."""
        from slayer.schemas import DataAnalystOutput, DataReport, FeatureIdea, FeaturePlan

        return DataAnalystOutput(
            report=DataReport(
                problem_type="binary_classification",
                target_column="target",
                summary="ok",
                key_findings=["f"],
                data_quality_issues=[],
                risks=[],
            ),
            feature_plan=FeaturePlan(
                ideas=[
                    FeatureIdea(
                        name="i1", rationale="r", expected_value="medium", based_on_columns=["x"]
                    )
                ]
            ),
        )

    @staticmethod
    def _strategy_plan():
        """Canned strategy plan."""
        from slayer.schemas import CVSettings, RoadmapStep, StrategyPlan

        return StrategyPlan(
            problem_type="binary_classification",
            target_column="target",
            cv=CVSettings(k_folds=2, seed=1, stratified=True),
            roadmap=[
                RoadmapStep(
                    title="Baseline",
                    description="lightgbm raw",
                    model_family="lightgbm",
                    priority=1,
                )
            ],
            min_cv_improvement=0.0005,
            guidance=["stratified"],
            summary="ok",
        )

    @staticmethod
    def _pipeline_proposal():
        """Canned runnable pipeline proposal."""
        from slayer.schemas import PipelineProposal

        return PipelineProposal(
            model_family="linear",
            description="logistic regression baseline",
            pipeline_code=(
                "from sklearn.linear_model import LogisticRegression\n\n"
                "def build_model():\n"
                "    return LogisticRegression()\n"
            ),
            feature_ideas_used=[],
            notes=None,
        )

    @staticmethod
    def _review_verdict():
        """Canned reviewer verdict."""
        from slayer.schemas import ReviewVerdict

        return ReviewVerdict(
            decision="approve",
            reasoning="baseline is sound",
            risk_flags=[],
        )

    def complete_structured(self, messages, model, response_model, **kwargs):
        """Return the canned output matching the requested model."""
        from slayer.llm import LLMResponse
        from slayer.schemas import DataAnalystOutput, PipelineProposal, ReviewVerdict, StrategyPlan

        canned = {
            DataAnalystOutput: self._analyst_output(),
            StrategyPlan: self._strategy_plan(),
            PipelineProposal: self._pipeline_proposal(),
            ReviewVerdict: self._review_verdict(),
        }[response_model]
        return LLMResponse(content="{}", model=model), canned


def test_run_command_end_to_end(tmp_path, monkeypatch) -> None:
    import yaml

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
    env = tmp_path / ".env"
    env.write_text(
        "KAGGLE_API_TOKEN=tok123456789012345\nOPENROUTER_API_KEY=key123456789012345\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["run", "comp-x", "--config-dir", str(config_dir), "--env-file", str(env)],
    )
    assert result.exit_code == 0
    assert "run created" in result.output
    assert "finished" in result.output
    assert "submitted" in result.output

    runs = [child for child in (tmp_path / "runs").iterdir() if (child / "run.json").is_file()]
    assert len(runs) == 1
    assert (runs[0] / "workspace" / "data" / "train.csv").is_file()
    assert (runs[0] / "workspace" / "reports" / "data_report.json").is_file()
    assert (runs[0] / "workspace" / "reports" / "strategy_plan.json").is_file()
    assert (runs[0] / "workspace" / "models").is_dir()
    assert len(list((runs[0] / "workspace" / "models").iterdir())) == 1
    assert (runs[0] / "journal.jsonl").is_file()

    journal_text = (runs[0] / "journal.jsonl").read_text(encoding="utf-8")
    assert "submission.mocked" in journal_text


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
    assert "KAGGLE_API_TOKEN or OPENROUTER_API_KEY missing" in result.output


class _FakeListKaggle:
    """Fake kaggle client for `slayer competitions`."""

    calls: list[dict] = []

    @classmethod
    def from_token(cls, token: str) -> "_FakeListKaggle":
        return cls()

    def list_competitions(self, **kwargs):
        from slayer.kaggle import CompetitionInfo

        _FakeListKaggle.calls.append(kwargs)
        return [CompetitionInfo(ref="playground-series-s6e10", title="Predict Things",
                                deadline="2026-11-30 23:59:00", evaluation_metric="Roc Auc Score",
                                max_daily_submissions=5, user_has_entered=True)]


def test_competitions_lists_refs(tmp_path, monkeypatch) -> None:
    """The browse command prints refs, deadlines, metrics, and a run hint."""
    monkeypatch.setattr(cli_main, "KaggleClient", _FakeListKaggle)
    env = tmp_path / ".env"
    env.write_text("KAGGLE_API_TOKEN=tok123456789012345\n", encoding="utf-8")
    result = runner.invoke(app, ["competitions", "--env-file", str(env), "-c", "all"])
    assert result.exit_code == 0
    assert "playground-series-s6e10" in result.output
    assert "deadline 2026-11-30" in result.output and "[entered]" in result.output
    assert "slayer run <ref>" in result.output
    assert _FakeListKaggle.calls[-1]["category"] is None


def test_ctrl_c_at_submission_prompt_becomes_keyboard_interrupt(monkeypatch) -> None:
    """Aborting the y/n prompt pauses the run (KeyboardInterrupt), not fails it."""
    import pytest

    def abort(*args, **kwargs):
        """Simulate Ctrl+C at the confirm prompt."""
        raise cli_main.typer.Abort()

    monkeypatch.setattr(cli_main.typer, "confirm", abort)
    with pytest.raises(KeyboardInterrupt):
        cli_main._confirm_submission("Submit?")
