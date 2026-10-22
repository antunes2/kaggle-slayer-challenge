import json
import zipfile

import pytest

from slayer.config import DefaultsConfig
from slayer.kaggle import CompetitionInfo, KaggleError
from slayer.loop import LoopEngine, Phase, RunContext
from slayer.loop.phases import HANDLERS
from slayer.workspace import EventType, RunStatus, RunStore


class FakeKaggle:
    def __init__(self) -> None:
        self.downloads: list[str] = []

    def list_files(self, competition: str):
        from types import SimpleNamespace

        return [SimpleNamespace(name="train.csv", total_bytes=10)]

    def find_competition(self, ref: str, **kwargs) -> CompetitionInfo:
        return CompetitionInfo(
            ref=ref,
            title="Comp X",
            evaluation_metric="AUC",
            deadline="2026-11-01",
            max_daily_submissions=5,
        )

    def download_files(self, competition: str, target_dir, force: bool = False):
        self.downloads.append(competition)
        zip_path = target_dir / f"{competition}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("train.csv", "id,target\n1,0\n")
            zf.writestr("test.csv", "id\n2\n")
            zf.writestr("sample_submission.csv", "id,target\n2,0.5\n")
        return zip_path


def _make_ctx(tmp_path, kaggle) -> tuple[RunStore, RunContext]:
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    ctx = RunContext(
        run=run,
        journal=store.journal(run),
        registry=store.registry(run),
        kaggle=kaggle,
        store=store,
    )
    return store, ctx


def test_engine_runs_implemented_phases_then_pauses(tmp_path) -> None:
    kaggle = FakeKaggle()
    store, ctx = _make_ctx(tmp_path, kaggle)
    handlers = {
        Phase.INIT: HANDLERS[Phase.INIT],
        Phase.DATA_COLLECTION: HANDLERS[Phase.DATA_COLLECTION],
    }

    final = LoopEngine(ctx, handlers=handlers).start()

    assert final.status == RunStatus.PAUSED
    assert kaggle.downloads == ["comp-x"]

    comp_report = ctx.run.workspace("reports") / "competition.json"
    assert comp_report.is_file()
    assert json.loads(comp_report.read_text())["ref"] == "comp-x"

    assert (ctx.run.workspace("data") / "train.csv").is_file()
    manifest = json.loads((ctx.run.workspace("data") / "manifest.json").read_text())
    assert len(manifest["files"]) == 3
    assert manifest["total_bytes"] > 0

    phases = [e.data["phase"] for e in ctx.journal.read(type=EventType.PHASE_ENTER)]
    assert phases == ["init", "data_collection"]
    assert ctx.journal.read(type=EventType.COMPETITION_CONTEXT)[0].data["ref"] == "comp-x"
    collected = ctx.journal.read(type=EventType.DATA_COLLECTED)
    assert collected[0].data["files"] == 3

    paused = ctx.journal.read(type=EventType.LOOP_PAUSED)
    assert "analysis" in paused[0].data["reason"]

    assert store.get(ctx.run.run_id).status == RunStatus.PAUSED


def test_engine_marks_run_failed_on_phase_error(tmp_path) -> None:
    class BrokenKaggle(FakeKaggle):
        def find_competition(self, ref: str, **kwargs) -> CompetitionInfo:
            raise KaggleError("competition not found")

    store, ctx = _make_ctx(tmp_path, BrokenKaggle())

    with pytest.raises(KaggleError):
        LoopEngine(ctx).start()

    assert store.get(ctx.run.run_id).status == RunStatus.FAILED
    assert ctx.journal.read(type=EventType.LOOP_PAUSED) == []


def test_engine_reports_phase_activity(tmp_path) -> None:
    store, ctx = _make_ctx(tmp_path, FakeKaggle())
    messages: list[str] = []
    ctx.reporter = messages.append
    handlers = {
        Phase.INIT: HANDLERS[Phase.INIT],
        Phase.DATA_COLLECTION: HANDLERS[Phase.DATA_COLLECTION],
    }

    LoopEngine(ctx, handlers=handlers).start()

    assert any("[phase] init" in m for m in messages)
    assert any("[phase] data_collection" in m for m in messages)
    assert any("[pause]" in m for m in messages)


def test_ctrl_c_pauses_the_run_cleanly(tmp_path) -> None:
    """KeyboardInterrupt inside a phase → status paused + journaled reason (Q0003)."""
    store, ctx = _make_ctx(tmp_path, FakeKaggle())

    def interrupted(c):
        """Simulate the user pressing Ctrl+C mid-phase."""
        raise KeyboardInterrupt

    final = LoopEngine(ctx, handlers={Phase.INIT: interrupted}).start()

    assert final.status == RunStatus.PAUSED
    assert store.get(ctx.run.run_id).status == RunStatus.PAUSED
    paused = ctx.journal.read(type=EventType.LOOP_PAUSED)
    assert "Ctrl+C" in paused[-1].data["reason"]
