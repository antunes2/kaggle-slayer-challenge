import zipfile
from types import SimpleNamespace

from slayer.config import DefaultsConfig
from slayer.loop import RunContext
from slayer.loop.data import data_collection_phase
from slayer.workspace import EventType, RunStore


class CountingKaggle:
    """Fake kaggle client that counts downloads and controls the remote file list."""

    def __init__(self) -> None:
        self.downloads = 0
        self.remote_files = [
            SimpleNamespace(name="train.csv", total_bytes=10),
            SimpleNamespace(name="test.csv", total_bytes=20),
        ]

    def list_files(self, competition: str):
        """Report the current remote file list."""
        return list(self.remote_files)

    def download_files(self, competition: str, target_dir, force: bool = False):
        """Simulate downloading the competition zip."""
        self.downloads += 1
        zip_path = target_dir / f"{competition}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("train.csv", "id,target\n1,0\n")
            zf.writestr("test.csv", "id\n2\n")
        return zip_path


def _make_ctx(tmp_path, kaggle, refresh_data: bool = False):
    """Create a store + context for a fresh run on comp-x."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig(refresh_data=refresh_data))
    ctx = RunContext(
        run=run,
        journal=store.journal(run),
        registry=store.registry(run),
        kaggle=kaggle,
        store=store,
    )
    return store, ctx


def test_second_run_reuses_cache(tmp_path) -> None:
    """A second run on the same competition skips the download."""
    kaggle = CountingKaggle()
    _, first = _make_ctx(tmp_path, kaggle)
    data_collection_phase(first)
    assert kaggle.downloads == 1

    _, second = _make_ctx(tmp_path, kaggle)
    data_collection_phase(second)
    assert kaggle.downloads == 1
    assert (second.run.workspace("data") / "train.csv").is_file()
    assert (second.run.workspace("data") / "manifest.json").is_file()

    events = second.journal.read(type=EventType.DATA_COLLECTED)
    assert events[0].data["from_cache"] is True
    first_events = first.journal.read(type=EventType.DATA_COLLECTED)
    assert first_events[0].data["from_cache"] is False


def test_cache_invalidated_when_remote_changes(tmp_path) -> None:
    """A changed remote file list forces a redownload."""
    kaggle = CountingKaggle()
    _, first = _make_ctx(tmp_path, kaggle)
    data_collection_phase(first)

    kaggle.remote_files.append(SimpleNamespace(name="extra.csv", total_bytes=5))
    _, second = _make_ctx(tmp_path, kaggle)
    data_collection_phase(second)
    assert kaggle.downloads == 2


def test_refresh_data_forces_redownload(tmp_path) -> None:
    """--refresh-data bypasses a valid cache."""
    kaggle = CountingKaggle()
    _, first = _make_ctx(tmp_path, kaggle)
    data_collection_phase(first)

    _, second = _make_ctx(tmp_path, kaggle, refresh_data=True)
    data_collection_phase(second)
    assert kaggle.downloads == 2


def test_cache_independence_between_competitions(tmp_path) -> None:
    """Caches are per-competition slugs and do not collide."""
    kaggle = CountingKaggle()
    _, a = _make_ctx(tmp_path, kaggle)
    data_collection_phase(a)
    assert kaggle.downloads == 1

    store = RunStore(tmp_path / "runs")
    run_b = store.create("comp-y", DefaultsConfig())
    ctx_b = RunContext(
        run=run_b, journal=store.journal(run_b), registry=store.registry(run_b),
        kaggle=kaggle, store=store,
    )
    data_collection_phase(ctx_b)
    assert kaggle.downloads == 2
