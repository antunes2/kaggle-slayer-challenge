import json

import numpy as np
import pandas as pd

from slayer.config import DefaultsConfig
from slayer.cv.harness import _fit_frames
from slayer.cv.harness import main as harness_main
from slayer.kaggle import CompetitionFileInfo
from slayer.loop import RunContext
from slayer.loop.external import (
    dataset_refs_from_digest,
    external_enabled,
    external_summary_md,
    load_manifest,
    prepare_external_data,
)
from slayer.workspace import EventType, RunStore
from tests.unit.test_harness import _setup as harness_setup

COMP = "playground-x"
DIGEST = """# digest
[EXTERNAL] orig = pd.read_csv('/kaggle/input/datasets/alice/orig-data/data.csv')
[EXTERNAL] orig2 = pd.read_csv('/kaggle/input/datasets/alice/orig-data/data.csv')
[EXTERNAL] old = pd.read_csv('/kaggle/input/legacy-set/train.csv')
[EXTERNAL] dup = pd.read_csv('/kaggle/input/orig-data/data.csv')
train = pd.read_csv('/kaggle/input/datasets/bob/not-tagged/x.csv')
[EXTERNAL] t = pd.read_csv('/kaggle/input/playground-x/train.csv')
"""


def test_external_policy_auto_is_playground_only() -> None:
    """auto → Playground only; on/off override the category."""
    assert external_enabled("auto", "Playground") is True
    assert external_enabled("auto", "Featured") is False
    assert external_enabled("on", "Featured") is True
    assert external_enabled("off", "Playground") is False


def test_dataset_refs_from_digest() -> None:
    """Owner refs ranked by mentions; bare slugs need search; dupes and the comp dropped."""
    assert dataset_refs_from_digest(DIGEST, COMP) == ["alice/orig-data", "?/legacy-set"]


class FakeDatasetKaggle:
    """Fake kaggle client serving dataset listing/search/download."""

    def __init__(self, sizes: dict[str, int]) -> None:
        self.sizes = sizes
        self.downloads: list[str] = []

    def find_dataset(self, slug: str):
        """Resolve bare slugs to a fixed owner."""
        return f"carol/{slug}"

    def dataset_files(self, ref: str):
        """One CSV (and a non-CSV) per dataset with the configured size."""
        return [CompetitionFileInfo(name="data.csv", total_bytes=self.sizes.get(ref, 100)),
                CompetitionFileInfo(name="readme.txt", total_bytes=10)]

    def download_dataset(self, ref: str, target_dir):
        """Write a tiny CSV plus a non-CSV file."""
        self.downloads.append(ref)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "data.csv").write_text("x,Target\n1,Yes\n2,No\n", encoding="utf-8")
        (target_dir / "readme.txt").write_text("hi", encoding="utf-8")
        return target_dir


def _ctx(tmp_path, kaggle, category: str = "Playground", policy: str = "auto") -> RunContext:
    """Run with competition.json of the given category."""
    store = RunStore(tmp_path / "runs")
    run = store.create(COMP, DefaultsConfig(external_data=policy))
    (run.workspace("reports") / "competition.json").write_text(
        json.dumps({"ref": COMP, "category": category}), encoding="utf-8"
    )
    return RunContext(run=run, journal=store.journal(run), registry=store.registry(run),
                      kaggle=kaggle, store=store)


def test_prepare_downloads_csvs_writes_manifest_and_is_idempotent(tmp_path) -> None:
    """Both datasets land as CSV-only folders with a manifest; a rerun reuses it."""
    kaggle = FakeDatasetKaggle({})
    ctx = _ctx(tmp_path, kaggle)
    manifest = prepare_external_data(ctx, DIGEST)
    assert [d["ref"] for d in manifest["datasets"]] == ["alice/orig-data", "carol/legacy-set"]
    entry = manifest["datasets"][0]["files"][0]
    assert entry["key"] == "data" and entry["rows"] == 2 and entry["columns"] == ["x", "Target"]
    assert manifest["datasets"][1]["files"][0]["key"] == "legacy-set__data"
    assert not list(ctx.run.workspace("external").rglob("*.txt"))
    assert ctx.journal.read(type=EventType.EXTERNAL_DATA)
    assert prepare_external_data(ctx, DIGEST) == load_manifest(ctx)
    assert len(kaggle.downloads) == 2


def test_prepare_respects_policy_and_size_cap(tmp_path) -> None:
    """Non-Playground auto → nothing; oversized datasets are skipped, not downloaded."""
    assert prepare_external_data(_ctx(tmp_path / "a", FakeDatasetKaggle({}), "Featured"),
                                 DIGEST) is None
    huge = FakeDatasetKaggle({"alice/orig-data": 10**10, "carol/legacy-set": 10**10})
    ctx = _ctx(tmp_path / "b", huge)
    assert prepare_external_data(ctx, DIGEST) is None
    assert huge.downloads == []
    assert ctx.journal.read(type=EventType.EXTERNAL_DATA)[-1].data["errors"]


def test_summary_lists_keys_rows_and_target_presence() -> None:
    """The prompt summary names each key, size, and whether the target is present."""
    manifest = {"datasets": [{"ref": "alice/orig-data", "files": [
        {"key": "data", "path": "/x", "rows": 1234, "columns": ["x", "Target"]}]}]}
    text = external_summary_md(manifest, "Target")
    assert 'external["data"]' in text and "1,234 rows" in text and "present" in text
    assert external_summary_md(None) == ""


def test_fit_frames_never_include_validation_rows() -> None:
    """External rows join the training slice only."""
    X = pd.DataFrame({"a": range(6)})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    X_ext, y_ext = pd.DataFrame({"a": [100, 101]}), pd.Series([1, 0])
    X_fit, y_fit = _fit_frames(X, y, np.array([0, 1, 2]), X_ext, y_ext)
    assert X_fit["a"].tolist() == [0, 1, 2, 100, 101]
    assert len(y_fit) == 5


EXTERNAL_PIPELINE = '''
from sklearn.linear_model import LogisticRegression

def make_features(df, external):
    df = df.drop(columns=["id"], errors="ignore").copy()
    df["orig_rate"] = float((external["orig"]["Target"] == "Yes").mean())
    return df

def external_rows(external):
    rows = external["orig"].copy()
    rows["Target"] = rows["Target"].map({"Yes": "Yes", "No": "No"})
    return rows

def build_model():
    return LogisticRegression()
'''


def _with_external(tmp_path, orig_csv: str) -> None:
    """Attach an external manifest with one 'orig' frame to the harness config."""
    ext = tmp_path / "orig.csv"
    ext.write_text(orig_csv, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"datasets": [{"ref": "a/b", "files": [
        {"key": "orig", "path": str(ext), "rows": 4, "columns": ["x", "y", "Target"]}]}]}))
    config_path = tmp_path / "cand" / "cv_config.json"
    config = json.loads(config_path.read_text())
    config["external_manifest"] = str(manifest)
    config_path.write_text(json.dumps(config))


def test_harness_uses_external_hooks_and_keeps_oof_competition_only(tmp_path) -> None:
    """Both hooks run; external rows are counted; OOF covers competition rows only."""
    harness_setup(tmp_path, EXTERNAL_PIPELINE)
    _with_external(tmp_path, "x,y,Target\n1,2,Yes\n4,1,No\n2,2,Yes\n3,1,No\n")
    assert harness_main(str(tmp_path / "cand")) == 0
    result = json.loads((tmp_path / "cand" / "cv_results.json").read_text())
    assert result["status"] == "ok" and result["external_rows"] == 4
    train = pd.read_csv(tmp_path / "data" / "train.csv")
    assert len(np.load(tmp_path / "cand" / "oof.npy")) == len(train)


def test_harness_rejects_external_labels_outside_competition_labels(tmp_path) -> None:
    """'satisfied' is not a competition label → loud, explained failure."""
    harness_setup(tmp_path, EXTERNAL_PIPELINE.replace(
        'rows["Target"].map({"Yes": "Yes", "No": "No"})', '"satisfied"'))
    _with_external(tmp_path, "x,y,Target\n1,2,Yes\n4,1,No\n")
    assert harness_main(str(tmp_path / "cand")) == 1
    result = json.loads((tmp_path / "cand" / "cv_results.json").read_text())
    assert "not competition labels" in result["error"]
