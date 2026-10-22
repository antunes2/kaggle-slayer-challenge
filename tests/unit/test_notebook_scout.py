import json
import os
import time

import pytest

from slayer.analysis.notebooks import MAX_TOTAL_CHARS, build_digest, extract_signal_lines
from slayer.config import DefaultsConfig
from slayer.kaggle import KaggleClient, KaggleError, NotebookInfo
from slayer.kaggle.client import _code_from_source
from slayer.loop import RunContext
from slayer.loop.phases import notebook_digest
from slayer.workspace import EventType, RunStore

COMP = "playground-x"

CODE = """
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/competitions/playground-x/train.csv')
orig = pd.read_csv('/kaggle/input/datasets/someone/original-data/data.csv')
# extract digits
train['inc_d2'] = (train['income'] // 100 % 10).astype('int8')
train['inc_d2'] = (train['income'] // 100 % 10).astype('int8')
plt.figure(figsize=(8, 4))
print(train.shape)
train['inc_cat'] = train['income'].astype(str)
model = LGBMClassifier(n_estimators=5000, learning_rate=0.01)
x = 1 + 1
"""


class FakeScoutKaggle:
    """Fake kaggle exposing notebook listing + source."""

    def __init__(self, sources: dict[str, str], fail_list: bool = False) -> None:
        self.sources = sources
        self.fail_list = fail_list
        self.list_calls = 0

    def list_notebooks(self, competition: str, n: int = 5):
        """Canned listing in insertion order."""
        self.list_calls += 1
        if self.fail_list:
            raise KaggleError("notebook listing failed: 503")
        return [NotebookInfo(ref=ref, title=f"T {ref}", votes=10) for ref in self.sources][:n]

    def notebook_code(self, ref: str) -> str:
        """Canned code (raises for refs mapped to None)."""
        if self.sources[ref] is None:
            raise KaggleError("notebook fetch failed")
        return self.sources[ref]


def test_extract_keeps_modelling_lines_and_tags_external() -> None:
    """Feature/model lines survive; plots/prints/imports/dupes drop; external data is tagged."""
    lines = extract_signal_lines(CODE, COMP)
    assert "train['inc_d2'] = (train['income'] // 100 % 10).astype('int8')" in lines
    assert lines.count("train['inc_d2'] = (train['income'] // 100 % 10).astype('int8')") == 1
    assert "train['inc_cat'] = train['income'].astype(str)" in lines
    assert any("n_estimators=5000" in line for line in lines)
    assert not any("plt." in line or "print(" in line or "import" in line for line in lines)
    assert not any(line == "x = 1 + 1" for line in lines)
    external = [line for line in lines if line.startswith("[EXTERNAL]")]
    assert external and all("orig" in line for line in external)


def test_digest_renders_untrusted_header_and_caps_size() -> None:
    """The digest frames notebooks as untrusted evidence and respects the char cap."""
    big = "\n".join(f"df['c{i}'] = df['a'].astype(str) + '{'x' * 120}'" for i in range(500))
    kaggle = FakeScoutKaggle({f"u/nb{i}": big for i in range(10)})
    digest = build_digest(kaggle, COMP)
    assert "UNTRUSTED reference material" in digest.markdown
    assert len(digest.markdown) <= MAX_TOTAL_CHARS
    assert 1 <= len(digest.refs) < 10


def test_digest_is_fail_soft() -> None:
    """Listing failures and per-notebook failures never raise."""
    down = build_digest(FakeScoutKaggle({}, fail_list=True), COMP)
    assert not down.available and "unavailable" in down.markdown
    partial = build_digest(FakeScoutKaggle({"u/bad": None, "u/good": CODE}), COMP)
    assert partial.refs == ["u/good"]
    assert partial.errors and "u/bad" in partial.errors[0]


def test_code_from_source_handles_ipynb_and_scripts() -> None:
    """Notebook JSON yields code cells only; plain scripts pass through."""
    nb = json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["# title"]},
        {"cell_type": "code", "source": ["a = 1\n", "b = 2"]},
    ]})
    assert _code_from_source(nb) == "a = 1\nb = 2"
    assert _code_from_source("x = 3") == "x = 3"


def test_client_list_notebooks_wraps_sdk(monkeypatch) -> None:
    """kernels_list results convert to NotebookInfo; SDK errors become KaggleError."""
    class Kernel:
        def __init__(self, ref, votes):
            self.ref, self.title, self.total_votes = ref, "t", votes

    class Api:
        def kernels_list(self, **kwargs):
            assert kwargs["competition"] == COMP and kwargs["sort_by"] == "voteCount"
            return [Kernel("a/one", 5), None, Kernel("b/two", 3)]

    infos = KaggleClient(Api()).list_notebooks(COMP, n=5)
    assert [i.ref for i in infos] == ["a/one", "b/two"]

    class Broken:
        def kernels_list(self, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(KaggleError, match="notebook listing failed"):
        KaggleClient(Broken()).list_notebooks(COMP)


def _ctx(tmp_path, kaggle) -> RunContext:
    """A bare run context with the given kaggle client."""
    store = RunStore(tmp_path / "runs")
    run = store.create(COMP, DefaultsConfig())
    return RunContext(run=run, journal=store.journal(run), registry=store.registry(run),
                      kaggle=kaggle, store=store)


def test_notebook_digest_caches_and_refreshes_when_stale(tmp_path) -> None:
    """The digest is written once, reused while fresh, rebuilt when stale; journaled."""
    kaggle = FakeScoutKaggle({"u/good": CODE})
    ctx = _ctx(tmp_path, kaggle)
    first = notebook_digest(ctx)
    assert "u/good" in first
    assert notebook_digest(ctx) == first and kaggle.list_calls == 1
    path = ctx.run.workspace("reports") / "notebook_digest.md"
    old = time.time() - 13 * 3600
    os.utime(path, (old, old))
    notebook_digest(ctx)
    assert kaggle.list_calls == 2
    assert len(ctx.journal.read(type=EventType.SCOUT_DIGEST)) == 2


def test_notebook_digest_without_scout_capable_client(tmp_path) -> None:
    """No kaggle client (or an old fake) → empty digest, no error."""
    assert notebook_digest(_ctx(tmp_path / "a", None)) == ""
    assert notebook_digest(_ctx(tmp_path / "b", object())) == ""
