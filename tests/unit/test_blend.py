import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from slayer.config import DefaultsConfig
from slayer.cv.blend import fold_scores, hill_climb
from slayer.cv.harness import main as harness_main
from slayer.loop import RunContext
from slayer.loop.blending import BLEND_EVALUATED, blend_phase
from slayer.workspace import ModelCandidate, RunStore
from tests.unit.test_harness import GOOD_PIPELINE
from tests.unit.test_harness import _setup as harness_setup

N = 400


def _signals(seed: int = 0):
    """A binary target plus two complementary noisy predictors and one useless one."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    a = y + rng.normal(0, 1.0, N)
    b = y + rng.normal(0, 1.0, N)
    junk = rng.normal(0, 1.0, N)
    return y, a, b, junk, np.arange(N) % 4


# --- harness -----------------------------------------------------------------


def test_harness_persists_aligned_oof(tmp_path) -> None:
    """oof.npy/fold_ids.npy cover every train row and reproduce the fold scores."""
    harness_setup(tmp_path, GOOD_PIPELINE)
    assert harness_main(str(tmp_path / "cand")) == 0
    cand = tmp_path / "cand"
    oof = np.load(cand / "oof.npy")
    fold_ids = np.load(cand / "fold_ids.npy")
    train = pd.read_csv(tmp_path / "data" / "train.csv")
    assert len(oof) == len(train) and not np.isnan(oof).any()
    assert set(fold_ids.tolist()) == {0, 1}
    result = json.loads((cand / "cv_results.json").read_text())
    recomputed = fold_scores(oof, train["Target"].to_numpy(), fold_ids, roc_auc_score)
    assert [round(s, 6) for s in recomputed] == [f["score"] for f in result["folds"]]


# --- hill climb --------------------------------------------------------------


def test_hill_climb_blends_complementary_members() -> None:
    """Two independent noisy views beat either alone; a useless member is not selected."""
    y, a, b, junk, folds = _signals()
    result = hill_climb({"c1": a, "c2": b, "c3": junk}, y, folds, roc_auc_score, True)
    assert set(result.weights) == {"c1", "c2"}
    assert abs(sum(result.weights.values()) - 1.0) < 1e-9
    assert result.mean > result.best_single_mean


def test_hill_climb_is_deterministic_and_direction_aware() -> None:
    """Same inputs give the same weights; lower-is-better metrics are minimized."""
    y, a, b, junk, folds = _signals(1)
    first = hill_climb({"c1": a, "c2": b}, y, folds, roc_auc_score, True)
    second = hill_climb({"c1": a, "c2": b}, y, folds, roc_auc_score, True)
    assert first == second

    def mae(t, p):
        """Mean absolute error."""
        return float(np.mean(np.abs(t - p)))

    target = y.astype(float)
    good, bad = target + 0.1, target + 2.0
    result = hill_climb({"good": good, "bad": bad}, target, folds, mae, False)
    assert result.weights == {"good": 1.0}


# --- blend step --------------------------------------------------------------


def _blend_ctx(tmp_path, metric: str = "Roc Auc Score", members: int = 2) -> RunContext:
    """A run with `members` approved candidates carrying OOF + submission artifacts."""
    y, a, b, _, folds = _signals()
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    labels = np.where(y == 1, "Yes", "No")
    pd.DataFrame({"id": range(N), "x": a, "Target": labels}).to_csv(
        run.workspace("data") / "train.csv", index=False
    )
    reports = run.workspace("reports")
    (reports / "competition.json").write_text(
        json.dumps({"ref": "comp-x", "evaluation_metric": metric}), encoding="utf-8"
    )
    (reports / "strategy_plan.json").write_text(json.dumps({
        "problem_type": "binary_classification", "target_column": "Target",
        "cv": {"k_folds": 4, "seed": 42, "stratified": True},
        "roadmap": [{"title": "s", "description": "d", "model_family": "m"}],
        "summary": "go",
    }), encoding="utf-8")
    registry = store.registry(run)
    for i, oof in enumerate([a, b][:members]):
        record = registry.add(ModelCandidate(model_type="lightgbm"))
        mean = float(np.mean(fold_scores(oof, labels, folds, roc_auc_score)))
        registry.record_cv(record.id, mean, 0.01)
        registry.set_status(record.id, "approved")
        folder = run.workspace("models") / record.id
        folder.mkdir(parents=True)
        np.save(folder / "oof.npy", oof)
        np.save(folder / "fold_ids.npy", folds.astype(np.int8))
        pd.DataFrame({"id": [1000, 1001], "Target": [0.2 + i * 0.2, 0.6]}).to_csv(
            folder / "submission.csv", index=False
        )
    return RunContext(run=run, journal=store.journal(run), registry=registry,
                      kaggle=None, store=store)


def test_blend_registers_approved_blend_with_averaged_submission(tmp_path) -> None:
    """A gaining blend becomes an approved Blend candidate with weighted submission."""
    ctx = _blend_ctx(tmp_path)
    messages: list[str] = []
    ctx.reporter = messages.append

    blend_phase(ctx)

    blends = [r for r in ctx.registry.list() if r.model_type == "Blend"]
    assert len(blends) == 1
    blend = blends[0]
    assert blend.status == "approved"
    assert "roadmap_step" not in blend.params
    weights = blend.params["blend"]
    sub = pd.read_csv(ctx.run.workspace("models") / blend.id / "submission.csv")
    expected = 0.2 * weights["cand-0001"] + 0.4 * weights["cand-0002"]
    assert abs(sub["Target"][0] - expected) < 1e-9
    assert (ctx.run.workspace("models") / blend.id / "oof.npy").exists()
    assert ctx.journal.read(type=BLEND_EVALUATED)[-1].data["candidate_id"] == blend.id

    blend_phase(ctx)
    assert len([r for r in ctx.registry.list() if r.model_type == "Blend"]) == 1
    assert any("already blended" in m for m in messages)


def test_blend_skips_small_pool_and_unsupported_metric(tmp_path) -> None:
    """One member, or a label metric, never produces a blend."""
    ctx = _blend_ctx(tmp_path / "a", members=1)
    blend_phase(ctx)
    assert not ctx.journal.read(type=BLEND_EVALUATED)

    ctx = _blend_ctx(tmp_path / "b", metric="Accuracy")
    messages: list[str] = []
    ctx.reporter = messages.append
    blend_phase(ctx)
    assert any("not supported" in m for m in messages)
    assert all(r.model_type != "Blend" for r in ctx.registry.list())


def test_blend_skips_mismatched_folds(tmp_path) -> None:
    """Members scored on different folds are never blended."""
    ctx = _blend_ctx(tmp_path)
    other = ctx.run.workspace("models") / "cand-0002" / "fold_ids.npy"
    np.save(other, (np.arange(N) % 2).astype(np.int8))
    messages: list[str] = []
    ctx.reporter = messages.append
    blend_phase(ctx)
    assert any("different folds" in m for m in messages)
