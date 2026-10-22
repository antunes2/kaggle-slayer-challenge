"""BLEND step (design 14): deterministic OOF blending of reviewer-approved candidates."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from slayer.cv.blend import hill_climb
from slayer.cv.harness import _find_csv
from slayer.cv.metrics import resolve_metric
from slayer.loop.engine import RunContext
from slayer.loop.phases import _load_plan
from slayer.schemas import CVResult, FoldScore
from slayer.workspace import EventType, ModelCandidate

BLEND_EVALUATED = EventType.BLEND_EVALUATED
POOL_STATUSES = ("approved", "submitted")


def _evaluated_pools(ctx: RunContext) -> set[str]:
    """Pool keys the blender has already evaluated in this run."""
    return {
        str(event.data.get("pool_key"))
        for event in ctx.journal.read(type=BLEND_EVALUATED)
    }


def _pool(ctx: RunContext) -> list:
    """Reviewer-approved, non-blend candidates with complete OOF artifacts."""
    members = []
    for record in ctx.registry.list():
        if record.status not in POOL_STATUSES or record.cv_score is None:
            continue
        if record.model_type == "Blend":
            continue
        folder = ctx.run.workspace("models") / record.id
        if all((folder / name).exists() for name in ("oof.npy", "fold_ids.npy",
                                                      "submission.csv")):
            members.append(record)
    return members


def blend_phase(ctx: RunContext) -> None:
    """Blend the approved pool once per distinct pool; register a winning Blend candidate."""
    plan, competition = _load_plan(ctx)
    metric = resolve_metric(competition.get("evaluation_metric") or "")
    supported = plan.problem_type == "regression" or (
        plan.problem_type == "binary_classification" and metric.needs_proba
    )
    if not supported:
        ctx.say(f"  blend: not supported for {plan.problem_type} / {metric.key} — skipping")
        return
    pool = _pool(ctx)
    if len(pool) < 2:
        ctx.say(f"  blend: needs >= 2 approved candidates with OOF predictions "
                f"(have {len(pool)})")
        return
    pool_key = "blend:" + ",".join(sorted(r.id for r in pool))
    if pool_key in _evaluated_pools(ctx):
        ctx.say("  blend: this candidate pool was already blended — skipping")
        return

    models = ctx.run.workspace("models")
    fold_ids = np.load(models / pool[0].id / "fold_ids.npy")
    oofs, submissions = {}, {}
    for record in pool:
        if not np.array_equal(np.load(models / record.id / "fold_ids.npy"), fold_ids):
            ctx.say(f"  blend: {record.id} used different folds — skipping blend")
            return
        oofs[record.id] = np.load(models / record.id / "oof.npy")
        submissions[record.id] = pd.read_csv(models / record.id / "submission.csv")
    y = pd.read_csv(
        _find_csv(ctx.run.workspace("data"), "train.csv"), usecols=[plan.target_column]
    )[plan.target_column].to_numpy()
    if len(y) != len(fold_ids):
        ctx.say("  blend: OOF length does not match train rows — skipping")
        return

    ctx.say(f"  blend: hill-climbing over {len(pool)} approved candidates")
    result = hill_climb(oofs, y, fold_ids, metric.scorer, metric.higher_is_better)
    sign = 1 if metric.higher_is_better else -1
    gain = sign * (result.mean - result.best_single_mean)
    event = {
        "pool_key": pool_key,
        "weights": result.weights,
        "mean": round(result.mean, 6),
        "best_single": result.best_single,
        "best_single_mean": round(result.best_single_mean, 6),
        "gain": round(gain, 6),
        "candidate_id": None,
    }
    if len(result.weights) < 2 or gain <= 0:
        ctx.journal.append(agent="blender", type=BLEND_EVALUATED, data=event,
                           run=ctx.run.run_id)
        ctx.say(f"  blend: no gain over best single {result.best_single} — not registered")
        return
    duplicate = next(
        (r for r in ctx.registry.list()
         if r.model_type == "Blend" and r.params.get("blend") == result.weights),
        None,
    )
    if duplicate is not None:
        event["candidate_id"] = duplicate.id
        event["duplicate"] = True
        ctx.journal.append(agent="blender", type=BLEND_EVALUATED, data=event,
                           run=ctx.run.run_id)
        ctx.say(f"  blend: same weights as existing {duplicate.id} — not registered again")
        return

    first = submissions[pool[0].id]
    target = plan.target_column
    for frame in submissions.values():
        if not frame.iloc[:, 0].equals(first.iloc[:, 0]):
            ctx.say("  blend: member submissions disagree on row ids — skipping")
            return
    blended_submission = first.copy()
    blended_submission[target] = sum(
        submissions[cid][target].astype(float) * weight
        for cid, weight in result.weights.items()
    )
    blended_oof = sum(oofs[cid] * weight for cid, weight in result.weights.items())

    members = ", ".join(f"{cid}×{w:.2f}" for cid, w in result.weights.items())
    record = ctx.registry.add(
        ModelCandidate(
            model_type="Blend",
            notes=f"OOF hill-climb blend: {members}"[:200],
            params={"blend": result.weights, "blend_pool": pool_key},
        )
    )
    folder = models / record.id
    folder.mkdir(parents=True, exist_ok=True)
    blended_submission.to_csv(folder / "submission.csv", index=False)
    np.save(folder / "oof.npy", blended_oof)
    np.save(folder / "fold_ids.npy", fold_ids)
    (folder / "blend.json").write_text(
        json.dumps({"weights": result.weights, "pool": pool_key}, indent=2), encoding="utf-8"
    )
    (folder / "cv_results.json").write_text(
        CVResult(
            candidate_id=record.id,
            metric=metric.key,
            status="ok",
            folds=[FoldScore(fold=i, score=round(s, 6))
                   for i, s in enumerate(result.fold_scores)],
            mean_score=round(result.mean, 6),
            std_score=round(result.std, 6),
            higher_is_better=metric.higher_is_better,
            train_rows=len(y),
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    ctx.registry.record_cv(record.id, round(result.mean, 6), round(result.std, 6))
    ctx.registry.set_status(record.id, "approved")
    event["candidate_id"] = record.id
    ctx.journal.append(agent="blender", type=BLEND_EVALUATED, data=event, run=ctx.run.run_id)
    ctx.say(
        f"  blend {record.id}: {result.mean:.5f} vs best single {result.best_single} "
        f"{result.best_single_mean:.5f} ({gain:+.5f}) — {members}"
    )
