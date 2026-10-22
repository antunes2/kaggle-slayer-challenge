"""Deterministic OOF blender: greedy forward selection with replacement (Caruana 2004)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

MAX_ADDITIONS = 15
MIN_GAIN = 1e-7


@dataclass(frozen=True)
class BlendResult:
    """Selected weights and the blend's per-fold scores vs the best single member."""

    weights: dict[str, float]
    fold_scores: list[float]
    mean: float
    std: float
    best_single: str
    best_single_mean: float


def fold_scores(
    pred: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, scorer: Callable
) -> list[float]:
    """Metric per CV fold, using the harness's fold assignment."""
    return [
        float(scorer(y[fold_ids == fold], pred[fold_ids == fold]))
        for fold in sorted(set(fold_ids.tolist()))
    ]


def _mean(scores: list[float]) -> float:
    """Arithmetic mean."""
    return sum(scores) / len(scores)


def hill_climb(
    oofs: Mapping[str, np.ndarray],
    y: np.ndarray,
    fold_ids: np.ndarray,
    scorer: Callable,
    higher_is_better: bool,
    max_additions: int = MAX_ADDITIONS,
) -> BlendResult:
    """Pick blend weights that maximize (or minimize) the mean per-fold metric."""
    if not oofs:
        raise ValueError("hill_climb needs at least one member")
    ids = sorted(oofs)
    sign = 1.0 if higher_is_better else -1.0
    singles = {cid: _mean(fold_scores(oofs[cid], y, fold_ids, scorer)) for cid in ids}
    best_single = max(ids, key=lambda cid: (sign * singles[cid], -ids.index(cid)))

    counts = {best_single: 1}
    total = oofs[best_single].astype(np.float64).copy()
    current = singles[best_single]
    for _ in range(max_additions):
        n = sum(counts.values())
        best_id, best_score = None, current
        for cid in ids:
            score = _mean(fold_scores((total + oofs[cid]) / (n + 1), y, fold_ids, scorer))
            if sign * (score - best_score) > MIN_GAIN:
                best_id, best_score = cid, score
        if best_id is None:
            break
        counts[best_id] = counts.get(best_id, 0) + 1
        total += oofs[best_id]
        current = best_score

    n = sum(counts.values())
    weights = {cid: counts[cid] / n for cid in sorted(counts)}
    blended = sum(oofs[cid] * w for cid, w in weights.items())
    scores = fold_scores(blended, y, fold_ids, scorer)
    mean = _mean(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    return BlendResult(
        weights=weights,
        fold_scores=scores,
        mean=mean,
        std=std,
        best_single=best_single,
        best_single_mean=singles[best_single],
    )
