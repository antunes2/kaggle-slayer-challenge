from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    roc_auc_score,
)


@dataclass(frozen=True)
class ResolvedMetric:
    """A metric resolved from its Kaggle display name."""

    key: str
    scorer: Callable
    higher_is_better: bool
    needs_proba: bool


class MetricError(Exception):
    """Raised when a competition metric is not in the registry."""

    pass


def _rmse(y_true, y_pred):
    """Root mean squared error."""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def _rmsle(y_true, y_pred):
    """Root mean squared log error."""
    return math.sqrt(mean_squared_log_error(y_true, y_pred))


def _macro_f1(y_true, y_pred):
    """Macro-averaged F1."""
    return f1_score(y_true, y_pred, average="macro")


_REGISTRY: dict[str, ResolvedMetric] = {}
for _key, _metric in {
    "roc_auc": ResolvedMetric("roc_auc", roc_auc_score, True, True),
    "accuracy": ResolvedMetric("accuracy", accuracy_score, True, False),
    "balanced_accuracy": ResolvedMetric(
        "balanced_accuracy", balanced_accuracy_score, True, False
    ),
    "macro_f1": ResolvedMetric("macro_f1", _macro_f1, True, False),
    "rmse": ResolvedMetric("rmse", _rmse, False, False),
    "mse": ResolvedMetric("mse", mean_squared_error, False, False),
    "msle": ResolvedMetric("msle", mean_squared_log_error, False, False),
    "rmsle": ResolvedMetric("rmsle", _rmsle, False, False),
    "mae": ResolvedMetric("mae", mean_absolute_error, False, False),
    "logloss": ResolvedMetric("logloss", log_loss, False, True),
}.items():
    _REGISTRY[_key] = _metric

_ALIAS_TO_KEY = {
    "roc_auc": "roc_auc",
    "rocauc": "roc_auc",
    "auc": "roc_auc",
    "area_under_curve": "roc_auc",
    "accuracy": "accuracy",
    "accuracy_score": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "balanced_accuracy_score": "balanced_accuracy",
    "macro_f1": "macro_f1",
    "f1": "macro_f1",
    "f1_macro": "macro_f1",
    "rmse": "rmse",
    "root_mean_squared_error": "rmse",
    "root_mean_squared_log_error": "rmsle",
    "rmsle": "rmsle",
    "mse": "mse",
    "mean_squared_error": "mse",
    "msle": "msle",
    "mean_squared_log_error": "msle",
    "mae": "mae",
    "mean_absolute_error": "mae",
    "logloss": "logloss",
    "log_loss": "logloss",
}


def normalize_metric_name(raw: str) -> str:
    """Normalize a Kaggle metric display name to a registry key."""
    text = str(raw or "").lower().strip()
    text = text.replace("-", "_").replace(" ", "_").replace("_score", "")
    text = text.replace("root_mean_squared_error", "rmse")
    text = text.replace("root_mean_squared_log_error", "rmsle")
    return text


def resolve_metric(raw: str) -> ResolvedMetric:
    """Resolve a raw Kaggle metric name to its scorer, direction, and proba need."""
    key = normalize_metric_name(raw)
    if key in _REGISTRY:
        return _REGISTRY[key]
    alias = _ALIAS_TO_KEY.get(key)
    if alias is not None:
        return _REGISTRY[alias]
    raise MetricError(
        f"unsupported metric '{raw}' (normalized: '{key}'). "
        f"Supported: {sorted(_REGISTRY)}"
    )
