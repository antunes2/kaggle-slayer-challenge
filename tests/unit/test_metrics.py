import pytest

from slayer.cv import MetricError, resolve_metric


def test_resolves_kaggle_metric_display_names():
    """Kaggle's display names normalize to registry entries."""
    assert resolve_metric("Roc Auc Score").key == "roc_auc"
    assert resolve_metric("Accuracy Score").key == "accuracy"
    assert resolve_metric("Root Mean Squared Error").key == "rmse"
    assert resolve_metric("Mean Squared Log Error").key == "msle"
    assert resolve_metric("Balanced Accuracy Score").key == "balanced_accuracy"


def test_metric_directions():
    """Direction flags are correct per metric family."""
    assert resolve_metric("Roc Auc Score").higher_is_better
    assert not resolve_metric("Root Mean Squared Error").higher_is_better
    assert not resolve_metric("Mean Absolute Error").higher_is_better
    assert resolve_metric("Accuracy Score").higher_is_better


def test_proba_requirements():
    """AUC-family metrics request probabilities; error metrics do not."""
    assert resolve_metric("Roc Auc Score").needs_proba
    assert not resolve_metric("Accuracy Score").needs_proba
    assert not resolve_metric("RMSE").needs_proba


def test_unknown_metric_raises_with_supported_list():
    """Unknown metrics fail loudly with the list of supported ones."""
    with pytest.raises(MetricError, match="unsupported metric"):
        resolve_metric("Some Exotic Metric 3000")
