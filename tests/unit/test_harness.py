import json

import pytest

from slayer.cv.harness import main

GOOD_PIPELINE = '''
from sklearn.linear_model import LogisticRegression

def make_features(df):
    df = df.copy()
    df["x_sq"] = df["x"] ** 2
    return df

def build_model():
    return LogisticRegression()
'''

BROKEN_PIPELINE = '''
def build_model():
    raise ValueError("boom")
'''

UNSEEN_CATEGORY_PIPELINE = '''
from sklearn.linear_model import LogisticRegression

def make_features(df):
    return __import__("pandas").get_dummies(df.drop(columns=["id"]), dtype=float)

def build_model():
    return LogisticRegression()
'''

DATA = {
    "train.csv": "id,x,y,Target\n0,1,2,Yes\n1,2,1,Yes\n2,3,2,No\n3,4,1,No\n"
    "4,1.5,2,Yes\n5,3.5,1,No\n6,2.5,2,Yes\n7,4.5,1,No\n",
    "test.csv": "id,x,y\n8,1.2,2\n9,3.3,1\n",
    "sample_submission.csv": "id,Target\n8,0.5\n9,0.5\n",
}


def _setup(tmp_path, pipeline_code: str) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name, content in DATA.items():
        (data_dir / name).write_text(content, encoding="utf-8")
    candidate_dir = tmp_path / "cand"
    candidate_dir.mkdir()
    (candidate_dir / "pipeline.py").write_text(pipeline_code, encoding="utf-8")
    config = {
        "candidate_id": "cand-0001",
        "data_dir": str(data_dir),
        "target_column": "Target",
        "problem_type": "binary_classification",
        "metric": "Roc Auc Score",
        "k_folds": 2,
        "seed": 42,
        "stratified": True,
    }
    (candidate_dir / "cv_config.json").write_text(json.dumps(config), encoding="utf-8")


def test_harness_happy_path(tmp_path):
    """A good pipeline produces folds, aggregates, hashes, and a submission."""
    _setup(tmp_path, GOOD_PIPELINE)
    exit_code = main(str(tmp_path / "cand"))

    assert exit_code == 0
    result = json.loads((tmp_path / "cand" / "cv_results.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["candidate_id"] == "cand-0001"
    assert len(result["folds"]) == 2
    assert all(0.0 <= f["score"] <= 1.0 for f in result["folds"])
    assert result["mean_score"] == pytest.approx(
        sum(f["score"] for f in result["folds"]) / 2, abs=1e-6
    )
    assert result["pipeline_sha256"]
    assert result["submission_sha256"]

    submission = (tmp_path / "cand" / "submission.csv").read_text(encoding="utf-8")
    lines = submission.strip().splitlines()
    assert lines[0] == "id,Target"
    assert len(lines) == 3


SLOW_PIPELINE = '''
import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SlowModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        time.sleep(3)
        self.classes_ = ["No", "Yes"]
        return self
    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)

def build_model():
    return SlowModel()
'''

SLEEP_FOREVER_PIPELINE = '''
import time

def build_model():
    time.sleep(60)
    return None
'''


def test_harness_error_pipeline_writes_structured_failure(tmp_path):
    """A broken pipeline still yields a structured error result."""
    _setup(tmp_path, BROKEN_PIPELINE)
    exit_code = main(str(tmp_path / "cand"))

    assert exit_code == 1
    result = json.loads((tmp_path / "cand" / "cv_results.json").read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert "ValueError" in result["error"]


def test_harness_probe_fold_aborts_slow_candidates(tmp_path):
    """A fold-0 timing that predicts a budget blowout aborts after one fold."""
    _setup(tmp_path, SLOW_PIPELINE)
    config = json.loads((tmp_path / "cand" / "cv_config.json").read_text(encoding="utf-8"))
    config["budget_s"] = 5
    (tmp_path / "cand" / "cv_config.json").write_text(json.dumps(config), encoding="utf-8")

    exit_code = main(str(tmp_path / "cand"))

    assert exit_code == 1
    result = json.loads((tmp_path / "cand" / "cv_results.json").read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert "probe fold" in result["error"]
    assert len(result["folds"]) == 1
    assert len(result["fold_times"]) == 1
    assert result["fold_times"][0] >= 3.0


def test_harness_self_enforced_deadline_writes_timeout(tmp_path):
    """The harness's own alarm fires for orphan-style overruns with no parent."""
    _setup(tmp_path, SLEEP_FOREVER_PIPELINE)
    config = json.loads((tmp_path / "cand" / "cv_config.json").read_text(encoding="utf-8"))
    config["budget_s"] = 2
    (tmp_path / "cand" / "cv_config.json").write_text(json.dumps(config), encoding="utf-8")

    exit_code = main(str(tmp_path / "cand"))

    assert exit_code == 1
    result = json.loads((tmp_path / "cand" / "cv_results.json").read_text(encoding="utf-8"))
    assert result["status"] == "timeout"
    assert "self-enforced" in result["error"]


def test_harness_ok_result_carries_fold_times(tmp_path):
    """Successful results include per-fold timings."""
    _setup(tmp_path, GOOD_PIPELINE)
    exit_code = main(str(tmp_path / "cand"))

    assert exit_code == 0
    result = json.loads((tmp_path / "cand" / "cv_results.json").read_text(encoding="utf-8"))
    assert len(result["fold_times"]) == 2
    assert all(t >= 0 for t in result["fold_times"])


def test_harness_aligns_test_columns_to_train(tmp_path):
    """Unseen categories in test get zero-filled instead of breaking the pipeline."""
    _setup(tmp_path, UNSEEN_CATEGORY_PIPELINE)
    data_dir = tmp_path / "data"
    (data_dir / "test.csv").write_text("id,x,y,NewCategory\n8,1.2,2,brand-new\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,Target\n8,0.5\n", encoding="utf-8")

    exit_code = main(str(tmp_path / "cand"))

    assert exit_code == 0
    result = json.loads((tmp_path / "cand" / "cv_results.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    submission = (tmp_path / "cand" / "submission.csv").read_text(encoding="utf-8")
    assert submission.startswith("id,Target")



