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


def test_harness_error_pipeline_writes_structured_failure(tmp_path):
    """A broken pipeline still yields a structured error result."""
    _setup(tmp_path, BROKEN_PIPELINE)
    exit_code = main(str(tmp_path / "cand"))

    assert exit_code == 1
    result = json.loads((tmp_path / "cand" / "cv_results.json").read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert "ValueError" in result["error"]
