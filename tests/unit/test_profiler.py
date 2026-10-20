import pytest

from slayer.analysis import AnalysisError, profile_data
from slayer.config import DefaultsConfig
from slayer.workspace import RunStore

BINARY_TRAIN = "id,Age,City,Target\n0,25,X,1\n1,35,,0\n2,45,X,1\n3,55,Y,0\n"
YESNO_TRAIN = "id,Age,Target\n0,25,Yes\n1,45,No\n2,65,Yes\n3,85,No\n"
BINARY_TEST = "id,Age,City\n5,60,Z\n"
BINARY_SUB = "id,Target\n5,0.5\n"

REGRESSION_TRAIN = "id,Age,Target\n0,20,1.5\n1,30,2.5\n2,40,3.5\n3,50,10.5\n"
MULTICLASS_TRAIN = "id,Size,Target\n0,1,a\n1,2,b\n2,3,c\n3,4,a\n"


def _make_run(tmp_path, train: str, test: str, submission: str) -> None:
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    data_dir = run.workspace("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(train, encoding="utf-8")
    (data_dir / "test.csv").write_text(test, encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text(submission, encoding="utf-8")
    return run


def test_profile_binary_classification(tmp_path) -> None:
    run = _make_run(tmp_path, BINARY_TRAIN, BINARY_TEST, BINARY_SUB)
    profile = profile_data(run)

    assert profile.target_column == "Target"
    assert profile.problem_type_guess == "binary_classification"
    assert profile.train_rows == 4
    assert profile.train_cols == 4
    assert profile.test_rows == 1

    city = next(c for c in profile.columns if c.name == "City")
    assert city.missing_pct == 25.0
    assert not city.numeric
    assert "X" in city.facts

    age = next(c for c in profile.columns if c.name == "Age")
    assert age.numeric
    assert "mean" in age.facts

    corr_names = [name for name, _ in profile.top_correlations]
    assert "Age" in corr_names
    assert "1" in profile.target_summary
    assert "0" in profile.target_summary

    assert (run.workspace("eda") / "profile.json").is_file()
    profile_md = (run.workspace("eda") / "profile.md").read_text(encoding="utf-8")
    assert "binary_classification" in profile_md
    assert "Correlations with target" in profile_md


def test_profile_string_binary_target_gets_correlations(tmp_path) -> None:
    run = _make_run(tmp_path, YESNO_TRAIN, BINARY_TEST, BINARY_SUB)
    profile = profile_data(run)
    assert profile.problem_type_guess == "binary_classification"
    corr_names = [name for name, _ in profile.top_correlations]
    assert "Age" in corr_names


def test_profile_regression_guess(tmp_path) -> None:
    run = _make_run(tmp_path, REGRESSION_TRAIN, BINARY_TEST, BINARY_SUB)
    profile = profile_data(run)
    assert profile.problem_type_guess == "regression"
    assert "mean" in profile.target_summary


def test_profile_multiclass_guess(tmp_path) -> None:
    run = _make_run(tmp_path, MULTICLASS_TRAIN, BINARY_TEST, BINARY_SUB)
    profile = profile_data(run)
    assert profile.problem_type_guess == "multiclass_classification"
    assert "a" in profile.target_summary


def test_profile_missing_submission_raises(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    (run.workspace("data")).mkdir(parents=True, exist_ok=True)
    (run.workspace("data") / "train.csv").write_text(BINARY_TRAIN, encoding="utf-8")
    with pytest.raises(AnalysisError, match="sample_submission"):
        profile_data(run)
