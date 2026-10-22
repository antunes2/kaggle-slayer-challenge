
from slayer.config import DefaultsConfig
from slayer.cv import CVProtocol, run_candidate
from slayer.workspace import RunStore

GOOD_PIPELINE = '''
import time
from sklearn.linear_model import LogisticRegression

def build_model():
    return LogisticRegression()
'''

SLEEP_PIPELINE = '''
import time

def build_model():
    time.sleep(60)
    return None
'''

DATA = {
    "train.csv": "id,x,Target\n0,1,Yes\n1,2,Yes\n2,3,No\n3,4,No\n4,1.5,Yes\n5,3.5,No\n",
    "test.csv": "id,x\n6,2.2\n",
    "sample_submission.csv": "id,Target\n6,0.5\n",
}


def _make_run(tmp_path, pipeline_code: str):
    """Create a real run with synthetic data and a candidate pipeline."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    data_dir = run.workspace("data")
    for name, content in DATA.items():
        (data_dir / name).write_text(content, encoding="utf-8")
    candidate_dir = run.workspace("models") / "cand-0001"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "pipeline.py").write_text(pipeline_code, encoding="utf-8")
    return run


def test_runner_subprocess_happy_path(tmp_path):
    """Full subprocess round trip returns a scored CVResult with artifacts."""
    run = _make_run(tmp_path, GOOD_PIPELINE)
    protocol = CVProtocol(metric="Roc Auc Score", k_folds=2, seed=42)

    result = run_candidate(run, "cand-0001", protocol, "binary_classification", "Target")

    assert result.status == "ok"
    assert result.candidate_id == "cand-0001"
    assert len(result.folds) == 2
    assert result.train_rows == 6
    assert (run.workspace("models") / "cand-0001" / "submission.csv").is_file()
    assert (run.workspace("models") / "cand-0001" / "run.log").is_file()
    assert result.pipeline_sha256


def test_runner_missing_pipeline_raises(tmp_path):
    """Missing pipeline.py fails before spawning anything."""
    store = RunStore(tmp_path / "runs")
    run = store.create("comp-x", DefaultsConfig())
    try:
        run_candidate(run, "cand-0009", CVProtocol(metric="AUC"), "binary", "T")
        raise AssertionError("expected CandidateRunError")
    except Exception as exc:
        assert "pipeline missing" in str(exc)


def test_runner_timeout_kills_and_reports(tmp_path):
    """A hanging pipeline is killed and reported as a timeout result."""
    run = _make_run(tmp_path, SLEEP_PIPELINE)
    protocol = CVProtocol(metric="Roc Auc Score", k_folds=2)

    result = run_candidate(run, "cand-0001", protocol, "binary_classification", "Target",
                           timeout_s=10)

    assert result.status == "timeout"
    assert "timeout" in result.error


def test_runner_error_pipeline_returns_error_result(tmp_path):
    """A crashing pipeline yields status=error with the traceback surfaced."""
    run = _make_run(tmp_path, "def build_model():\n    raise RuntimeError('nope')\n")
    protocol = CVProtocol(metric="Roc Auc Score", k_folds=2)

    result = run_candidate(run, "cand-0001", protocol, "binary_classification", "Target")

    assert result.status == "error"
    assert "RuntimeError" in (result.error or "")


def test_ctrl_c_during_cv_kills_the_harness(tmp_path, monkeypatch) -> None:
    """The detached harness is killed on Ctrl+C instead of being orphaned."""
    import subprocess

    import pytest

    from slayer.config import DefaultsConfig
    from slayer.cv import runner as runner_module
    from slayer.schemas import CVProtocol
    from slayer.workspace import RunStore

    run = RunStore(tmp_path / "runs").create("comp-x", DefaultsConfig())
    folder = run.workspace("models") / "cand-0001"
    folder.mkdir(parents=True)
    (folder / "pipeline.py").write_text("def build_model():\n    pass\n", encoding="utf-8")
    killed = []

    class InterruptedProcess:
        pid = 0

        def wait(self, timeout=None):
            """Simulate Ctrl+C while waiting for the harness."""
            raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: InterruptedProcess())
    monkeypatch.setattr(runner_module, "_kill_process_group", killed.append)

    with pytest.raises(KeyboardInterrupt):
        runner_module.run_candidate(
            run, "cand-0001", CVProtocol(metric="Roc Auc Score"), "binary_classification", "y"
        )
    assert len(killed) == 1
