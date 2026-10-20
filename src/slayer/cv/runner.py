"""Sandbox runner: executes the CV harness in a subprocess for one candidate."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from slayer.schemas.cv import CVProtocol, CVResult
from slayer.workspace import Run


class CandidateRunError(Exception):
    """Raised when a candidate evaluation cannot be started or parsed."""

    pass


def _pipeline_sha(candidate_dir: Path) -> str:
    """Fingerprint the candidate's pipeline.py."""
    return sha256((candidate_dir / "pipeline.py").read_bytes()).hexdigest()


def run_candidate(
    run: Run,
    candidate_id: str,
    protocol: CVProtocol,
    problem_type: str,
    target_column: str,
    timeout_s: int = 600,
) -> CVResult:
    """Run one candidate's pipeline through the CV harness in a sandboxed subprocess."""
    candidate_dir = run.workspace("models") / candidate_id
    pipeline_path = candidate_dir / "pipeline.py"
    if not pipeline_path.exists():
        raise CandidateRunError(f"candidate pipeline missing: {pipeline_path}")

    config = {
        "candidate_id": candidate_id,
        "data_dir": str(run.workspace("data").resolve()),
        "target_column": target_column,
        "problem_type": problem_type,
        "metric": protocol.metric,
        "k_folds": protocol.k_folds,
        "seed": protocol.seed,
        "stratified": protocol.stratified,
    }
    (candidate_dir / "cv_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    log_path = candidate_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-m", "slayer.cv.harness", str(candidate_dir.resolve())],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=candidate_dir,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(process)

    if timed_out:
        return CVResult(
            candidate_id=candidate_id,
            metric=protocol.metric,
            status="timeout",
            error=f"harness exceeded timeout of {timeout_s}s and was killed",
            pipeline_sha256=_pipeline_sha(candidate_dir),
        )

    results_path = candidate_dir / "cv_results.json"
    if not results_path.exists():
        return CVResult(
            candidate_id=candidate_id,
            metric=protocol.metric,
            status="error",
            error=f"harness produced no results; log tail:\n{_log_tail(log_path)}",
            pipeline_sha256=_pipeline_sha(candidate_dir),
        )
    return CVResult.model_validate_json(results_path.read_text(encoding="utf-8"))


def _kill_process_group(process: subprocess.Popen) -> None:
    """Kill the harness subprocess and its entire process group."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def _log_tail(log_path: Path, limit: int = 1500) -> str:
    """Return the last portion of the harness log for error reporting."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(log unavailable)"
    return text[-limit:]
