"""CV harness entry point: runs inside the sandbox subprocess for one candidate."""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
import time
import traceback
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from slayer.cv.metrics import ResolvedMetric, resolve_metric
from slayer.schemas.cv import CVResult, FoldScore

PROBE_OVERHEAD_FACTOR = 1.25


def _sha256_file(path: Path) -> str:
    """Compute the sha256 hex digest of a file."""
    return sha256(path.read_bytes()).hexdigest()


def _load_pipeline(pipeline_path: Path):
    """Import the candidate's pipeline.py from an explicit file path."""
    spec = importlib.util.spec_from_file_location("candidate_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_csv(data_dir: Path, name: str) -> Path:
    """Locate a competition CSV by exact name with a glob fallback."""
    exact = data_dir / name
    if exact.exists():
        return exact
    matches = sorted(data_dir.glob(name.replace(".csv", "*.csv")))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"missing {name} in {data_dir}")


def _positive_class(y: pd.Series) -> object:
    """Pick the positive class for binary targets (lexicographically last)."""
    return sorted(y.dropna().astype(str).unique())[-1]


def _fold_predictions(model, X_val, y_val, metric: ResolvedMetric, problem_type: str):
    """Predictions in the form the metric consumes (positive-class proba for binary AUC)."""
    if metric.needs_proba and problem_type == "binary_classification":
        if hasattr(model, "predict_proba"):
            classes = [str(c) for c in model.classes_]
            positive = _positive_class(y_val)
            return model.predict_proba(X_val)[:, classes.index(str(positive))]
    if metric.needs_proba and hasattr(model, "predict_proba"):
        return model.predict_proba(X_val)
    return model.predict(X_val)


def _score_fold(
    model, X_val, y_val, metric: ResolvedMetric, problem_type: str
) -> tuple[float, np.ndarray]:
    """Score one fold with the resolved metric; also return the predictions used."""
    preds = np.asarray(_fold_predictions(model, X_val, y_val, metric, problem_type))
    return float(metric.scorer(y_val, preds)), preds


def _blendable(preds: np.ndarray) -> bool:
    """Whether fold predictions can be averaged across candidates (1-D numeric)."""
    return preds.ndim == 1 and np.issubdtype(preds.dtype, np.number)


def _predict_submission(
    model, test: pd.DataFrame, target_column: str, submission: pd.DataFrame, problem_type: str
) -> pd.DataFrame:
    """Produce the submission frame in the exact sample_submission shape."""
    out = submission.copy()
    if problem_type == "binary_classification" and hasattr(model, "predict_proba"):
        classes = [str(c) for c in model.classes_]
        positive = sorted(classes)[-1]
        out[target_column] = model.predict_proba(test)[:, classes.index(positive)]
    else:
        out[target_column] = model.predict(test)
    return out


def _install_deadline(budget_s: float) -> None:
    """Set a best-effort self-enforced deadline (orphan-safe timeout, Q0003)."""

    def handler(signum, frame):
        """Raise on the alarm so the harness can persist a timeout result."""
        raise TimeoutError(
            f"harness timeout: self-enforced budget of {budget_s}s exceeded"
        )

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, budget_s)


class _ProbeTimeout(Exception):
    """Raised when fold 0's timing predicts the total budget will be blown."""

    pass


def _execute(config: dict, candidate_dir: Path) -> CVResult:
    """Run the full CV + final-fit + submission flow for one candidate."""
    started = time.time()
    pipeline_path = candidate_dir / "pipeline.py"
    data_dir = Path(config["data_dir"])
    target_column = config["target_column"]
    problem_type = config["problem_type"]
    metric = resolve_metric(config["metric"])
    k_folds = int(config["k_folds"])
    budget_s = float(config["budget_s"]) if config.get("budget_s") else None
    candidate_id = config["candidate_id"]

    if budget_s:
        _install_deadline(budget_s)

    folds: list[FoldScore] = []
    fold_times: list[float] = []

    def persist(result: CVResult) -> None:
        """Write the current (possibly partial) CVResult to cv_results.json."""
        (candidate_dir / "cv_results.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )

    def failure(message: str, status: str = "error") -> CVResult:
        """Build a failure result carrying whatever partial folds exist."""
        return CVResult(
            candidate_id=candidate_id,
            metric=metric.key,
            status=status,
            folds=folds,
            fold_times=fold_times,
            error=message,
            pipeline_sha256=_sha256_file(pipeline_path),
        )

    try:
        train = pd.read_csv(_find_csv(data_dir, "train.csv"))
        test = pd.read_csv(_find_csv(data_dir, "test.csv"))
        submission = pd.read_csv(_find_csv(data_dir, "sample_submission.csv"))

        pipeline = _load_pipeline(pipeline_path)
        y = train[target_column]
        X = train.drop(columns=[target_column])
        make_features = getattr(pipeline, "make_features", None)
        if make_features is not None:
            X = make_features(X)
            test = make_features(test)
        test = test.reindex(columns=X.columns, fill_value=0)

        splitter_cls = StratifiedKFold if config.get("stratified", True) else KFold
        splitter = splitter_cls(
            n_splits=k_folds, shuffle=True, random_state=int(config["seed"])
        )
        oof = np.full(len(X), np.nan, dtype=np.float64)
        fold_ids = np.full(len(X), -1, dtype=np.int8)
        blendable = True
        for i, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            fold_start = time.time()
            model = pipeline.build_model()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            score, preds = _score_fold(
                model, X.iloc[val_idx], y.iloc[val_idx], metric, problem_type
            )
            fold_ids[val_idx] = i
            if blendable and _blendable(preds):
                oof[val_idx] = preds
            else:
                blendable = False
            fold_time = time.time() - fold_start
            folds.append(FoldScore(fold=i, score=round(score, 6)))
            fold_times.append(round(fold_time, 2))
            persist(
                CVResult(
                    candidate_id=candidate_id,
                    metric=metric.key,
                    status="running",
                    folds=folds,
                    fold_times=fold_times,
                    higher_is_better=metric.higher_is_better,
                    pipeline_sha256=_sha256_file(pipeline_path),
                )
            )
            if budget_s and i == 0:
                estimated_total = fold_time * (k_folds + 1)
                if estimated_total > budget_s * PROBE_OVERHEAD_FACTOR:
                    raise _ProbeTimeout(
                        f"probe fold took {fold_time:.1f}s; estimated total "
                        f"~{estimated_total:.0f}s exceeds the {budget_s:.0f}s budget — "
                        "reduce model/preprocessing cost"
                    )

        scores = [f.score for f in folds]
        mean = round(sum(scores) / len(scores), 6)
        std = round((sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5, 6)

        final_model = pipeline.build_model()
        final_model.fit(X, y)
        out_frame = _predict_submission(
            final_model, test, target_column, submission, problem_type
        )
        submission_path = candidate_dir / "submission.csv"
        out_frame.to_csv(submission_path, index=False)
        if blendable:
            np.save(candidate_dir / "oof.npy", oof)
            np.save(candidate_dir / "fold_ids.npy", fold_ids)

        result = CVResult(
            candidate_id=candidate_id,
            metric=metric.key,
            status="ok",
            folds=folds,
            fold_times=fold_times,
            mean_score=mean,
            std_score=std,
            higher_is_better=metric.higher_is_better,
            train_rows=len(train),
            elapsed_s=round(time.time() - started, 2),
            pipeline_sha256=_sha256_file(pipeline_path),
            submission_sha256=_sha256_file(submission_path),
        )
        persist(result)
        return result
    except TimeoutError as exc:
        persist(failure(str(exc), status="timeout"))
        raise
    except Exception as exc:
        persist(
            failure(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}")
        )
        raise


def main(candidate_dir: str) -> int:
    """Entry point for `python -m slayer.cv.harness <candidate_dir>`."""
    candidate_dir_path = Path(candidate_dir)
    config_path = candidate_dir_path / "cv_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    exit_code = 0
    try:
        result = _execute(config, candidate_dir_path)
    except Exception as exc:
        if not (candidate_dir_path / "cv_results.json").exists():
            (candidate_dir_path / "cv_results.json").write_text(
                CVResult(
                    candidate_id=config.get("candidate_id", "unknown"),
                    metric=config.get("metric", "unknown"),
                    status="error",
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}",
                ).model_dump_json(indent=2),
                encoding="utf-8",
            )
        exit_code = 1
        result = CVResult.model_validate_json(
            (candidate_dir_path / "cv_results.json").read_text(encoding="utf-8")
        )
    print(json.dumps(result.model_dump(), indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
