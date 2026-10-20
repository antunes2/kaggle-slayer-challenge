"""CV harness entry point: runs inside the sandbox subprocess for one candidate."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from slayer.cv.metrics import ResolvedMetric, resolve_metric
from slayer.schemas.cv import CVResult, FoldScore


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


def _score_fold(
    model, X_val, y_val, metric: ResolvedMetric, problem_type: str
) -> float:
    """Score one fold with the resolved metric, using proba when required."""
    if metric.needs_proba and problem_type == "binary_classification":
        if hasattr(model, "predict_proba"):
            classes = [str(c) for c in model.classes_]
            positive = _positive_class(y_val)
            proba = model.predict_proba(X_val)[:, classes.index(str(positive))]
            return float(metric.scorer(y_val, proba))
    if metric.needs_proba and hasattr(model, "predict_proba"):
        return float(metric.scorer(y_val, model.predict_proba(X_val)))
    return float(metric.scorer(y_val, model.predict(X_val)))


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


def _execute(config: dict, candidate_dir: Path) -> CVResult:
    """Run the full CV + final-fit + submission flow for one candidate."""
    started = time.time()
    pipeline_path = candidate_dir / "pipeline.py"
    data_dir = Path(config["data_dir"])
    target_column = config["target_column"]
    problem_type = config["problem_type"]
    metric = resolve_metric(config["metric"])

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
        n_splits=int(config["k_folds"]), shuffle=True, random_state=int(config["seed"])
    )
    folds: list[FoldScore] = []
    for i, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        model = pipeline.build_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        score = _score_fold(model, X.iloc[val_idx], y.iloc[val_idx], metric, problem_type)
        folds.append(FoldScore(fold=i, score=round(score, 6)))

    scores = [f.score for f in folds]
    mean = round(sum(scores) / len(scores), 6)
    std = round(
        (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5, 6
    )

    final_model = pipeline.build_model()
    final_model.fit(X, y)
    out_frame = _predict_submission(
        final_model, test, target_column, submission, problem_type
    )
    submission_path = candidate_dir / "submission.csv"
    out_frame.to_csv(submission_path, index=False)

    return CVResult(
        candidate_id=config["candidate_id"],
        metric=metric.key,
        status="ok",
        folds=folds,
        mean_score=mean,
        std_score=std,
        higher_is_better=metric.higher_is_better,
        train_rows=len(train),
        elapsed_s=round(time.time() - started, 2),
        pipeline_sha256=_sha256_file(pipeline_path),
        submission_sha256=_sha256_file(submission_path),
    )


def main(candidate_dir: str) -> int:
    """Entry point for `python -m slayer.cv.harness <candidate_dir>`."""
    candidate_dir_path = Path(candidate_dir)
    config_path = candidate_dir_path / "cv_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        result = _execute(config, candidate_dir_path)
        exit_code = 0
    except Exception as exc:
        result = CVResult(
            candidate_id=config.get("candidate_id", "unknown"),
            metric=config.get("metric", "unknown"),
            status="error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}",
        )
        exit_code = 1
    (candidate_dir_path / "cv_results.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    print(json.dumps(result.model_dump(), indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
