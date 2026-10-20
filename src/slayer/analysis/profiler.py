from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from slayer.analysis.models import ColumnProfile, DatasetProfile
from slayer.analysis.render import render_profile_md
from slayer.workspace import Run


class AnalysisError(Exception):
    """Raised when the dataset cannot be profiled."""

    pass


def _find_csv(data_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    """Locate a CSV by trying each glob pattern in order."""
    for pattern in patterns:
        matches = sorted(data_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _fmt(value) -> str:
    """Format a statistic compactly ('-' for NaN)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _clean_corr(value) -> float:
    """Round a correlation, treating NaN as zero."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return round(float(value), 4)


def profile_data(run: Run) -> DatasetProfile:
    """Compute the dataset profile and write eda/profile.{json,md}."""
    data_dir = run.workspace("data")
    train_path = _find_csv(data_dir, ("train.csv", "train*.csv"))
    test_path = _find_csv(data_dir, ("test.csv", "test*.csv"))
    submission_path = _find_csv(data_dir, ("sample_submission.csv", "*submission*.csv"))
    if train_path is None or submission_path is None:
        raise AnalysisError(f"missing train/sample_submission CSVs in {data_dir}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path else None
    submission = pd.read_csv(submission_path)

    sub_cols = list(submission.columns)
    targets = [c for c in sub_cols if c != sub_cols[0]]
    if not targets:
        raise AnalysisError(f"could not detect target from {submission_path.name} columns")
    target_column = str(targets[0])
    notes = []
    if len(targets) > 1:
        notes.append(f"multiple submission columns {targets}; using '{target_column}'")

    target = train[target_column]
    target_nunique = int(target.nunique(dropna=True))
    if target_nunique == 2:
        guess = "binary_classification"
    elif str(target.dtype) == "object":
        guess = "multiclass_classification"
    elif target.dtype.kind == "f":
        guess = "regression"
    elif target_nunique <= 20:
        guess = "multiclass_classification"
    else:
        guess = "regression"

    columns = []
    for col in train.columns:
        series = train[col]
        missing_pct = round(float(series.isna().mean() * 100), 2)
        nunique = int(series.nunique(dropna=True))
        numeric = pd.api.types.is_numeric_dtype(series)
        facts: dict[str, str] = {}
        if numeric:
            desc = series.describe()
            facts = {
                "mean": _fmt(desc.get("mean")),
                "std": _fmt(desc.get("std")),
                "min": _fmt(desc.get("min")),
                "max": _fmt(desc.get("max")),
            }
        else:
            value_counts = series.value_counts(normalize=True, dropna=True).head(3)
            for value, share in value_counts.items():
                facts[str(value)[:30]] = f"{share * 100:.1f}%"
        columns.append(
            ColumnProfile(
                name=str(col),
                dtype=str(series.dtype),
                missing_pct=missing_pct,
                nunique=nunique,
                numeric=numeric,
                facts=facts,
            )
        )

    if guess == "regression":
        desc = target.describe()
        target_summary = {key: _fmt(value) for key, value in desc.items()}
    else:
        value_counts = target.value_counts(normalize=True, dropna=True).head(10)
        target_summary = {str(key): f"{value * 100:.1f}%" for key, value in value_counts.items()}
        notes.append(f"target has {target_nunique} class(es)")

    top_correlations: list[tuple[str, float]] = []
    target_numeric = target
    if not pd.api.types.is_numeric_dtype(target) and target_nunique == 2:
        classes = sorted(target.dropna().astype(str).unique())
        target_numeric = target.astype(str).map({c: i for i, c in enumerate(classes)})
    if pd.api.types.is_numeric_dtype(target_numeric):
        numeric_cols = [
            c
            for c in train.columns
            if c != target_column and pd.api.types.is_numeric_dtype(train[c])
        ]
        if numeric_cols:
            corrs = train[numeric_cols].corrwith(pd.to_numeric(target_numeric)).dropna()
            ordered = corrs.abs().sort_values(ascending=False).head(10)
            top_correlations = [(str(c), _clean_corr(corrs[c])) for c in ordered.index]

    profile = DatasetProfile(
        competition=run.competition,
        train_rows=len(train),
        train_cols=train.shape[1],
        test_rows=len(test) if test is not None else 0,
        target_column=target_column,
        problem_type_guess=guess,
        columns=columns,
        target_summary=target_summary,
        top_correlations=top_correlations,
        notes=notes,
    )
    eda_dir = run.workspace("eda")
    eda_dir.mkdir(parents=True, exist_ok=True)
    (eda_dir / "profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    (eda_dir / "profile.md").write_text(render_profile_md(profile), encoding="utf-8")
    return profile
