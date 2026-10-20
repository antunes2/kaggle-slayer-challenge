# 06 — CV Harness & Sandbox Runner

- **Phase:** 2 (step 2.1) · **Status:** implemented · **Date:** 2026-10-20
- **ADR:** [D0003](../decisions/D0003-orchestrator-deterministic-pipeline.md) (deterministic gates) · plan risk row "hallucinated pipelines"

## Goal

The single source of truth for candidate quality: a standardized cross-validation
harness that trains any candidate pipeline, scores it with the competition metric,
and produces the submission file. Agents **never compute their own scores** — a
pipeline that doesn't run through the harness can't become a candidate.

## Architecture: fixed harness, generated plugin

The researcher agent only ever generates a small `pipeline.py` plugin with a strict
interface:

```python
def make_features(df: pd.DataFrame) -> pd.DataFrame: ...   # optional (fit-free transforms)
def build_model(): ...                                     # required — sklearn-compatible estimator
```

`make_features` only ever sees the **feature frame** — the harness extracts the target
column before invoking it (targets can neither leak into features nor get mangled by
one-hot encoders) — and after the transform it aligns test columns to train columns
(zero-filling unseen categories), so naive `get_dummies` pipelines work.

The harness (our tested code) does everything else:

1. Load train/test/sample_submission from the run's `workspace/data/`.
2. Import `pipeline.py` from the candidate dir, apply `make_features` to features.
3. Run K-fold CV (Stratified for classification) with the protocol's seed — fresh
   `build_model()` per fold; score each fold with the resolved metric.
4. Fit on the full train set, predict test, write `submission.csv` in the exact
   sample_submission shape.
5. Write `cv_results.json` (a `CVResult`, with sha256 fingerprints of `pipeline.py`
   and the submission).

Why not let the LLM write the whole training script: a free-tier model writing full CV
loops is the biggest fragility in the system (wrong folds, wrong metric, fabricated
numbers, malformed submissions). This is the plan's "known-good skeleton" mitigation
applied to modelling itself — and it makes scores *comparable by construction*.

## Sandbox (v1, per plan §8.3)

The harness runs as a **subprocess** (`python -m slayer.cv.harness <candidate_dir>`):

- `cwd` scoped to the candidate dir; `start_new_session=True` so a timeout kills the
  whole process group (hangs, infinite loops).
- Hard timeout (default 600s) → synthetic `CVResult(status="timeout")`.
- stdout+stderr captured to `run.log` — the researcher gets the traceback to fix.
- Harness writes `cv_results.json` even on error (status + traceback), so the runner
  always parses a structured result.
- v1 runs with the project interpreter; docker isolation is a later hardening step.

## Metric registry

`resolve_metric(raw_kaggle_metric)` → scorer + direction + `needs_proba`, with alias
normalization (e.g. `"Roc Auc Score"` → `roc_auc_score`, higher-is-better, proba).
Supported: ROC AUC, accuracy, balanced accuracy, macro F1, RMSE, MSE, MSLE, RMSLE, MAE,
log loss. Unknown metric → clear error (the planner prompt lists supported metrics;
widening the registry is trivial).

Scoring convention: for binary classification with proba-metrics, the positive class is
the **lexicographically last** class (`"Yes" > "No"`, `"1" > "0"`) — documented
assumption, enforced identically in CV and in the final submission.

## Interface

```python
CVProtocol(metric: str, k_folds=5, seed=42, stratified=True, higher_is_better=True)
CVResult(candidate_id, metric, folds[], mean_score, std_score, higher_is_better,
         train_rows, elapsed_s, pipeline_sha256, submission_sha256, status, error)

run_candidate(run: Run, candidate_id: str, protocol: CVProtocol, problem_type: str,
              target_column: str, timeout_s:600) -> CVResult
```

Candidate dir layout (`workspace/models/<cand-id>/`): `pipeline.py` (generated),
`cv_config.json` + `cv_results.json` (harness I/O), `submission.csv`, `run.log`.

## Tests

- Metric registry: normalization, direction, needs_proba, unknown-metric error.
- Harness (in-process, synthetic data): happy path folds/means/hashes/submission shape;
  error path writes status=error.
- Runner (subprocess): happy path on synthetic run; broken pipeline → error + log tail;
  sleeping pipeline → timeout status.

## Retro

Implemented as designed; live-validated against the real EV-purchases data
(hand-written LightGBM baseline: **0.94135 AUC ± 0.0006** over 3 folds, 668k rows, ~49s).
Findings that changed the design **during** this step:

- **Target extraction moved before `make_features`.** First test run exposed that a
  naive one-hot encoder in generated code destroys the target column (and, worse, a
  buggy pipeline could leak it into features). The harness now extracts `y` first and
  hands `make_features` the feature frame only — leakage-by-accident is structurally
  impossible in the plugin interface.
- **Test-column alignment is harness work.** After `make_features`, test features are
  reindexed to train columns (zero-fill for unseen categories). Naive `get_dummies`
  pipelines now "just work", which keeps generated code radically simpler.
- The structured error path paid off immediately: my own hand-written baseline crashed
  first (LightGBM vs pandas-3 `str` dtypes) and the harness reported it cleanly with a
  traceback — exactly the loop the researcher agent will use (fail → read log → fix).
