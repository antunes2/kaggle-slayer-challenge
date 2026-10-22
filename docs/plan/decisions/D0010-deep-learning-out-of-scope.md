# D0010 — Deep learning stays out of scope (hardware)

- **Status:** accepted (confirms plan §3 non-goal with evidence)
- **Date:** 2026-10-22

## Context

The s6e10 notebook digest shows top public notebooks using neural tabular models
(RealMLP via PyTabKit, TabPFN) inside their stacks — a real source of model diversity
our gradient-boosting-only toolbox lacks.

## Options considered

1. Allow deep-learning libraries in pipelines (torch, pytabkit, tabpfn).
2. Keep the classical toolbox (LightGBM, XGBoost, CatBoost, scikit-learn).

## Decision

Option 2. The operating machine is a modest laptop with an integrated GPU: neural
models on ~700k-row datasets would train for hours to days per candidate, far past the
per-candidate time budget, making campaigns impractical.

## Consequences

- A known ceiling on ensemble diversity and on matching top public notebooks
  (documented in `docs/limitations.md`).
- Revisit on GPU hardware: the harness contract already supports any sklearn-compatible
  estimator, so the change would be mostly dependency + prompt work.
