# Limitations & Known Gaps

> Honest inventory of what Slayer does **not** guarantee, as delivered (v1.0,
> 2026-10-22). Each entry: what it is, why it matters, what would fix it. Evidence and
> history: `docs/plan/changes/C0004-evidence-driven-strategy.md`, `docs/plan/designs/`.

## 1. Results — the top-20% goal is not achieved

| Evidence | Value |
|---|---|
| First real Kaggle submission (`playground-series-s6e9`) | LB 0.94149 ≈ **top 71%** (closed comp, estimated position) |
| Best CV after the evidence-driven changes (C0004) | 0.94426 ≈ **top 45%** (CV-based estimate) |
| Top-20% line on that leaderboard | 0.94640 |

- **Why:** the remaining gap is mostly *representation*: information hidden in how
  the data was generated. The strongest public notebooks also train much larger models
  and use the competition's *original* dataset. Slayer deliberately reads only the
  competition's own files (a kept decision), which caps it below those notebooks.
- **Fix paths:** allow vetted external datasets (a rule change, plus a small harness
  change); much higher model capacity with internal early stopping; more re-entry
  rounds driven by the notebook digest.
- **Implication:** treat leaderboard percentile as *best effort*. The system reliably
  produces valid, reviewed, gated submissions. It does not promise a ranking.

## 2. Generality — validated on one kind of competition

- **Live-validated:** binary classification with ROC AUC, on one Playground episode.
- **Unit-tested only:** regression, multiclass, and the other registry metrics
  (accuracy, balanced accuracy, macro F1, RMSE/MSE/MSLE/RMSLE, MAE, logloss).
- **Hard assumptions:** the data has `train.csv`, `test.csv`, and
  `sample_submission.csv`, and the target column is identified from the sample
  submission. Multi-file, image, text, time-series, or code-competition formats are
  unsupported.
- **Unsupported metrics** (anything outside the registry) fail loudly at PLANNING.
- **OOF blending** works only for binary-probability metrics and regression.
- **Fix path:** one live run per problem type; data-layout adapters; more registry
  metrics.

## 3. Reliability

- **Ctrl+C** pauses the run cleanly and kills the running training subprocess (Q0003).
  A *hard* kill (`kill -9`, power loss) can still leave an orphaned harness process;
  stop it with `pkill -9 -f slayer.cv.harness`. The harness self-deadline cannot
  interrupt a long C-level fit.
- **Network:** connection blips are retried for about 7.5 minutes; longer outages fail
  the run (resumable). Kaggle API calls have no comparable backoff.
- **Leaderboard position:** our entry is matched by username. Late submissions to
  closed competitions are never listed, so the position is *estimated* by inserting
  our score into the field. Dry runs estimate from CV.
- **Mock LB scores** in dry runs are simulated (CV plus deterministic noise), not real.
- **Cost accounting:** when OpenRouter omits the cost, it is estimated from catalog
  prices.

## 4. Security & sandboxing

- LLM-generated `pipeline.py` runs in a **local subprocess of the project venv**, not a
  container. It is restricted by prompt contract and review, not by isolation: no
  filesystem, network, or resource sandbox beyond the time budget.
- The public-notebook digest is untrusted text. It is never executed and is framed to
  the planner as evidence, but it is still LLM input (prompt-injection risk mitigated,
  not eliminated).
- **Fix path:** docker sandbox for the harness (planned Phase 5).

## 5. Dependence on the LLM

- Plan and code quality depend on prompts and model choice (`config/agents.yaml`). Runs
  are not bit-reproducible: same inputs, different proposals.
- The reviewer is an LLM: it can approve a subtle leak or send back a fine candidate.
  Scores are always harness-computed, never LLM-claimed.
- The notebook scout is a line-filter heuristic. It can miss logic inside helper
  functions, titles' score claims are unverified, and new competitions have few notebooks.

## 6. Not built (out of scope for v1)

- Phase 5: background daemon + `attach`, `slayer chat`, agentic orchestrator (the
  `orchestrator` agent entry is reserved, unused).
- Interactive competition picker (a browse command exists: `slayer competitions`).
- Automatic competition selection (an explicit non-goal).
- Deep learning, GPU training.

## 7. Small known wrinkles

- Unused workspace folders (`features/`, `submissions/`, `logs/`) are created per run.
- Steps inherit planner guidance append-only; overriding an old rule relies on an
  "Overrides earlier guidance:" convention.
- The submission gate's paired-fold rule (≥ k-1 winning folds, mean > 2× std) is a
  heuristic, not a formal test.
