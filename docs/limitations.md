# Limitations & Known Gaps

> Honest inventory of what Slayer does **not** guarantee, as delivered (v1.0,
> 2026-10-22). Each entry: what it is, why it matters, what would fix it. Evidence and
> history: `docs/plan/changes/C0004-evidence-driven-strategy.md`, `docs/plan/designs/`.

## 1. Results — the top-20% goal is not achieved

| Evidence | Value |
|---|---|
| s6e9 (closed) — first real submission | LB 0.94149 ≈ **top 71%** (estimated position) |
| s6e9 — best CV after the evidence-driven changes (C0004) | 0.94426 ≈ top 45% (CV-based estimate) |
| **s6e10 (active, live campaign, 2026-10-22)** — 4 real submissions, 2 re-entry rounds, $0.80 | LB 0.95974 → 0.95988 → **0.96015**, rank 1240/2714 = **top 45.7%** (listed) |
| s6e10 top-20% line | 0.96178 (gap 0.00163) |

- **Why:** the strongest public notebooks combine three things: (a) the competition's
  *original* dataset, (b) deep-learning tabular models (RealMLP, TabPFN) for ensemble
  diversity, (c) large stacks. Slayer now supports (a) (D0009, not yet live-validated)
  and has a blender, but (b) is out of scope on the target hardware (D0010).
- **Competitive context:** top teams build one specialised pipeline per competition
  and iterate for weeks. Slayer is a general system that spends about $1 and a few
  hours per campaign.
- **Implication:** treat leaderboard percentile as *best effort*. The system reliably
  produces valid, reviewed, gated submissions. It does not promise a ranking.

## 2. Generality — validated on two binary-AUC competitions

- **Live-validated:** binary classification with ROC AUC on two Playground episodes
  (s6e9 EV purchases, string target; s6e10 airline satisfaction, bool target,
  untouched configuration).
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

## 3b. External data (D0009)

- On by default only for **Playground** competitions (`external_data: auto`). Kaggle's
  API cannot read competition rules: before `--external-data` on any other
  competition, **check its rules yourself**. Breaking them risks disqualification.
- Datasets are found only via top public notebooks' load paths (max 2, CSV, ≤ 500 MB).
  If notebooks don't use one, nothing is downloaded. Bare legacy paths are resolved by
  search, which can pick the wrong dataset.
- Not yet validated live: whether it raises the score is unproven.

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

- **Deep learning** (D0010): the target machine is a laptop with an integrated GPU;
  neural tabular models would train for days. This is a known ceiling on ensemble
  diversity versus top notebooks.
- Phase 5: background daemon + `attach`, `slayer chat`, agentic orchestrator (the
  `orchestrator` agent entry is reserved, unused).
- Interactive competition picker (a browse command exists: `slayer competitions`).
- Automatic competition selection (an explicit non-goal).

## 7. Small known wrinkles

- Unused workspace folders (`features/`, `submissions/`, `logs/`) are created per run.
- Steps inherit planner guidance append-only; overriding an old rule relies on an
  "Overrides earlier guidance:" convention.
- The submission gate's paired-fold rule (≥ k-1 winning folds, mean > 2× std) is a
  heuristic, not a formal test.

## 8. Improvement opportunities (analysed, not implemented)

Ranked by expected gain per effort. "Gain" is an honest estimate, unverified unless
stated.

| # | Opportunity | Type | Expected gain | Risk / trade-off |
|---|---|---|---|---|
| 1 | Show the planner each candidate's OOF correlation and marginal blend contribution, so it plans *diverse* members (blends were the only LB gains on s6e10) | code + prompt | plausible +0.0002–0.0005 | the planner may chase weak "diverse" models |
| 2 | Stronger planner model (it chooses the ideas; it runs on the cheapest tier) | config | plausible, small | ~$0.15/run; unverified |
| 3 | Stacking meta-learner on OOF (nested CV) instead of weighted average only | code | +0.0001–0.0003 typical | overfitting if not nested; more code |
| 4 | Average the 5 fold models' test predictions instead of one full refit | harness | +0–0.0002, more stable | low |
| 5 | Stronger reviewer model (fewer nit-pick REVISEs, each costs ~$0.05 + 5 min) | config | speed, not score | small cost |
| — | Researcher sees notebook code directly | prompt | little evidence of need | +$0.15/run |
| — | More capacity (lower LR, more trees) | prompt | **hurt** on s6e9 (0.94263 vs 0.94426) | skip |
| — | Deep learning, GPU | scope | largest remaining lever | D0010 (hardware) |

Cheapest real lever today: resume a live campaign daily (each round added about
+0.0001–0.0003 on the leaderboard for ~$0.40).
