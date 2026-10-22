# 14 — BLEND: OOF Persistence + Deterministic Blender

- **Phase:** 4 (step 4.3) · **Status:** implemented · **Date:** 2026-10-22
- **Principle:** code computes scores (architecture.md) — blending is arithmetic over
  harness-produced predictions, so it is code-owned, not an LLM step.

## Goal

Make ensembling cheap and reliable. Until now an "Ensemble" roadmap step refit every
member inside one pipeline (16 min live) and nothing could combine candidates that had
already been scored. Persisting out-of-fold (OOF) predictions lets a deterministic
blender combine any reviewed candidates at near-zero cost — typically +0.001–0.003 on
playground tabular comps.

## Harness: persist predictions

All candidates share the same folds (same splitter, seed, k and target), so OOF
vectors align row by row. The harness now records, per fold, the predictions it already
computes for scoring, and on success writes:

- `oof.npy` — float64, one prediction per train row (binary positive-class
  probability, or the regression prediction),
- `fold_ids.npy` — int8 fold index per train row.

Only **blendable** predictions are saved: 1-D and numeric (binary proba metrics,
regression). Multiclass probabilities and label predictions (accuracy/F1) are not
saved — averaging them is not meaningful.

## Blender (`cv/blend.py`)

Greedy forward selection with replacement (Caruana et al. 2004):

1. start from the best single member,
2. repeatedly add (with replacement) the member that most improves the **mean
   per-fold metric** of the averaged blend, using the same folds as CV,
3. stop when nothing improves or after 15 additions; weights = selection counts / total.

Deterministic: ties go to the lowest candidate id.

## Blend step (`loop/blending.py`, end of every ITERATE pass)

- **Pool**: candidates the reviewer approved (`approved` / `submitted`), with a CV
  score, `oof.npy`, `fold_ids.npy`, and `submission.csv`, excluding other blends.
  Supported only when blending is meaningful (binary + probability metric, or
  regression). Needs ≥ 2 members whose fold ids match.
- **Dedupe**: the pool key (sorted member ids) is journaled as `blend.evaluated`; a
  pool that has already been evaluated is skipped (resumes and rounds re-blend only
  after the pool grows).
- **Result**: if the blend uses ≥ 2 members and beats the best single member's mean
  fold score, it is registered as a `Blend` candidate (`params.blend` = weights,
  `params.blend_pool` = key), with `blend.json`, `cv_results.json`, `oof.npy`, and
  `submission.csv` (weighted average of member submissions, verified to share the
  id order). Status goes straight to `approved` — every member was reviewer-approved
  and the blend itself is deterministic code — then `submit_phase` applies the usual
  CV, quota, and approval gates.
- Blends never count as roadmap steps (no `roadmap_step` param).

## Risks

- Selection and evaluation use the same OOF → slightly optimistic CV. With few members
  and large datasets the effect is small, and the `min_cv_improvement` submission gate
  still applies.
- Only candidates scored by the new harness have OOF; older runs blend nothing.

## Tests

- Harness writes `oof.npy`/`fold_ids.npy` that line up with train rows; the mean fold
  score recomputed from OOF matches the CV result.
- Hill-climb: a complementary pair beats both singles; a dominated member gets weight 0;
  results are deterministic.
- Blend step: registers an approved Blend with averaged submission + journal event;
  skips < 2 members, unsupported metrics, mismatched folds, and pools already evaluated.

## Retro (2026-10-22, live run `20261022-031417`)

- **Correctness: verified.** OOF fold scores reproduce the harness CV; blends were
  registered and gated live (cand-0007: 0.94200 vs best single 0.94196).
- **Optimism: measured, negligible.** Nested check — weights chosen on 4 folds, scored
  on the held-out 5th: mean gain **+0.00003** (in-sample: +0.00004).
- **Value: ~zero on this run.** LightGBM vs CatBoost OOF correlation **0.9984** — the
  candidates were near-copies, so there was nothing to blend. The blender becomes
  useful only when candidates use different representations (C0004).
- **Duplicate fix:** a grown pool that yields identical weights no longer registers a
  second, identical Blend (cand-0013 duplicated cand-0007 live).
