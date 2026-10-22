# C0004 — Evidence-driven strategy: from a generic playbook to competition evidence

- **Date:** 2026-10-22

## What changed

The agents stopped following a fixed, generic tabular playbook and now reason from
evidence:

1. **Public-notebook scout** (new, [design 15](../designs/15-notebook-scout.md)): before
   planning (and when re-planning), code reads the competition's most-voted public
   notebooks through the official Kaggle API and hands the planner a deterministic,
   size-capped digest (~3-4k tokens) of their feature/encoding/model lines.
2. **Prompts** (`ml_researcher.md`, `planner.md`, `planner_extend.md`, `reviewer.md`):
   - the researcher's "ratios / interactions / log1p beat elaborate constructions" and
     the planner's fixed "baseline → feature groups → alternative family → tuning →
     blend" sequence are replaced by **representation hypotheses ordered by evidence**
     (values as categories + in-estimator target encoding, digit/rounding structure,
     category-combination keys, in-estimator frequency encoding) and a **capacity**
     rule (use the unused time budget);
   - class weighting is no longer mandated for ranking metrics;
   - "noise" now means the **paired** fold-difference noise, not one model's fold std
     (planner gate guidance + reviewer statistical-honesty rule);
   - the planner never plans a plain-average step (blending is automatic).
3. **Robustness**: connection errors get their own long retry budget (overnight runs
   died on a 4-second DNS blip); the blender no longer registers duplicate blends.

All prompt text stays **generic for tabular data** — nothing names this competition.
Competition specifics arrive only through the data profile and the notebook digest.

## Why — the evidence (two live runs on `playground-series-s6e9`, 2026-10-22)

Known (measured on our artifacts):

| Finding | Evidence |
|---|---|
| Hard ceiling | Every candidate across 2 runs / 3 re-entry rounds: CV 0.9415-0.9420 (target line 0.94640). |
| Generic ideas only | No pipeline ever used digit features, numeric-as-category encoding, or frequency encoding; target encoding was tried only on the 6 string columns (no gain). |
| Models are near-copies | OOF correlation LightGBM vs CatBoost = **0.9984**. |
| Blender: honest but idle | Nested check (weights chosen on 4 folds, scored on the 5th): **+0.00003** mean gain — not optimistic, just nothing to combine. |
| Wrong noise yardstick | cand-0005 vs cand-0001 paired fold diffs: +0.000183 mean, **0.000028 std** — 28× below the fold std (0.0008) the prompts called "noise"; the 0.0005 gate blocked consistent 5/5-fold gains. |
| Class weighting is inert | Identical pipelines with/without `class_weight="balanced"` scored identically (0.941555). |
| Unused capacity | Folds took 15-25 s of a ~300 s per-fold budget; models used 600-1200 trees at lr 0.03-0.05. |
| Steep top of the LB | 0.9460 ≈ top 35%, 0.9463 ≈ top 27%, 0.9464 ≈ top 20%. |
| Fragile overnight runs | Run died on `Temporary failure in name resolution`; old retry budget was 3.5 s total. |

From the public notebooks (read via the API; their claims are **not** reproduced by us):

- Top single models report CV ≈ 0.946. The most common ingredients: digit features
  (`(x // 10**k) % 10`), numerics copied as strings and target-encoded, frequency
  encoding, category-combination keys, low learning rates with many trees.
- One notebook reporting CV 0.94614 (HistGBM) uses **no external data** — so the
  ceiling is reachable inside our rules in principle. Others reaching LB 0.94633-0.94650
  also use the competition's original dataset (external data; our rule forbids it —
  user decision 2026-10-22: keep the rule).

Inferred: the 0.0044 gap is a **representation** gap (information the raw values
hide), not a model-family, tuning, or ensembling gap.

## Live test (2026-10-22, fork `20261022-031417-…-scout-test`, one re-entry round)

Fork of the plateaued run with its 3 unfinished generic round-2 steps removed and one
extra round allowed. Round 3 with the scout + new prompts:

- Scout: 4 notebooks, ~3.4k tokens. Planner assessment cited the gap/spread ratio,
  the unused time budget (median fold 24 s of 1800 s), and patterns from named
  notebooks; planned: digit/rounding features → frequency encoding + combined keys →
  multi-smoothing target encoding → capacity increase. No `[EXTERNAL]` idea planned.
- **cand-0016 (digit/rounding features, LightGBM): CV 0.94404 ± 0.00064** — the first
  candidate above the 0.9420 ceiling (+0.0020, ~4× the entire previous spread). On the
  leaderboard table that is ≈ top 46% (from ≈ top 53%).
- **cand-0017 (frequency encoding + combined keys): CV 0.94426 ± 0.00069** — approved
  and (mock) submitted. Remaining round steps: multi-smoothing target encoding 0.94323;
  "capacity increase" 0.94263 (worse — more capacity on the richer feature set did not
  pay here). Test stopped after these to free the CPU for the live campaign.
- The reviewer REVISE'd cand-0016 for a guidance-compliance detail (digit features not
  typed as categorical), so it stays `scored` and cannot be submitted until revised —
  the re-review / REVISE handling gap (step 4.5) now blocks the best candidate.

The hypothesis is **confirmed in direction**; the magnitude still trails the public
notebooks' 0.946, which also use far higher capacity (lr 0.005-0.02, 10-20k trees,
early stopping).

## Plan impact

- Master plan §11: new step **4.4** (notebook scout + evidence-driven prompts + retry
  hardening). Step 4.2/4.3 retros updated (re-entry and blending work end-to-end but
  produced no score gain with generic ideas).
- Next step recorded: **paired-fold submission gate + reviewer re-review fix** (code
  part of the noise finding; the prompt part landed here).
- No ADR superseded; D0003 holds (code builds the digest; the LLM only reads it).
