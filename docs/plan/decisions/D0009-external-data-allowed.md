# D0009 — External data: allowed (public Kaggle datasets; auto = Playground only)

- **Status:** accepted (reverses the "competition files only" rule kept in C0004)
- **Date:** 2026-10-22

## Context

On both live competitions the strongest public notebooks also train on, or derive
statistics from, the competition's *original* dataset (s6e9: LB 0.94633-0.94650;
s6e10: 15 `[EXTERNAL]` digest lines, e.g. `arseniyshutko/binary-aviation-satisfaction-129k`).
Slayer reading only the competition's files was a structural cap on results.

## Options considered

1. Keep the rule (C0004's position): safest, capped results.
2. Always allow: simplest; risky where competition rules forbid external data.
3. **Allow with a policy**: `auto` = on for Playground competitions (whose rules
   typically allow public external data), off otherwise; `on`/`off` override.

## Decision

Option 3. `defaults.external_data: auto | on | off`, CLI `--external-data/--no-external-data`.
Datasets are discovered only from top public notebooks (the digest's `[EXTERNAL]` lines),
must be public Kaggle datasets, max 2 per run, CSV only, ≤ 500 MB total, downloaded
via the official API into `workspace/external/` with a manifest (source refs → credit
and reproducibility). Pipelines receive them through two optional hooks
(`make_features(df, external)`, `external_rows(external)`); external rows join
**training folds only**, so CV and OOF stay competition-only and honest.

## Consequences

- **Rules compliance is the user's responsibility** — the API cannot read a
  competition's rules; `auto` limits the default to Playground. For any other
  competition, check its rules before `--external-data`.
- Not yet live-validated (next resume of the s6e10 run is the test).
- Data licenses of downloaded datasets apply; the manifest records each source.
