# D0008 — Competition discovery: a browse command

- **Status:** accepted
- **Date:** 2026-10-22
- **Settles:** [Q0004](../questions/Q0004-competition-discovery-ux.md)

## Context

Users had to know a competition's exact ref. The user asked for a way to see the
available competitions, choose one, and run it.

## Options considered

Browse command (option 1), interactive picker in `slayer run` (option 2), heuristic
suggestions (option 3, edges into the auto-selection non-goal). See Q0004.

## Decision

`slayer competitions` — lists competitions via the existing `list_competitions`
(official API) with `--search`, `--category` (default `playground`; `all` disables the
filter), `--sort-by` (default `latestDeadline`), `-n`. Shows ref, deadline, metric,
daily quota, title, and `[entered]`; ends with the `slayer run <ref>` hint. The user
still picks; no picker dependency, no auto-selection.

## Consequences

- Zero new dependencies; read-only. Closed competitions can appear (Kaggle returns
  them); the deadline column makes that visible.
- A picker (option 2) remains possible later on top of the same listing.
