# Q0004 — How does the user pick a competition? (discovery UX)

- **Status:** open (documented; user decision pending on flavor)
- **Raised:** 2026-10-20
- **Blocks:** nothing — `slayer run <ref>` works today with the exact ref.

## Question

Today the user must supply the exact Kaggle competition ref
(`slayer run playground-series-s6e9`). Should the CLI surface browsable options
instead of requiring foreknowledge?

## Context

- The building blocks already exist and are live-proven:
  `KaggleClient.list_competitions(search=..., category=..., page_token=...)`
  (used by `find_competition`) — metric, deadline, and quota are shown per comp.
- Auto-*selection* of competitions is an explicit v1 non-goal (plan §3) — the user
  stays in charge of *which* competition; the question is only about ergonomics.
- INIT already validates the ref immediately with an actionable error if wrong.

## Options

1. **`slayer competitions` browse command** — list active comps with
   `--search/--category/--deadline-before` filters (ref, title, metric, deadline,
   quota). ~30 lines + tests; composes with the user copying a ref into
   `slayer run`. Recommended minimum.
2. **Interactive picker** — `slayer run` with no argument prompts with a searchable
   list (typer menus are limited; would want a small fuzzy-prompt lib). Nicer, more
   dependency + code than the value it adds for a power user.
3. **`slayer run --suggest`** — heuristic suggestions (category + deadline fit +
   "playground first" policy). Edges toward the auto-selection non-goal; needs an
   explicit user policy to be trustworthy.

## Recommendation

**(1)** in a Phase 4 ops step; revisit (2) only if the CLI's audience broadens;
**(3)** stays a non-goal for v1. Nothing to decide urgently — the current flow
works and fails fast on bad refs.
