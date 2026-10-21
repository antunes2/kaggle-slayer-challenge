# D0006 — Model routing: fast Gemini Flash tier as default

- **Status:** accepted
- **Date:** 2026-10-20
- **Settles:** [Q0001](../questions/Q0001-openrouter-models-per-agent.md)

## Context

The initial free model (`minimax/minimax-m3.1-flash:free`) is a *reasoning* model:
excellent analyst/planner output quality, but 1.5–3 minutes per structured call on the
free tier. That made foreground runs fragile (120s shell timeouts killed runs
mid-call) and slowed every live validation.

## Options considered (benchmark, small structured call, 2026-10-20)

| Model | Latency | Quality signal | Cost |
|-------|---------|----------------|------|
| google/gemini-2.5-flash | **1.0s** | strong ("LightGBM" for tabular AUC) | $0.30/M in, $2.50/M out |
| google/gemini-2.5-flash-lite | **1.2s** | good | $0.10/M in, $0.40/M out |
| openai/gpt-4o-mini | 1.4s | weaker ("Random Forest") | $0.15/M in |
| qwen/qwen3-235b-a22b-2507 | 18.3s | good | $0.09/M in |
| minimax/minimax-m3.1-flash:free | 90–180s | strong | free |

## Decision

Default agent routing moves to **`google/gemini-2.5-flash`**, with
**`google/gemini-2.5-flash-lite`** for the high-volume orchestrator and data-analyst
roles (per-agent overrides in `config/agents.yaml`). A full run's LLM spend drops to
seconds of wall time and roughly $0.01–0.10 — well inside the per-run USD budget, which
becomes meaningful enforcement rather than a formality.

## Consequences

- Foreground runs and live tests are no longer at the mercy of reasoning-token latency.
- The free minimax model remains a valid per-agent override for zero-cost operation —
  Q0001's "revisit on quality failure" trigger still applies: if the Flash tier
  produces unusable analyst/planner/researcher output, upgrade the *specific* agent
  (e.g. to a stronger reasoning model) rather than the whole roster.
- Budget accounting (journal `agent.call` costs) now tracks real money.
