# Q0001 — Which OpenRouter model(s) should each agent use?

- **Status:** open
- **Raised:** 2026-10-19
- **Blocks:** nothing — default in place (`minimax/minimax-m3.1-flash:free` for all
  agents via `config/agents.yaml` defaults).

## Question

`config/agents.yaml` routes each agent to an OpenRouter model. Which model per agent
gives the best quality/cost tradeoff for this project?

## Context

- Roles differ: planner and ml_researcher need reasoning + code-generation quality;
  reviewer needs disciplined adversarial checking; orchestrator routes and summarizes.
- The existing `test_openrouter.sh` uses `minimax/minimax-m3.1-flash:free` — zero cost,
  unknown ceiling for multi-step codegen tasks.
- Strong models cost real money against the per-run USD budget (default: $10).

## Options

1. **Free model everywhere to start** — measure quality per phase; upgrade where the
   loop actually fails. Cheapest path to a working pipeline.
2. **Strong model for planner + ml_researcher from day one** (e.g. a Claude Sonnet /
   GPT-class model), cheap models elsewhere.
3. **Per-run quality tier** — `economy | balanced | premium` presets in `slayer.yaml`
  mapping to per-agent models.

## Recommendation

Start with **(1)** — revisit once the loop runs end-to-end, because actual failure
modes will tell us which agent needs a better model. Revisit trigger: any phase that
consistently produces unusable output. Option (3) is a nice later addition once we
have evidence for what each role needs.
