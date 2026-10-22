# Q0001 — Which OpenRouter model(s) should each agent use?

- **Status:** settled → [D0006](../decisions/D0006-model-routing-gemini-flash.md)
- **Raised:** 2026-10-19
- **Blocks:** nothing — default in place (`google/gemini-2.5-flash` tier per
  `config/agents.yaml`).

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

**Note added 2026-10-20 (post live-validation):** the live iteration runs suggest the
highest-leverage upgrade is the **researcher only** — the other agents' live output
(analyst reports, plans, verdicts) is already strong on the cheap tier, while codegen
is where cycles get burned on repairs. Since routing is per-agent, the cheap A/B
experiment is: same competition, same budget, `ml_researcher.model` upgraded (e.g. a
frontier reasoning model), compare CV trajectories and failed-candidate counts. A
better model should mainly *reduce variance* (fewer broken candidates, better
reviews) and raise the idea ceiling — but the evidence so far says the bigger
percentile gains come from iteration depth + ensembles + real LB feedback, not
model tier.

**Update 2026-10-22 — A/B started:** `ml_researcher` switched from `openai/gpt-5-mini`
to `anthropic/claude-sonnet-5.5` ($2/M in, $10/M out; structured-output ping verified,
cost catalog-estimated). Baseline to compare against (run `20261022-001955`, gpt-5-mini):
9 candidates, 3 failed + 1 stuck, 4/4 CatBoost attempts broken on contract errors, CV
0.94147–0.94183. Measure on the next fresh run: failed-candidate count, repair cycles,
CV trajectory, LLM spend.
