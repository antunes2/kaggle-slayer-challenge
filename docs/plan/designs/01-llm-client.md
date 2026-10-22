# 01 — LLM Client (OpenRouter)

- **Phase:** 0 (step 0.2) · **Status:** implemented · **Date:** 2026-10-19
- **Open questions:** [Q0001](../questions/Q0001-openrouter-models-per-agent.md) (model routing)

## Goal

A single, thin, testable gateway from agents to OpenRouter chat completions:
plain-text chat, schema-validated structured outputs, retries with backoff, and
per-call token/USD accounting. Agents never touch HTTP directly.

## Interface

```python
LLMClient(
    api_key: str, base_url: str = "https://openrouter.ai/api/v1",
    timeout_s: int = 180, max_retries: int = 3,
    retry_wait_s: float = 0.5,              # exponential backoff base; 0 in tests
    transport: httpx.BaseTransport | None = None,  # test injection (MockTransport)
)

client.complete(messages, model, *, temperature=0.2, max_tokens=4096) -> LLMResponse
client.complete_structured(messages, model, response_model: type[T], ...) -> tuple[LLMResponse, T]
client.ping(model) -> LLMResponse           # tiny "reply ok" call, used by `slayer init`
client.cost -> CostReport                   # calls, total_tokens, total_usd, priced_calls
```

- `messages`: `[{"role": ..., "content": ...}]`.
- `LLMResponse`: `content`, `model`, `usage` (tokens), `cost_usd` (None when the
  provider doesn't report a price).

## Structured outputs strategy

1. Request `response_format: json_schema` with the pydantic model's JSON schema
   (OpenRouter normalizes/pass-through per provider).
2. If the reply isn't pure JSON, extract the outermost `{...}` block and re-parse
   (models love wrapping JSON in prose/fences).
3. Validate with pydantic. Failure raises `LLMValidationError` with the raw content
   preserved for the journal/debugging.

## Retries & errors

- Retried: connection errors, HTTP 408/429/5xx — exponential backoff (`retry_wait_s * 2^attempt`).
- Not retried: other 4xx (auth, bad request) — raise immediately.
- Taxonomy: `LLMError` (base, optional `status_code`) · `LLMValidationError` (with `raw`).
- Constructing a client with an empty API key raises `LLMError` — guards `slayer init`
  and the loop from calling with missing secrets.

## Cost accounting

Per call: record usage tokens and OpenRouter's `cost` (USD) when present (free models
report `0.0`). Accumulated in an in-memory `CostReport`; wiring into the journal and
per-run budget enforcement happens in step 0.3 / Phase 3 — the client only measures.

## Tests

- **Unit** (httpx `MockTransport`, no network): happy path, payload shape (response_format
  sent), schema validation, `{...}` fallback extraction, invalid content → `LLMValidationError`,
  429-then-200 retry, exhausted retries on 500, no-retry on 400, missing `choices`,
  empty API key, cost accumulation.
- **Integration** (`tests/integration/`): live ping, gated by `SLAYER_RUN_LLM_INTEGRATION=1`
  + a present API key.
- **CLI**: `slayer init` openrouter ping via `_llm_ping()` helper — monkeypatched in unit
  tests to keep them hermetic; real ping exercised by `slayer init` in real usage.

## Retro

Implemented as designed; hardened by live findings. Notes:
- **`response_format` was removed entirely** (2026-10-21): strict providers (OpenAI
  models via OpenRouter) validate the JSON schema we sent and reject pydantic's raw
  `model_json_schema()` output ("'required' is required to be an array"), while lax
  providers mostly ignored it anyway. The real enforcement mechanism is the
  prompt-embedded schema + `{...}` fallback extraction + pydantic validation — which
  has carried every live success. Schema strictness now lives where it can't break
  requests.
- The `{...}` fallback extraction is genuinely needed — models don't always honor
  schema requests and love wrapping JSON in prose/fences (they also invent their own
  JSON shape; embedding the schema in the prompt — done in `Agent.run_structured` — is
  the reliable path for weak models).
- Reasoning models can exhaust `max_tokens` before emitting any content
  (`content: null`, `finish_reason: "length"`). The client now raises an actionable
  `LLMError` for this case; default agent `max_tokens` is 16384.
- Cost field: OpenRouter includes top-level `cost` per completion; we only trust it when
  the value is numeric, else `cost_usd=None` and the call counts as unpriced.
