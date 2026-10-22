import json
import time
from collections.abc import Sequence
from typing import TypedDict, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Message(TypedDict):
    role: str
    content: str


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: LLMUsage = LLMUsage()
    cost_usd: float | None = None


class CostReport(BaseModel):
    calls: int = 0
    total_tokens: int = 0
    total_usd: float = 0.0
    priced_calls: int = 0


class LLMError(Exception):
    """Base error for LLM client failures."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """Wrap a failure message with an optional HTTP status code."""
        super().__init__(message)
        self.status_code = status_code


class LLMValidationError(LLMError):
    """Raised when structured output fails schema validation."""

    def __init__(self, message: str, raw: str) -> None:
        """Keep the raw model output for debugging."""
        super().__init__(message)
        self.raw = raw


def _retryable(status_code: int) -> bool:
    """Whether an HTTP status code is transient and worth retrying."""
    return status_code == 408 or status_code == 429 or status_code >= 500


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: int = 180,
        max_retries: int = 3,
        retry_wait_s: float = 0.5,
        rate_limit_wait_s: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Configure the client with credentials, transport, and retry policy."""
        if not api_key:
            raise LLMError("OpenRouter API key is empty")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_wait_s = retry_wait_s
        self._rate_limit_wait_s = rate_limit_wait_s
        self._transport = transport
        self._calls = 0
        self._tokens = 0
        self._usd = 0.0
        self._priced_calls = 0

    @property
    def cost(self) -> CostReport:
        """Accumulated usage and cost across all calls made by this client."""
        return CostReport(
            calls=self._calls,
            total_tokens=self._tokens,
            total_usd=self._usd,
            priced_calls=self._priced_calls,
        )

    def complete(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Run one chat completion (schema enforcement lives in the prompt + validation)."""
        payload: dict = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._request(payload)
        return self._parse(data, fallback_model=model)

    def complete_structured(
        self,
        messages: Sequence[Message],
        model: str,
        response_model: type[T],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> tuple[LLMResponse, T]:
        """Complete a request and validate the reply against a pydantic model."""
        response = self.complete(
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = self._validate(response.content, response_model)
        return response, parsed

    def ping(self, model: str) -> LLMResponse:
        """Send a tiny 'reply ok' request to verify connectivity and auth."""
        return self.complete(
            [Message(role="user", content="Reply with exactly: ok")],
            model,
            temperature=0.0,
            max_tokens=16,
        )

    def _request(self, payload: dict) -> dict:
        """POST the payload with retries and backoff; returns the parsed JSON."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "slayer",
        }
        last_error: LLMError | None = None
        for attempt in range(self._max_retries + 1):
            retry_after_s: float | None = None
            try:
                with httpx.Client(timeout=self._timeout_s, transport=self._transport) as client:
                    r = client.post(
                        f"{self._base_url}/chat/completions", headers=headers, json=payload
                    )
            except httpx.HTTPError as exc:
                last_error = LLMError(f"connection error: {exc}")
            else:
                if r.status_code == 200:
                    return r.json()
                snippet = r.text[:200]
                last_error = LLMError(
                    f"HTTP {r.status_code}: {snippet}", status_code=r.status_code
                )
                if r.status_code == 429:
                    try:
                        retry_after_s = float(r.headers.get("Retry-After", ""))
                    except ValueError:
                        retry_after_s = None
                if not _retryable(r.status_code):
                    raise last_error
            if attempt < self._max_retries:
                wait = self._retry_wait_s * (2**attempt)
                if last_error is not None and last_error.status_code == 429:
                    wait = retry_after_s if retry_after_s is not None else max(
                        wait, self._rate_limit_wait_s * (attempt + 1)
                    )
                if wait > 0:
                    time.sleep(wait)
        assert last_error is not None
        raise last_error

    def _parse(self, data: dict, *, fallback_model: str) -> LLMResponse:
        """Convert an OpenRouter response into an LLMResponse, tracking usage."""
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"response missing choices: {json.dumps(data)[:200]}")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        finish_reason = choices[0].get("finish_reason")
        if not content:
            raise LLMError(
                f"empty completion content (finish_reason={finish_reason}); the model likely "
                "exhausted max_tokens before emitting output (reasoning models burn tokens "
                "on reasoning first) — raise max_tokens for this agent"
            )
        usage_data = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        cost = data.get("cost")
        if not isinstance(cost, (int, float)):
            cost = None
        self._calls += 1
        self._tokens += usage.total_tokens
        if cost is not None:
            self._usd += float(cost)
            self._priced_calls += 1
        return LLMResponse(
            content=content,
            model=data.get("model", fallback_model),
            usage=usage,
            cost_usd=cost,
        )

    @staticmethod
    def _extract_json(content: str) -> dict:
        """Parse JSON from a reply, falling back to the outermost {...} block."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end > start:
                return json.loads(content[start : end + 1])
            raise

    @classmethod
    def _validate(cls, content: str, response_model: type[T]) -> T:
        """Validate raw content against a pydantic model, raising a clear error."""
        try:
            data = cls._extract_json(content)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMValidationError(
                f"structured output failed validation for {response_model.__name__}: {exc}",
                raw=content,
            ) from exc
