import json
import re

import httpx
import pytest
from pydantic import BaseModel

from slayer.llm import LLMClient, LLMError, LLMValidationError


class _Verdict(BaseModel):
    decision: str
    confidence: float


def _response_payload(content: str = "ok", cost: float | None = 0.0001) -> dict:
    return {
        "model": "test/model",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "cost": cost,
    }


def _client(handler, **kwargs) -> LLMClient:
    return LLMClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        retry_wait_s=0.0,
        **kwargs,
    )


def test_complete_returns_response_and_records_cost():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        assert payload["model"] == "test/model"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json=_response_payload())

    client = _client(handler)
    response = client.complete([{"role": "user", "content": "hi"}], "test/model")
    assert response.content == "ok"
    assert response.model == "test/model"
    assert response.usage.total_tokens == 7
    assert response.cost_usd == 0.0001
    report = client.cost
    assert report.calls == 1
    assert report.total_tokens == 7
    assert report.total_usd == pytest.approx(0.0001)
    assert report.priced_calls == 1


def test_cost_accumulates_across_calls():
    payloads = [_response_payload(cost=0.1), _response_payload(cost=0.2)]
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=payloads[len(calls) - 1])

    client = _client(handler)
    client.complete([{"role": "user", "content": "a"}], "test/model")
    client.complete([{"role": "user", "content": "b"}], "test/model")
    report = client.cost
    assert report.calls == 2
    assert report.total_tokens == 14
    assert report.total_usd == pytest.approx(0.3)


def test_unpriced_call_reports_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response_payload(cost=None))

    client = _client(handler)
    response = client.complete([{"role": "user", "content": "hi"}], "test/model")
    assert response.cost_usd is None
    assert client.cost.total_usd == 0.0
    assert client.cost.priced_calls == 0


def test_structured_output_validates_without_response_format():
    """Schema enforcement is prompt + validation; response_format is never sent."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.read())
        content = json.dumps({"decision": "approve", "confidence": 0.9})
        return httpx.Response(200, json=_response_payload(content))

    client = _client(handler)
    _, verdict = client.complete_structured(
        [{"role": "user", "content": "review"}], "test/model", _Verdict
    )
    assert verdict.decision == "approve"
    assert verdict.confidence == 0.9
    assert "response_format" not in seen["payload"]


def test_structured_output_falls_back_to_braces_extraction():
    def handler(request: httpx.Request) -> httpx.Response:
        verdict = json.dumps({"decision": "reject", "confidence": 0.4})
        content = f"Sure! Here you go:\n```json\n{verdict}\n```\nDone."
        return httpx.Response(200, json=_response_payload(content))

    client = _client(handler)
    _, verdict = client.complete_structured(
        [{"role": "user", "content": "review"}], "test/model", _Verdict
    )
    assert verdict.decision == "reject"


def test_structured_output_invalid_content_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response_payload("no json here at all"))

    client = _client(handler)
    with pytest.raises(LLMValidationError):
        client.complete_structured(
            [{"role": "user", "content": "review"}], "test/model", _Verdict
        )


def test_structured_output_schema_violation_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        content = json.dumps({"decision": "approve", "confidence": "high"})
        return httpx.Response(200, json=_response_payload(content))

    client = _client(handler)
    with pytest.raises(LLMValidationError) as exc_info:
        client.complete_structured(
            [{"role": "user", "content": "review"}], "test/model", _Verdict
        )
    assert exc_info.value.raw == json.dumps({"decision": "approve", "confidence": "high"})


def test_retry_on_429_honors_retry_after_header():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"error": {"message": "rate limited"}},
            )
        return httpx.Response(200, json=_response_payload())

    client = LLMClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        retry_wait_s=0.0,
        rate_limit_wait_s=0.0,
    )
    response = client.complete([{"role": "user", "content": "hi"}], "test/model")
    assert response.content == "ok"
    assert len(calls) == 2


def test_retry_on_429_then_success():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=_response_payload())

    client = _client(handler, max_retries=2)
    response = client.complete([{"role": "user", "content": "hi"}], "test/model")
    assert response.content == "ok"
    assert len(calls) == 2


def test_retries_exhausted_raises():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    client = _client(handler, max_retries=2)
    with pytest.raises(LLMError, match="HTTP 500"):
        client.complete([{"role": "user", "content": "hi"}], "test/model")
    assert len(calls) == 3


def test_no_retry_on_client_error():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    client = _client(handler, max_retries=3)
    with pytest.raises(LLMError, match="HTTP 400"):
        client.complete([{"role": "user", "content": "hi"}], "test/model")
    assert len(calls) == 1


def test_missing_choices_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x", "model": "test/model"})

    client = _client(handler)
    with pytest.raises(LLMError, match=re.escape("missing choices")):
        client.complete([{"role": "user", "content": "hi"}], "test/model")


def test_truncated_completion_raises_actionable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "test/model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 4000, "total_tokens": 4100},
                "cost": 0.0,
            },
        )

    client = _client(handler)
    with pytest.raises(LLMError, match="max_tokens"):
        client.complete([{"role": "user", "content": "hi"}], "test/model")


def test_empty_api_key_raises():
    with pytest.raises(LLMError, match="API key is empty"):
        LLMClient(api_key="")


def test_ping_sends_minimal_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.read())
        return httpx.Response(200, json=_response_payload())

    client = _client(handler)
    response = client.ping("test/model")
    assert response.content == "ok"
    assert seen["payload"]["max_tokens"] == 16
    assert seen["payload"]["temperature"] == 0.0
    assert "ok" in seen["payload"]["messages"][0]["content"]


def test_connection_blip_is_survived_with_its_own_retry_budget():
    """DNS/Wi-Fi failures retry beyond max_retries (own budget) and then succeed."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= 3:
            raise httpx.ConnectError("Temporary failure in name resolution")
        return httpx.Response(200, json=_response_payload())

    client = _client(handler, max_retries=1, connection_wait_s=0.0, connection_retries=5)
    response = client.complete([{"role": "user", "content": "hi"}], "test/model")
    assert response.content == "ok"
    assert len(calls) == 4


def test_persistent_connection_failure_raises_after_budget():
    """A connection that never comes back fails loudly after connection_retries."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("Temporary failure in name resolution")

    client = _client(handler, connection_wait_s=0.0, connection_retries=2)
    with pytest.raises(LLMError, match="connection error"):
        client.complete([{"role": "user", "content": "hi"}], "test/model")
    assert len(calls) == 3
