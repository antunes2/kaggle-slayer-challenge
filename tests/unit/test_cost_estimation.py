
import httpx
import pytest

from slayer.llm import LLMClient


def _routing_transport(completion_payload: dict) -> httpx.MockTransport:
    """Serve a price catalog on GET /models and a completion on POST /chat/completions."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "test/model",
                            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                        }
                    ]
                },
            )
        return httpx.Response(200, json=completion_payload)

    return httpx.MockTransport(handler)


def test_missing_cost_is_estimated_from_catalog_prices() -> None:
    """Without a provider cost, usage x catalog prices yields an estimated cost."""
    payload = {
        "model": "test/model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    client = LLMClient(
        api_key="test-key", transport=_routing_transport(payload), retry_wait_s=0.0
    )
    response = client.complete([{"role": "user", "content": "hi"}], "test/model")
    assert response.cost_usd == pytest.approx(100 * 0.000001 + 50 * 0.000002)
    assert response.cost_estimated is True
    assert client.cost.total_usd == pytest.approx(0.0002)
    assert client.cost.priced_calls == 1


def test_provider_cost_beats_estimation() -> None:
    """A real provider cost is used as-is and not flagged estimated."""
    payload = {
        "model": "test/model",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "cost": 0.001,
    }
    client = LLMClient(
        api_key="test-key", transport=_routing_transport(payload), retry_wait_s=0.0
    )
    response = client.complete([{"role": "user", "content": "hi"}], "test/model")
    assert response.cost_usd == 0.001
    assert response.cost_estimated is False


def test_no_catalog_no_cost_stays_none() -> None:
    """With no catalog reachable and no cost, the call stays unpriced."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(500, json={})
        return httpx.Response(
            200,
            json={
                "model": "unknown/model",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    client = LLMClient(
        api_key="test-key", transport=httpx.MockTransport(handler), retry_wait_s=0.0
    )
    response = client.complete([{"role": "user", "content": "hi"}], "unknown/model")
    assert response.cost_usd is None
    assert client.cost.priced_calls == 0
