import os

import pytest

from slayer.config import DEFAULT_MODEL, load_secrets
from slayer.llm import LLMClient

pytestmark = pytest.mark.skipif(
    os.environ.get("SLAYER_RUN_LLM_INTEGRATION") != "1",
    reason="live OpenRouter tests: set SLAYER_RUN_LLM_INTEGRATION=1",
)


def test_live_ping():
    secrets = load_secrets()
    if not secrets.openrouter_api_key:
        pytest.skip("no OPENROUTER_API_KEY configured")
    client = LLMClient(api_key=secrets.openrouter_api_key)
    response = client.ping(DEFAULT_MODEL)
    assert len(response.content) > 0
    assert client.cost.calls == 1
