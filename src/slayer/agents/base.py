from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from slayer.config import AgentModelConfig
from slayer.llm import LLMClient, LLMResponse, Message
from slayer.workspace import EventType, Journal

T = TypeVar("T", bound=BaseModel)


class AgentError(Exception):
    """Raised when an agent cannot be constructed or run."""

    pass


class Agent:
    name: str = "agent"
    prompt_file: str = "agent"

    def __init__(
        self,
        llm: LLMClient,
        config: AgentModelConfig,
        journal: Journal,
        run_id: str,
        prompts_dir: Path | None = None,
    ) -> None:
        """Load the agent's prompt and store its clients and config."""
        self._llm = llm
        self._config = config
        self._journal = journal
        self._run_id = run_id
        self._prompts_dir = prompts_dir or Path("prompts")
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        """Read the agent's system prompt from its markdown file."""
        path = self._prompts_dir / f"{self.prompt_file}.md"
        if not path.exists():
            raise AgentError(f"prompt file missing: {path}")
        return path.read_text(encoding="utf-8")

    def run_structured(self, user_message: str, response_model: type[T]) -> tuple[LLMResponse, T]:
        """Run one schema-validated LLM call and journal its usage/cost."""
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        full_message = (
            f"{user_message}\n\n"
            "Your entire output must be valid JSON matching exactly this schema:\n"
            f"{schema}"
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=full_message),
        ]
        response, parsed = self._llm.complete_structured(
            messages,
            self._config.model,
            response_model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        self._journal.append(
            agent=self.name,
            type=EventType.AGENT_CALL,
            data={
                "model": response.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "cost_usd": response.cost_usd,
            },
            run=self._run_id,
        )
        return response, parsed
