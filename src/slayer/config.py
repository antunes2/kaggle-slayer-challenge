from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

AGENT_NAMES = ("orchestrator", "planner", "data_analyst", "ml_researcher", "reviewer")
DEFAULT_MODEL = "google/gemini-2.5-flash"

AutonomyPolicy = Literal["approve-first", "agentic"]


class DefaultsConfig(BaseModel):
    """Run defaults set in slayer.yaml and overridable via CLI flags."""

    target_percentile: int = Field(default=20, ge=1, le=100)
    budget_usd: float = Field(default=10.0, gt=0)
    submissions_per_day: int = Field(default=4, ge=1, le=10)
    autonomy: AutonomyPolicy = "approve-first"
    dry_run: bool = True
    refresh_data: bool = False


class PathsConfig(BaseModel):
    """Filesystem paths used by the tool."""

    runs_dir: Path = Path("runs")


class LLMClientConfig(BaseModel):
    """OpenRouter connection settings."""

    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: int = Field(default=180, gt=0)
    max_retries: int = Field(default=3, ge=0)


class SlayerConfig(BaseModel):
    """Root of the slayer.yaml configuration."""

    defaults: DefaultsConfig = DefaultsConfig()
    paths: PathsConfig = PathsConfig()
    llm: LLMClientConfig = LLMClientConfig()


class AgentModelConfig(BaseModel):
    """Model routing entry for one agent."""

    model: str
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=16384, gt=0)


class AgentsConfig(BaseModel):
    """Per-agent model routing resolved from agents.yaml."""

    model_config = ConfigDict(extra="forbid")

    orchestrator: AgentModelConfig
    planner: AgentModelConfig
    data_analyst: AgentModelConfig
    ml_researcher: AgentModelConfig
    reviewer: AgentModelConfig


class Secrets(BaseSettings):
    """API credentials loaded from the environment and .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    openrouter_api_key: str = ""
    kaggle_api_token: str = ""


def _read_yaml_mapping(path: Path) -> dict:
    """Read a YAML file as a mapping; missing or empty files yield {}."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_slayer_config(path: Path | None = None) -> SlayerConfig:
    """Load and validate slayer.yaml, falling back to built-in defaults."""
    data = _read_yaml_mapping(path or Path("config/slayer.yaml"))
    return SlayerConfig.model_validate(data)


def load_agents_config(path: Path | None = None) -> AgentsConfig:
    """Load agents.yaml, merging per-agent entries over the defaults block."""
    data = _read_yaml_mapping(path or Path("config/agents.yaml"))
    defaults = dict(data.get("defaults") or {})
    defaults.setdefault("model", DEFAULT_MODEL)
    unknown = set(data) - set(AGENT_NAMES) - {"defaults"}
    if unknown:
        raise ValueError(f"unknown agent entries in agents config: {sorted(unknown)}")
    merged = {name: {**defaults, **(data.get(name) or {})} for name in AGENT_NAMES}
    return AgentsConfig.model_validate(merged)


def load_secrets(env_file: Path | None = None) -> Secrets:
    """Load secrets from an explicit env file or the default .env."""
    if env_file is None:
        return Secrets()
    return Secrets(_env_file=env_file)
