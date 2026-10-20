from pathlib import Path

import typer

from slayer._version import __version__
from slayer.config import (
    AGENT_NAMES,
    DEFAULT_MODEL,
    load_agents_config,
    load_secrets,
    load_slayer_config,
)
from slayer.kaggle import KaggleClient, KaggleError
from slayer.llm import LLMClient, LLMError

app = typer.Typer(add_completion=False, help="Slayer — a team of agents that competes on Kaggle.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"slayer {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Slayer — a team of agents that competes on Kaggle."""


def _mask(value: str) -> str:
    if not value:
        return "MISSING"
    if len(value) <= 8:
        return "set"
    return f"{value[:4]}...{value[-4:]} ({len(value)} chars)"


def _llm_ping(
    api_key: str, base_url: str, timeout_s: int, max_retries: int, model: str
) -> tuple[bool, str]:
    if not api_key:
        return False, "skipped (no API key)"
    try:
        client = LLMClient(
            api_key=api_key, base_url=base_url, timeout_s=timeout_s, max_retries=max_retries
        )
        response = client.ping(model)
        snippet = response.content.strip().replace("\n", " ")[:40]
        return True, f"ok (model={model}, reply={snippet!r})"
    except LLMError as exc:
        return False, f"failed (model={model}): {exc}"


def _kaggle_ping(token: str) -> tuple[bool, str]:
    if not token:
        return False, "skipped (no token)"
    try:
        client = KaggleClient.from_token(token)
        username = client.ping()
        return True, f"ok (user={username})"
    except KaggleError as exc:
        return False, f"failed: {exc}"


@app.command()
def init(
    config_dir: Path = typer.Option(
        Path("config"), "--config-dir", help="Directory holding slayer.yaml and agents.yaml."
    ),
    env_file: Path = typer.Option(
        Path(".env"), "--env-file", help="Env file holding API credentials."
    ),
) -> None:
    """Validate configuration files, secrets, and API connectivity."""
    typer.secho("Slayer init — environment check", bold=True)
    ok = True

    typer.echo("")
    typer.echo("Config")
    cfg = None
    slayer_path = config_dir / "slayer.yaml"
    try:
        if slayer_path.exists():
            cfg = load_slayer_config(slayer_path)
            typer.secho(f"  {slayer_path}: loaded", fg="green")
        else:
            cfg = load_slayer_config(slayer_path)
            typer.secho(f"  {slayer_path}: not found, using built-in defaults", fg="yellow")
        d = cfg.defaults
        typer.echo(
            f"    target=top-{d.target_percentile}%  budget=${d.budget_usd}  "
            f"submissions/day={d.submissions_per_day}  autonomy={d.autonomy}  dry_run={d.dry_run}"
        )
    except Exception as exc:
        typer.secho(f"  {slayer_path}: INVALID — {exc}", fg="red")
        ok = False

    agents_path = config_dir / "agents.yaml"
    try:
        if agents_path.exists():
            agents = load_agents_config(agents_path)
            typer.secho(f"  {agents_path}: loaded", fg="green")
        else:
            agents = load_agents_config(agents_path)
            typer.secho(f"  {agents_path}: not found, using built-in defaults", fg="yellow")
        for name in AGENT_NAMES:
            agent_cfg = getattr(agents, name)
            typer.echo(
                f"    {name:<14} model={agent_cfg.model}  temperature={agent_cfg.temperature}"
            )
    except Exception as exc:
        typer.secho(f"  {agents_path}: INVALID — {exc}", fg="red")
        ok = False

    typer.echo("")
    typer.echo("Secrets")
    if not env_file.exists():
        typer.secho(f"  ({env_file} not found; relying on process environment)", fg="yellow")
    secrets = load_secrets(env_file)
    for label, value in (
        ("OPENROUTER_API_KEY", secrets.openrouter_api_key),
        ("KAGGLE_API_TOKEN", secrets.kaggle_api_token),
    ):
        if value:
            typer.secho(f"  {label}: {_mask(value)}", fg="green")
        else:
            typer.secho(f"  {label}: MISSING", fg="red")
            ok = False

    typer.echo("")
    typer.echo("API pings")
    if secrets.openrouter_api_key and cfg is not None:
        ok_ping, detail = _llm_ping(
            secrets.openrouter_api_key,
            cfg.llm.base_url,
            cfg.llm.timeout_s,
            cfg.llm.max_retries,
            DEFAULT_MODEL,
        )
        typer.secho(f"  openrouter: {detail}", fg="green" if ok_ping else "red")
        ok = ok and ok_ping
    else:
        typer.secho("  openrouter: skipped (no API key or invalid config)", fg="yellow")
    if secrets.kaggle_api_token:
        ok_ping, detail = _kaggle_ping(secrets.kaggle_api_token)
        typer.secho(f"  kaggle:     {detail}", fg="green" if ok_ping else "red")
        ok = ok and ok_ping
    else:
        typer.secho("  kaggle:     skipped (no token)", fg="yellow")

    typer.echo("")
    if ok:
        typer.secho("Environment ready.", fg="green", bold=True)
    else:
        typer.secho("Environment NOT ready — fix the items above.", fg="red", bold=True)
        raise typer.Exit(1)


def main() -> None:
    app()
