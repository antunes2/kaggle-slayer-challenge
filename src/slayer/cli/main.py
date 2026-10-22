"""Slayer CLI: init, run, status, journal."""

import json
from pathlib import Path

import typer

from slayer._version import __version__
from slayer.analysis import AnalysisError
from slayer.config import (
    AGENT_NAMES,
    DEFAULT_MODEL,
    DefaultsConfig,
    load_agents_config,
    load_secrets,
    load_slayer_config,
)
from slayer.cv import CandidateRunError, MetricError
from slayer.kaggle import KaggleClient, KaggleError
from slayer.llm import LLMClient, LLMError
from slayer.loop import DataCollectionError, LoopEngine, RunContext
from slayer.submissions import KaggleSink, MockSink
from slayer.workspace import EventType, RunStatus, RunStore

app = typer.Typer(add_completion=False, help="Slayer — a team of agents that competes on Kaggle.")


def _version_callback(value: bool) -> None:
    """Print the version and exit when --version is passed."""
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
    """Mask a secret for display, showing only head/tail and length."""
    if not value:
        return "MISSING"
    if len(value) <= 8:
        return "set"
    return f"{value[:4]}...{value[-4:]} ({len(value)} chars)"


def _llm_ping(
    api_key: str, base_url: str, timeout_s: int, max_retries: int, model: str
) -> tuple[bool, str]:
    """Ping OpenRouter with a tiny request; returns (ok, detail)."""
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
    """Authenticate against Kaggle and return the username; returns (ok, detail)."""
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


@app.command()
def run(
    competition: str = typer.Argument(
        ..., help="Kaggle competition ref, e.g. playground-series-s5e10"
    ),
    config_dir: Path = typer.Option(Path("config"), "--config-dir"),
    env_file: Path = typer.Option(Path(".env"), "--env-file"),
    budget_usd: float = typer.Option(None, help="LLM spend cap for this run (default from config)"),
    autonomy: str = typer.Option(None, help="approve-first | agentic (default from config)"),
    dry_run: bool = typer.Option(
        None, "--dry-run/--live", help="Mock submissions (default from config)"
    ),
    refresh_data: bool = typer.Option(
        None, "--refresh-data/--use-cache", help="Force a data redownload (default from config)"
    ),
    harness_timeout_s: int = typer.Option(
        None, "--harness-timeout-s", help="Per-candidate CV time budget in seconds (default 600)"
    ),
    submissions_per_day: int = typer.Option(
        None, help="Max Kaggle submissions per day (default from config)"
    ),
) -> None:
    """Start the orchestration loop on a competition."""
    try:
        cfg = load_slayer_config(config_dir / "slayer.yaml")
    except Exception as exc:
        typer.secho(f"invalid slayer config: {exc}", fg="red")
        raise typer.Exit(1) from exc
    overrides = {
        key: value
        for key, value in {
            "budget_usd": budget_usd,
            "autonomy": autonomy,
            "dry_run": dry_run,
            "refresh_data": refresh_data,
            "harness_timeout_s": harness_timeout_s,
            "submissions_per_day": submissions_per_day,
        }.items()
        if value is not None
    }
    try:
        defaults = DefaultsConfig(**{**cfg.defaults.model_dump(), **overrides})
    except Exception as exc:
        typer.secho(f"invalid option combination: {exc}", fg="red")
        raise typer.Exit(1) from exc

    secrets = load_secrets(env_file)
    if not secrets.kaggle_api_token or not secrets.openrouter_api_key:
        typer.secho("KAGGLE_API_TOKEN or OPENROUTER_API_KEY missing — check your env.", fg="red")
        raise typer.Exit(1)

    store = RunStore(cfg.paths.runs_dir)
    run = store.create(competition, defaults)
    typer.secho(f"run created: {run.run_id}", bold=True)
    typer.echo(f"  workspace: {run.root}")
    _execute_run(config_dir=config_dir, env_file=env_file, run=run)


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run id to resume"),
    config_dir: Path = typer.Option(Path("config"), "--config-dir"),
    env_file: Path = typer.Option(Path(".env"), "--env-file"),
) -> None:
    """Resume a paused or failed run, skipping completed phases."""
    cfg = load_slayer_config(config_dir / "slayer.yaml")
    store = RunStore(cfg.paths.runs_dir)
    try:
        run = store.get(run_id)
    except KeyError as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1) from exc
    typer.secho(f"resuming run {run.run_id} (status: {run.status})", bold=True)
    _execute_run(config_dir=config_dir, env_file=env_file, run=run)


def _execute_run(config_dir: Path, env_file: Path, run) -> None:
    """Build all clients for an existing run and execute the loop to completion."""
    cfg = load_slayer_config(config_dir / "slayer.yaml")
    secrets = load_secrets(env_file)
    if not secrets.kaggle_api_token:
        typer.secho("KAGGLE_API_TOKEN missing — cannot run without Kaggle access.", fg="red")
        raise typer.Exit(1)
    if not secrets.openrouter_api_key:
        typer.secho("OPENROUTER_API_KEY missing — agents cannot run.", fg="red")
        raise typer.Exit(1)
    try:
        kaggle = KaggleClient.from_token(secrets.kaggle_api_token)
    except KaggleError as exc:
        typer.secho(f"kaggle authentication failed: {exc}", fg="red")
        raise typer.Exit(1) from exc
    try:
        agents_cfg = load_agents_config(config_dir / "agents.yaml")
    except Exception as exc:
        typer.secho(f"invalid agents config: {exc}", fg="red")
        raise typer.Exit(1) from exc
    llm = LLMClient(
        api_key=secrets.openrouter_api_key,
        base_url=cfg.llm.base_url,
        timeout_s=cfg.llm.timeout_s,
        max_retries=cfg.llm.max_retries,
    )
    store = RunStore(cfg.paths.runs_dir)
    ctx = RunContext(
        run=run,
        journal=store.journal(run),
        registry=store.registry(run),
        kaggle=kaggle,
        store=store,
        reporter=lambda message: typer.echo(f"  {message}"),
        llm=llm,
        agent_configs=agents_cfg,
        sink=MockSink() if run.dry_run else KaggleSink(kaggle),
        approver=lambda question: typer.confirm(question, default=False),
    )
    try:
        final = LoopEngine(ctx).start()
    except (KaggleError, DataCollectionError, AnalysisError, LLMError, MetricError,
            CandidateRunError) as exc:
        typer.secho(f"run failed: {exc}", fg="red", bold=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        typer.secho(f"run failed unexpectedly: {exc}", fg="red", bold=True)
        typer.echo(f"details: {run.root / 'journal.jsonl'}")
        raise typer.Exit(1) from exc
    if final.status == RunStatus.FINISHED:
        typer.secho(f"run finished: {run.run_id}", fg="green", bold=True)
    else:
        typer.secho(f"run {final.status}: {run.run_id}", fg="yellow", bold=True)


@app.command()
def status(
    run_id: str = typer.Argument(None, help="Run id — omit to list all runs"),
    config_dir: Path = typer.Option(Path("config"), "--config-dir"),
) -> None:
    """Show run status: phase history, candidates, and the latest CV score."""
    cfg = load_slayer_config(config_dir / "slayer.yaml")
    store = RunStore(cfg.paths.runs_dir)
    if run_id is None:
        runs = store.list()
        if not runs:
            typer.echo("No runs yet — start one with: slayer run <competition>")
            return
        for run in runs:
            typer.echo(f"{run.run_id}  {run.status:<8}  {run.competition}")
        return
    try:
        run = store.get(run_id)
    except KeyError as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1) from exc
    journal = store.journal(run)
    phases = [e.data.get("phase") for e in journal.read(type=EventType.PHASE_ENTER)]
    typer.secho(f"run {run.run_id}", bold=True)
    typer.echo(f"  competition: {run.competition}")
    typer.echo(f"  status:      {run.status}")
    typer.echo(f"  workspace:   {run.root}")
    if phases:
        typer.echo(f"  phases:      {' -> '.join(str(p) for p in phases if p)}")
    paused = journal.read(type=EventType.LOOP_PAUSED)
    if paused:
        typer.echo(f"  paused:      {paused[-1].data.get('reason')}")
    agent_calls = journal.read(type=EventType.AGENT_CALL)
    priced = [
        event.data["cost_usd"]
        for event in agent_calls
        if event.data.get("cost_usd") is not None
    ]
    spend = sum(priced)
    pricing_note = "" if priced else " (free/unpriced model)"
    typer.echo(
        f"  llm spend:   ${spend:.4f} / ${run.budget_usd:.2f} budget "
        f"({len(agent_calls)} calls{pricing_note})"
    )
    records = store.registry(run).list()
    if records:
        by_status: dict[str, int] = {}
        for record in records:
            by_status[record.status] = by_status.get(record.status, 0) + 1
        summary = ", ".join(f"{count} {name}" for name, count in sorted(by_status.items()))
        typer.echo(f"  candidates:  {summary}")
        scored = [r for r in records if r.cv_score is not None]
        if scored:
            latest = scored[-1]
            typer.echo(f"  latest cv:   {latest.id} = {latest.cv_score} ({latest.model_type})")


@app.command()
def journal(
    run_id: str = typer.Argument(..., help="Run id"),
    limit: int = typer.Option(20, "--limit", "-l", help="Show the last N events"),
    config_dir: Path = typer.Option(Path("config"), "--config-dir"),
) -> None:
    """Show the run's journal timeline (last N events)."""
    cfg = load_slayer_config(config_dir / "slayer.yaml")
    store = RunStore(cfg.paths.runs_dir)
    try:
        run = store.get(run_id)
    except KeyError as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1) from exc
    events = store.journal(run).read(limit=limit)
    if not events:
        typer.echo("(no events)")
        return
    for event in events:
        summary = json.dumps(event.data, sort_keys=True, default=str)
        typer.echo(f"{event.ts} [{event.agent}] {event.type} {summary[:120]}")


def main() -> None:
    """Console script entry point."""
    app()
