"""Re-entry campaign (D0007): the shortfall gate and the REPLAN phase."""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime

from slayer.agents import PlannerExtensionAgent
from slayer.agents.planner import render_strategy_plan_md
from slayer.analysis import AnalysisError
from slayer.cv import build_cv_protocol
from slayer.loop.engine import CampaignStop, ReentryDecision, RunContext
from slayer.loop.phases import _load_plan, notebook_digest, run_spend
from slayer.schemas import RoadmapExtension, RoadmapStep, StrategyPlan
from slayer.workspace import EventType, Journal


def rounds_done(journal: Journal) -> int:
    """Number of successful roadmap extensions (re-entry rounds started) in this run."""
    return sum(
        1
        for event in journal.read(type=EventType.ROADMAP_EXTENDED)
        if event.data.get("steps_added", 0) > 0
    )


def planner_stopped(journal: Journal) -> str | None:
    """The planner's stop reason if it ever declared the campaign exhausted."""
    for event in journal.read(type=EventType.ROADMAP_EXTENDED):
        if event.data.get("stop"):
            return event.data.get("stop_reason") or "planner declared the campaign exhausted"
    return None


def latest_position(journal: Journal) -> dict | None:
    """The most recent MONITOR position that carries a percentile, if any."""
    positions = journal.read(type=EventType.MONITOR_POSITION)
    if not positions or positions[-1].data.get("percentile") is None:
        return None
    return positions[-1].data


def _parse_deadline(raw: str | None) -> datetime | None:
    """Parse a Kaggle deadline string (assumed UTC when naive); None if unparsable."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _deadline_seconds_left(ctx: RunContext, now: datetime | None = None) -> float | None:
    """Seconds until the competition deadline (negative when passed); None if unknown."""
    path = ctx.run.workspace("reports") / "competition.json"
    if not path.exists():
        return None
    deadline = _parse_deadline(json.loads(path.read_text(encoding="utf-8")).get("deadline"))
    if deadline is None:
        return None
    return (deadline - (now or datetime.now(UTC))).total_seconds()


def decide_reentry(ctx: RunContext, now: datetime | None = None) -> ReentryDecision:
    """Deterministic re-entry gate: first failing check stops; all pass → re-enter."""
    done = rounds_done(ctx.journal)
    position = latest_position(ctx.journal)
    spend = run_spend(ctx)
    seconds_left = _deadline_seconds_left(ctx, now)
    stopped = planner_stopped(ctx.journal)

    if stopped:
        decision = ReentryDecision(False, f"planner stopped: {stopped}", done)
    elif done >= ctx.run.max_rounds:
        decision = ReentryDecision(False, f"round cap reached ({done}/{ctx.run.max_rounds})", done)
    elif position is None:
        decision = ReentryDecision(False, "no position signal — cannot judge shortfall", done)
    elif position.get("met"):
        decision = ReentryDecision(
            False, f"target met (top {position['percentile']}%)", done
        )
    elif spend >= ctx.run.budget_usd:
        decision = ReentryDecision(
            False, f"LLM budget exhausted (${spend:.2f} >= ${ctx.run.budget_usd})", done
        )
    elif seconds_left is not None and 0 < seconds_left < ctx.run.harness_timeout_s:
        decision = ReentryDecision(
            False, f"deadline too close ({seconds_left / 60:.0f} min left)", done
        )
    else:
        gap = position.get("gap")
        gap_text = f", gap {gap:+.5f}" if isinstance(gap, (int, float)) else ""
        decision = ReentryDecision(
            True,
            f"top {position['percentile']}% vs target {position['target_percentile']}%"
            f"{gap_text}",
            done + 1,
        )
    ctx.journal.append(
        agent="loop",
        type=EventType.REENTRY_DECISION,
        data={
            "reenter": decision.reenter,
            "reason": decision.reason,
            "round": decision.round,
            "rounds_done": done,
            "max_rounds": ctx.run.max_rounds,
            "percentile": position.get("percentile") if position else None,
            "gap": position.get("gap") if position else None,
            "spend_usd": round(spend, 6),
            "budget_usd": ctx.run.budget_usd,
            "deadline_seconds_left": seconds_left,
        },
        run=ctx.run.run_id,
    )
    return decision


def _failure_line(ctx: RunContext, candidate_id: str) -> str:
    """First line of a failed candidate's harness error (or a stuck-run note)."""
    path = ctx.run.workspace("models") / candidate_id / "cv_results.json"
    if not path.exists():
        return "no result (syntax error or never evaluated)"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unreadable result"
    error = (data.get("error") or "").strip()
    if error:
        return error.splitlines()[0][:140]
    return f"status={data.get('status')}"


def _fold_times(ctx: RunContext, candidate_id: str) -> list[float]:
    """Per-fold wall times recorded by the harness for one candidate."""
    path = ctx.run.workspace("models") / candidate_id / "cv_results.json"
    if not path.exists():
        return []
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("fold_times") or [])
    except json.JSONDecodeError:
        return []


def history_stats(ctx: RunContext) -> dict:
    """Deterministic summary stats of what has been tried: CV spread, noise, fold time."""
    scored = [r for r in ctx.registry.list() if r.cv_score is not None]
    cvs = [r.cv_score for r in scored]
    stds = [r.cv_std for r in scored if r.cv_std is not None]
    times = [t for r in scored for t in _fold_times(ctx, r.id)]
    return {
        "scored": len(scored),
        "failed": sum(1 for r in ctx.registry.list() if r.status == "failed"),
        "cv_spread": round(max(cvs) - min(cvs), 6) if len(cvs) > 1 else None,
        "median_fold_std": round(statistics.median(stds), 6) if stds else None,
        "median_fold_s": round(statistics.median(times), 1) if times else None,
    }


def render_history_md(ctx: RunContext) -> str:
    """Every candidate as a markdown table row, with failure lines for failed ones."""
    records = ctx.registry.list()
    if not records:
        return "No candidates yet."
    lines = [
        "| id | roadmap step | family | status | cv ± std | lb | notes / failure |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        cv = f"{r.cv_score:.5f} ± {r.cv_std or 0:.5f}" if r.cv_score is not None else "-"
        lb = "-"
        if r.lb_score is not None:
            mock = (r.submission_id or "").startswith("mock-")
            lb = f"{r.lb_score:.5f}{' (mock)' if mock else ''}"
        if r.status == "failed":
            note = _failure_line(ctx, r.id)
        elif r.status == "proposed":
            note = f"never completed ({_failure_line(ctx, r.id)})"
        else:
            note = (r.notes or "")[:80]
        note = note.replace("|", "/")
        step = str(r.params.get("roadmap_step", "-")).replace("|", "/")
        lines.append(
            f"| {r.id} | {step} | {r.model_type} | {r.status} | {cv} | {lb} | {note} |"
        )
    return "\n".join(lines)


def render_position_md(position: dict, stats: dict) -> str:
    """Position, gap, and the gap-vs-spread comparison as plain bullet lines."""
    basis = {
        "listed": "our listed leaderboard entry",
        "lb": "our best real LB score inserted into the field (not listed)",
        "cv": "our best CV score inserted into the field (no real LB score yet)",
    }.get(position.get("basis", ""), "unknown basis")
    lines = [
        f"- position: ~rank {position.get('rank')}/{position.get('total')} "
        f"(top {position.get('percentile')}%), target top {position.get('target_percentile')}% "
        f"— basis: {basis}",
        f"- our best score: {position.get('best_score')}",
        f"- score at the target line: {position.get('target_score')}",
        f"- gap to the target line: {position.get('gap')} (positive = short)",
        f"- spread of all scored CVs so far: {stats.get('cv_spread')}; "
        f"median fold std: {stats.get('median_fold_std')}",
    ]
    gap, spread = position.get("gap"), stats.get("cv_spread")
    if isinstance(gap, (int, float)) and isinstance(spread, (int, float)) and spread > 0:
        lines.append(f"- gap / spread ratio: {gap / spread:.1f}x")
    return "\n".join(lines)


def merge_extension(
    plan: StrategyPlan, extension: RoadmapExtension, round_number: int
) -> tuple[StrategyPlan, list[str]]:
    """Append the extension's steps (unique titles, renumbered priorities) and guidance."""
    titles = {step.title for step in plan.roadmap}
    next_priority = max(step.priority for step in plan.roadmap) + 1
    added: list[RoadmapStep] = []
    for step in extension.steps:
        title = step.title.strip() or f"Round {round_number} step"
        if title in titles:
            title = f"{title} (round {round_number})"
        suffix = 2
        base = title
        while title in titles:
            title = f"{base} #{suffix}"
            suffix += 1
        titles.add(title)
        added.append(step.model_copy(update={"title": title, "priority": next_priority}))
        next_priority += 1
    guidance = list(plan.guidance)
    for item in extension.new_guidance:
        if item not in guidance:
            guidance.append(item)
    merged = plan.model_copy(
        update={"roadmap": [*plan.roadmap, *added], "guidance": guidance}
    )
    return merged, [step.title for step in added]


def _constraints_md(ctx: RunContext, round_number: int, stats: dict) -> str:
    """Round budget/time constraints for the planner."""
    spend = run_spend(ctx)
    seconds_left = _deadline_seconds_left(ctx)
    if seconds_left is None:
        deadline = "unknown"
    elif seconds_left <= 0:
        deadline = "passed (late submissions are scored but not ranked)"
    else:
        deadline = f"{seconds_left / 3600:.1f} hours left"
    return "\n".join(
        [
            f"- round {round_number} of at most {ctx.run.max_rounds}",
            f"- LLM budget remaining: ${max(ctx.run.budget_usd - spend, 0):.2f} "
            f"of ${ctx.run.budget_usd:.2f}",
            f"- harness time budget per candidate: {ctx.run.harness_timeout_s}s "
            "(all folds + final fit)",
            f"- observed median fold time so far: {stats.get('median_fold_s')}s",
            f"- candidates scored / failed so far: {stats['scored']} / {stats['failed']}",
            f"- competition deadline: {deadline}",
        ]
    )


def replan_phase(ctx: RunContext) -> None:
    """REPLAN: the planner extends the roadmap from history + gap, or stops the campaign."""
    if ctx.llm is None or ctx.agent_configs is None:
        raise AnalysisError("replan requires the LLM client — check OPENROUTER_API_KEY")
    plan, competition = _load_plan(ctx)
    build_cv_protocol(competition.get("evaluation_metric") or "", plan.cv)
    round_number = rounds_done(ctx.journal) + 1
    position = latest_position(ctx.journal) or {}
    stats = history_stats(ctx)

    planner = PlannerExtensionAgent(
        llm=ctx.llm,
        config=ctx.agent_configs.planner,
        journal=ctx.journal,
        run_id=ctx.run.run_id,
    )
    extension = planner.extend(
        strategy_md=render_strategy_plan_md(plan),
        history_md=render_history_md(ctx),
        position_md=render_position_md(position, stats),
        constraints_md=_constraints_md(ctx, round_number, stats),
        round_number=round_number,
        notebook_digest=notebook_digest(ctx),
    )
    reports_dir = ctx.run.workspace("reports")
    (reports_dir / f"roadmap_round_{round_number}.json").write_text(
        extension.model_dump_json(indent=2), encoding="utf-8"
    )
    ctx.say(f"  assessment: {extension.assessment[:200]}")

    if extension.stop:
        reason = extension.stop_reason or "no credible hypothesis left"
        ctx.journal.append(
            agent="planner",
            type=EventType.ROADMAP_EXTENDED,
            data={
                "round": round_number,
                "steps_added": 0,
                "stop": True,
                "stop_reason": reason,
                "assessment": extension.assessment[:400],
            },
            run=ctx.run.run_id,
        )
        raise CampaignStop(reason)

    merged, titles = merge_extension(plan, extension, round_number)
    (reports_dir / "strategy_plan.json").write_text(
        merged.model_dump_json(indent=2), encoding="utf-8"
    )
    (reports_dir / "strategy_plan.md").write_text(
        render_strategy_plan_md(merged), encoding="utf-8"
    )
    ctx.journal.append(
        agent="planner",
        type=EventType.ROADMAP_EXTENDED,
        data={
            "round": round_number,
            "steps_added": len(titles),
            "titles": titles,
            "new_guidance": len(merged.guidance) - len(plan.guidance),
            "stop": False,
            "assessment": extension.assessment[:400],
        },
        run=ctx.run.run_id,
    )
    ctx.say(f"  round {round_number}: {len(titles)} new roadmap steps — " + "; ".join(titles))
