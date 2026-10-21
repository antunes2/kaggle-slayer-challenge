from __future__ import annotations

import json

from slayer.agents import DataAnalystAgent, PlannerAgent
from slayer.agents.planner import render_strategy_plan_md
from slayer.analysis import AnalysisError, profile_data
from slayer.analysis.render import render_data_report_md
from slayer.cv import build_cv_protocol
from slayer.loop.data import data_collection_phase
from slayer.loop.engine import Handler, Phase, RunContext
from slayer.schemas import DataAnalystOutput
from slayer.workspace import EventType


def init_phase(ctx: RunContext) -> None:
    """INIT: resolve the competition, probe data access, and store its context."""
    info = ctx.kaggle.find_competition(ctx.run.competition)
    files = ctx.kaggle.list_files(ctx.run.competition)
    report_path = ctx.run.workspace("reports") / "competition.json"
    report_path.write_text(info.model_dump_json(indent=2), encoding="utf-8")
    ctx.journal.append(
        agent="loop",
        type=EventType.COMPETITION_CONTEXT,
        data={
            "ref": info.ref,
            "title": info.title,
            "deadline": info.deadline,
            "metric": info.evaluation_metric,
            "max_daily_submissions": info.max_daily_submissions,
            "data_files": [f.name for f in files],
        },
        run=ctx.run.run_id,
    )
    ctx.say(
        f"  competition: {info.title} "
        f"(metric={info.evaluation_metric}, quota={info.max_daily_submissions}/day)"
    )
    ctx.say(f"  data access ok ({len(files)} files — rules accepted)")


def analysis_phase(ctx: RunContext) -> None:
    """ANALYSIS: run the profiler, then the data analyst agent over its output."""
    if ctx.llm is None or ctx.agent_configs is None:
        raise AnalysisError("analysis requires the LLM client — check OPENROUTER_API_KEY")
    profile = profile_data(ctx.run)
    ctx.say(
        f"  profiled train {profile.train_rows:,}x{profile.train_cols}, "
        f"target={profile.target_column} (guess: {profile.problem_type_guess})"
    )
    competition_json = (ctx.run.workspace("reports") / "competition.json").read_text(
        encoding="utf-8"
    )
    profile_md = (ctx.run.workspace("eda") / "profile.md").read_text(encoding="utf-8")
    analyst = DataAnalystAgent(
        llm=ctx.llm,
        config=ctx.agent_configs.data_analyst,
        journal=ctx.journal,
        run_id=ctx.run.run_id,
    )
    output: DataAnalystOutput = analyst.analyze(
        profile_md=profile_md, competition_json=competition_json
    )
    reports_dir = ctx.run.workspace("reports")
    (reports_dir / "data_report.json").write_text(
        output.report.model_dump_json(indent=2), encoding="utf-8"
    )
    (reports_dir / "feature_plan.json").write_text(
        output.feature_plan.model_dump_json(indent=2), encoding="utf-8"
    )
    (reports_dir / "data_report.md").write_text(
        render_data_report_md(output), encoding="utf-8"
    )
    ctx.journal.append(
        agent="data_analyst",
        type=EventType.AGENT_OUTPUT,
        data={
            "problem_type": output.report.problem_type,
            "key_findings": len(output.report.key_findings),
            "feature_ideas": len(output.feature_plan.ideas),
        },
        run=ctx.run.run_id,
    )
    ctx.say(
        f"  data report: {output.report.problem_type}; "
        f"{len(output.feature_plan.ideas)} feature ideas"
    )


def planning_phase(ctx: RunContext) -> None:
    """PLANNING: produce the per-competition StrategyPlan from the analysis artifacts."""
    if ctx.llm is None or ctx.agent_configs is None:
        raise AnalysisError("planning requires the LLM client — check OPENROUTER_API_KEY")
    reports_dir = ctx.run.workspace("reports")
    competition_json = (reports_dir / "competition.json").read_text(encoding="utf-8")
    data_report_json = (reports_dir / "data_report.json").read_text(encoding="utf-8")
    feature_plan_json = (reports_dir / "feature_plan.json").read_text(encoding="utf-8")

    planner = PlannerAgent(
        llm=ctx.llm,
        config=ctx.agent_configs.planner,
        journal=ctx.journal,
        run_id=ctx.run.run_id,
    )
    plan = planner.plan(
        competition_json=competition_json,
        data_report_json=data_report_json,
        feature_plan_json=feature_plan_json,
    )

    evaluation_metric = json.loads(competition_json).get("evaluation_metric") or ""
    build_cv_protocol(evaluation_metric, plan.cv)

    (reports_dir / "strategy_plan.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    (reports_dir / "strategy_plan.md").write_text(
        render_strategy_plan_md(plan), encoding="utf-8"
    )
    ctx.journal.append(
        agent="planner",
        type=EventType.AGENT_OUTPUT,
        data={
            "problem_type": plan.problem_type,
            "roadmap_steps": len(plan.roadmap),
            "k_folds": plan.cv.k_folds,
            "min_cv_improvement": plan.min_cv_improvement,
        },
        run=ctx.run.run_id,
    )
    ctx.say(
        f"  strategy: {plan.problem_type}, {len(plan.roadmap)} roadmap steps, "
        f"{plan.cv.k_folds}-fold CV, gate >= {plan.min_cv_improvement}"
    )


HANDLERS: dict[Phase, Handler] = {
    Phase.INIT: init_phase,
    Phase.DATA_COLLECTION: data_collection_phase,
    Phase.ANALYSIS: analysis_phase,
    Phase.PLANNING: planning_phase,
}
