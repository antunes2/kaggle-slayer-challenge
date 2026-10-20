from __future__ import annotations

from slayer.agents import DataAnalystAgent
from slayer.analysis import AnalysisError, profile_data
from slayer.analysis.render import render_data_report_md
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


HANDLERS: dict[Phase, Handler] = {
    Phase.INIT: init_phase,
    Phase.DATA_COLLECTION: data_collection_phase,
    Phase.ANALYSIS: analysis_phase,
}
