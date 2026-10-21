from __future__ import annotations

import json

from slayer.agents import DataAnalystAgent, MLResearcherAgent, PlannerAgent
from slayer.agents.planner import render_strategy_plan_md
from slayer.analysis import AnalysisError, profile_data
from slayer.analysis.render import render_data_report_md
from slayer.cv import CandidateRunError, build_cv_protocol, run_candidate
from slayer.loop.data import data_collection_phase
from slayer.loop.engine import Handler, Phase, RunContext
from slayer.schemas import DataAnalystOutput, StrategyPlan
from slayer.workspace import EventType, ModelCandidate


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


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences a model may wrap around the code."""
    text = code.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip() + "\n"


def baseline_phase(ctx: RunContext) -> None:
    """BASELINE: researcher writes the first pipeline; harness scores it; registry records it."""
    if ctx.llm is None or ctx.agent_configs is None:
        raise AnalysisError("baseline requires the LLM client — check OPENROUTER_API_KEY")
    reports_dir = ctx.run.workspace("reports")
    competition_json = (reports_dir / "competition.json").read_text(encoding="utf-8")
    data_report_json = (reports_dir / "data_report.json").read_text(encoding="utf-8")
    feature_plan_json = (reports_dir / "feature_plan.json").read_text(encoding="utf-8")
    strategy_json = (reports_dir / "strategy_plan.json").read_text(encoding="utf-8")
    profile_md = (ctx.run.workspace("eda") / "profile.md").read_text(encoding="utf-8")

    plan = StrategyPlan.model_validate_json(strategy_json)
    step = sorted(plan.roadmap, key=lambda s: s.priority)[0]

    researcher = MLResearcherAgent(
        llm=ctx.llm,
        config=ctx.agent_configs.ml_researcher,
        journal=ctx.journal,
        run_id=ctx.run.run_id,
    )
    proposal = researcher.propose(
        profile_md=profile_md,
        data_report_json=data_report_json,
        feature_plan_json=feature_plan_json,
        guidance=plan.guidance,
        roadmap_step=step,
    )
    ctx.journal.append(
        agent="ml_researcher",
        type=EventType.AGENT_OUTPUT,
        data={
            "model_family": proposal.model_family,
            "feature_ideas_used": proposal.feature_ideas_used,
        },
        run=ctx.run.run_id,
    )
    ctx.say(f"  researcher proposes {proposal.model_family}: {step.title}")

    record = ctx.registry.add(
        ModelCandidate(
            model_type=proposal.model_family,
            notes=(proposal.description or step.title)[:200],
        )
    )
    candidate_dir = ctx.run.workspace("models") / record.id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    pipeline_code = _strip_code_fences(proposal.pipeline_code)
    (candidate_dir / "pipeline.py").write_text(pipeline_code, encoding="utf-8")

    try:
        compile(pipeline_code, "pipeline.py", "exec")
    except SyntaxError as exc:
        ctx.registry.set_status(record.id, "failed")
        raise CandidateRunError(f"candidate {record.id} has a syntax error: {exc}") from exc

    evaluation_metric = json.loads(competition_json).get("evaluation_metric") or ""
    protocol = build_cv_protocol(evaluation_metric, plan.cv)
    result = run_candidate(
        ctx.run,
        record.id,
        protocol,
        plan.problem_type,
        plan.target_column,
    )
    if result.status != "ok":
        ctx.registry.set_status(record.id, "failed")
        tail = (result.error or "")[-400:]
        raise CandidateRunError(
            f"candidate {record.id} failed the harness ({result.status}):\n{tail}"
        )

    ctx.registry.record_cv(record.id, result.mean_score, result.std_score)
    ctx.say(
        f"  baseline {record.id} scored: {result.mean_score:.5f} +/- {result.std_score:.5f} "
        f"({protocol.k_folds} folds, {result.elapsed_s:.0f}s)"
    )


HANDLERS: dict[Phase, Handler] = {
    Phase.INIT: init_phase,
    Phase.DATA_COLLECTION: data_collection_phase,
    Phase.ANALYSIS: analysis_phase,
    Phase.PLANNING: planning_phase,
    Phase.BASELINE: baseline_phase,
}
