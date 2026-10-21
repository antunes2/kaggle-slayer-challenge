from __future__ import annotations

import json

from slayer.agents import DataAnalystAgent, MLResearcherAgent, PlannerAgent, ReviewerAgent
from slayer.agents.planner import render_strategy_plan_md
from slayer.analysis import AnalysisError, profile_data
from slayer.analysis.render import render_data_report_md
from slayer.cv import CandidateRunError, build_cv_protocol, run_candidate
from slayer.kaggle import QuotaManager
from slayer.loop.data import data_collection_phase
from slayer.loop.engine import Handler, Phase, RunContext
from slayer.schemas import CVResult, DataAnalystOutput, StrategyPlan
from slayer.workspace import CandidateRecord, EventType, ModelCandidate


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
    reports_dir = ctx.run.workspace("reports")
    if (reports_dir / "data_report.json").exists() and (
        reports_dir / "feature_plan.json"
    ).exists():
        ctx.say("  analysis artifacts exist — skipping")
        return
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
    if (reports_dir / "strategy_plan.json").exists():
        ctx.say("  strategy plan exists — skipping")
        return
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


def _evaluate_proposal(
    ctx: RunContext, plan: StrategyPlan, proposal, protocol, step
) -> tuple[CandidateRecord, CVResult]:
    """Register, write, and evaluate one proposal through the harness."""
    record = ctx.registry.add(
        ModelCandidate(
            model_type=proposal.model_family,
            notes=(proposal.description or step.title)[:200],
            params={"roadmap_step": step.title, "priority": step.priority},
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
        return record, CVResult(
            candidate_id=record.id,
            metric=protocol.metric,
            status="error",
            error=f"syntax error: {exc}",
        )
    result = run_candidate(ctx.run, record.id, protocol, plan.problem_type, plan.target_column)
    if result.status == "ok":
        ctx.registry.record_cv(record.id, result.mean_score, result.std_score)
    else:
        ctx.registry.set_status(record.id, "failed")
    return record, result


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
    existing = [
        r
        for r in ctx.registry.list()
        if r.params.get("roadmap_step") == step.title and r.status != "failed"
    ]
    if existing:
        ctx.say(f"  baseline for '{step.title}' already exists ({existing[0].id}) — skipping")
        return
    evaluation_metric = json.loads(competition_json).get("evaluation_metric") or ""
    protocol = build_cv_protocol(evaluation_metric, plan.cv)

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

    record, result = _evaluate_proposal(ctx, plan, proposal, protocol, step)
    if result.status != "ok":
        tail = (result.error or "")[-400:]
        raise CandidateRunError(
            f"candidate {record.id} failed the harness ({result.status}):\n{tail}"
        )
    ctx.say(
        f"  baseline {record.id} scored: {result.mean_score:.5f} +/- {result.std_score:.5f} "
        f"({protocol.k_folds} folds, {result.elapsed_s:.0f}s)"
    )


def _load_plan(ctx: RunContext) -> tuple[StrategyPlan, dict]:
    """Load the strategy plan and competition context for later phases."""
    reports_dir = ctx.run.workspace("reports")
    plan = StrategyPlan.model_validate_json(
        (reports_dir / "strategy_plan.json").read_text(encoding="utf-8")
    )
    competition = json.loads((reports_dir / "competition.json").read_text(encoding="utf-8"))
    return plan, competition


def _iteration_context(ctx: RunContext) -> str:
    """Build the researcher's feedback bundle: candidates, last review, last failure."""
    parts = ["Candidates so far:\n" + ctx.registry.summary_for_agent(n=8)]
    verdicts = ctx.journal.read(type=EventType.AGENT_OUTPUT, agent="reviewer")
    if verdicts:
        last = verdicts[-1].data
        parts.append(
            f"Last review verdict: {last.get('decision')} "
            f"(risk flags: {last.get('risk_flags')}): {last.get('reasoning', '')}"
        )
    failed = [r for r in ctx.registry.list() if r.status == "failed"]
    if failed:
        log_path = ctx.run.workspace("models") / failed[-1].id / "run.log"
        if log_path.exists():
            parts.append(
                "Last failed candidate log tail (fix this):\n"
                + log_path.read_text(encoding="utf-8", errors="replace")[-500:]
            )
    return "\n\n".join(parts)


def _budget_out(ctx: RunContext) -> bool:
    """Check the run's LLM budget; journal a warning and return True when exhausted."""
    cost = getattr(ctx.llm, "cost", None)
    if cost is None:
        return False
    used = cost.total_usd
    if used >= ctx.run.budget_usd:
        ctx.journal.append(
            agent="loop",
            type=EventType.BUDGET_WARNING,
            data={"used_usd": used, "budget_usd": ctx.run.budget_usd},
            run=ctx.run.run_id,
        )
        ctx.say(f"  LLM budget exhausted (${used:.4f} >= ${ctx.run.budget_usd}) — stopping")
        return True
    return False


def iterate_phase(ctx: RunContext) -> None:
    """ITERATE: one candidate per pending roadmap step, with repair-retry and budget stops."""
    if ctx.llm is None or ctx.agent_configs is None:
        raise AnalysisError("iterate requires the LLM client — check OPENROUTER_API_KEY")
    plan, competition = _load_plan(ctx)
    protocol = build_cv_protocol(competition.get("evaluation_metric") or "", plan.cv)

    existing_steps = {
        r.params.get("roadmap_step")
        for r in ctx.registry.list()
        if r.params.get("roadmap_step") and r.status != "failed"
    }
    pending = [
        s for s in sorted(plan.roadmap, key=lambda s: s.priority) if s.title not in existing_steps
    ]
    if not pending:
        ctx.say("  no pending roadmap steps — nothing to iterate")
        return

    reports_dir = ctx.run.workspace("reports")
    profile_md = (ctx.run.workspace("eda") / "profile.md").read_text(encoding="utf-8")
    data_report_json = (reports_dir / "data_report.json").read_text(encoding="utf-8")
    feature_plan_json = (reports_dir / "feature_plan.json").read_text(encoding="utf-8")
    researcher = MLResearcherAgent(
        llm=ctx.llm,
        config=ctx.agent_configs.ml_researcher,
        journal=ctx.journal,
        run_id=ctx.run.run_id,
    )

    def propose(step, extra: str = ""):
        """Ask the researcher for a pipeline for the given step."""
        proposal = researcher.propose(
            profile_md=profile_md,
            data_report_json=data_report_json,
            feature_plan_json=feature_plan_json,
            guidance=plan.guidance,
            roadmap_step=step,
            iteration_context=_iteration_context(ctx) + (f"\n\n{extra}" if extra else ""),
        )
        ctx.journal.append(
            agent="ml_researcher",
            type=EventType.AGENT_OUTPUT,
            data={
                "roadmap_step": step.title,
                "model_family": proposal.model_family,
                "feature_ideas_used": proposal.feature_ideas_used,
            },
            run=ctx.run.run_id,
        )
        return proposal

    for step in pending:
        if _budget_out(ctx):
            return
        ctx.say(f"  iterating step {step.priority}: {step.title}")
        proposal = propose(step)
        record, result = _evaluate_proposal(ctx, plan, proposal, protocol, step)
        if result.status != "ok":
            ctx.say(f"  {record.id} failed ({result.status}) — one repair attempt")
            if _budget_out(ctx):
                return
            failure_note = f"Your previous attempt failed:\n{(result.error or '')[-600:]}"
            proposal = propose(step, extra=failure_note)
            record, result = _evaluate_proposal(ctx, plan, proposal, protocol, step)
            if result.status != "ok":
                ctx.say(f"  {record.id} failed again — moving to next step")
                continue
        ctx.say(
            f"  {record.id} scored: {result.mean_score:.5f} +/- {result.std_score:.5f}"
        )
        review_phase(ctx)
        submit_phase(ctx)


def review_phase(ctx: RunContext) -> None:
    """REVIEW: the reviewer agent judges the best scored, unreviewed candidate."""
    if ctx.llm is None or ctx.agent_configs is None:
        raise AnalysisError("review requires the LLM client — check OPENROUTER_API_KEY")
    plan, competition = _load_plan(ctx)
    protocol = build_cv_protocol(competition.get("evaluation_metric") or "", plan.cv)

    scored = [
        r
        for r in ctx.registry.list()
        if r.cv_score is not None and r.status not in ("approved", "rejected", "submitted")
    ]
    if not scored:
        ctx.say("  nothing to review — no unreviewed scored candidates")
        return
    scored.sort(key=lambda r: r.cv_score, reverse=protocol.higher_is_better)
    record = scored[0]

    models_dir = ctx.run.workspace("models") / record.id
    pipeline_path = models_dir / "pipeline.py"
    cv_results_path = models_dir / "cv_results.json"
    pipeline_code = (
        pipeline_path.read_text(encoding="utf-8") if pipeline_path.exists() else ""
    )
    cv_results = (
        cv_results_path.read_text(encoding="utf-8") if cv_results_path.exists() else "{}"
    )
    data_report_json = (ctx.run.workspace("reports") / "data_report.json").read_text(
        encoding="utf-8"
    )
    reviewer = ReviewerAgent(
        llm=ctx.llm,
        config=ctx.agent_configs.reviewer,
        journal=ctx.journal,
        run_id=ctx.run.run_id,
    )
    verdict = reviewer.review(
        pipeline_code=pipeline_code,
        cv_results_json=cv_results,
        strategy_guidance=plan.guidance,
        data_report_json=data_report_json,
    )
    ctx.journal.append(
        agent="reviewer",
        type=EventType.AGENT_OUTPUT,
        data={
            "candidate_id": record.id,
            "decision": verdict.decision,
            "risk_flags": verdict.risk_flags,
            "reasoning": verdict.reasoning[:400],
        },
        run=ctx.run.run_id,
    )
    if verdict.decision == "approve":
        ctx.registry.set_status(record.id, "approved")
    elif verdict.decision == "reject":
        ctx.registry.set_status(record.id, "rejected")
    ctx.say(
        f"  review {record.id}: {verdict.decision.upper()} — {verdict.reasoning[:120]}"
    )


def submit_phase(ctx: RunContext) -> None:
    """SUBMIT: gate (CV improvement + quota + approval) then submit via the sink."""
    if ctx.sink is None:
        raise AnalysisError("submit requires a submission sink — check run configuration")
    plan, competition = _load_plan(ctx)
    protocol = build_cv_protocol(competition.get("evaluation_metric") or "", plan.cv)

    approved = [
        r
        for r in ctx.registry.list()
        if r.cv_score is not None and r.status == "approved"
    ]
    if not approved:
        ctx.say("  nothing approved to submit — skipping submission")
        return
    approved.sort(key=lambda r: r.cv_score, reverse=protocol.higher_is_better)
    record = approved[0]

    best_submitted = ctx.registry.best(
        1, higher_is_better=protocol.higher_is_better, submitted_only=True
    )
    if best_submitted:
        delta = (record.cv_score - best_submitted[0].cv_score) * (
            1 if protocol.higher_is_better else -1
        )
        if delta < plan.min_cv_improvement:
            ctx.say(
                f"  skipping {record.id}: CV improvement {delta:.6f} < "
                f"gate {plan.min_cv_improvement}"
            )
            return

    if not ctx.run.dry_run:
        quota = QuotaManager(ctx.journal, ctx.run.submissions_per_day)
        if not quota.can_submit():
            ctx.say(f"  daily submission quota exhausted ({ctx.run.submissions_per_day}/day)")
            return

    if ctx.run.autonomy == "approve-first":
        if ctx.approver is None:
            ctx.say("  approval required (approve-first) but no approver — skipping")
            return
        question = (
            f"Submit {record.id} ({record.model_type}, CV {record.cv_score:.5f}) "
            f"to {ctx.run.competition}?"
        )
        if not ctx.approver(question):
            ctx.say(f"  user declined submission of {record.id}")
            return

    message = f"slayer {record.id}: {record.model_type} CV={record.cv_score:.5f}"
    result = ctx.sink.submit(
        ctx.run, ctx.registry, ctx.journal, record, message=message
    )
    ctx.say(
        f"  submitted {record.id} as {result.submission_id} "
        f"{'(mock)' if result.simulated else '(kaggle)'}"
    )


HANDLERS: dict[Phase, Handler] = {
    Phase.INIT: init_phase,
    Phase.DATA_COLLECTION: data_collection_phase,
    Phase.ANALYSIS: analysis_phase,
    Phase.PLANNING: planning_phase,
    Phase.BASELINE: baseline_phase,
    Phase.ITERATE: iterate_phase,
    Phase.REVIEW: review_phase,
    Phase.SUBMIT: submit_phase,
}
