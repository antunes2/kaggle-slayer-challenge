from __future__ import annotations

import json
import math
import re
import time

from slayer.agents import DataAnalystAgent, MLResearcherAgent, PlannerAgent, ReviewerAgent
from slayer.agents.planner import render_strategy_plan_md
from slayer.analysis import AnalysisError, profile_data
from slayer.analysis.notebooks import build_digest
from slayer.analysis.render import render_data_report_md
from slayer.cv import CandidateRunError, build_cv_protocol, run_candidate
from slayer.kaggle import QuotaManager
from slayer.loop.data import data_collection_phase
from slayer.loop.engine import Handler, Phase, RunContext
from slayer.loop.external import external_summary_md, load_manifest, prepare_external_data
from slayer.schemas import CVResult, DataAnalystOutput, StrategyPlan
from slayer.workspace import CandidateRecord, EventType, ModelCandidate, llm_spend


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


DIGEST_MAX_AGE_S = 12 * 3600


def notebook_digest(ctx: RunContext, max_age_s: float = DIGEST_MAX_AGE_S) -> str:
    """The public-notebook digest for the planner: cached in reports/, refreshed when stale.

    Fail-soft: without a scout-capable kaggle client, or when nothing is available,
    the last digest (or an empty string) is used and the run continues.
    """
    path = ctx.run.workspace("reports") / "notebook_digest.md"
    cached = path.read_text(encoding="utf-8") if path.exists() else ""
    if cached and time.time() - path.stat().st_mtime < max_age_s:
        return cached
    if ctx.kaggle is None or not hasattr(ctx.kaggle, "list_notebooks"):
        return cached
    digest = build_digest(ctx.kaggle, ctx.run.competition)
    if not digest.available and cached:
        return cached
    path.write_text(digest.markdown, encoding="utf-8")
    ctx.journal.append(
        agent="scout",
        type=EventType.SCOUT_DIGEST,
        data={"notebooks": digest.refs, "chars": len(digest.markdown), "errors": digest.errors},
        run=ctx.run.run_id,
    )
    ctx.say(
        f"  scout: digest of {len(digest.refs)} public notebooks "
        f"(~{len(digest.markdown) // 4} tokens)"
        if digest.available
        else "  scout: no public notebook digest available"
    )
    return digest.markdown if digest.available else ""


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
    digest = notebook_digest(ctx)
    manifest = prepare_external_data(ctx, digest)
    target_column = json.loads(data_report_json).get("target_column")

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
        notebook_digest=digest,
        external_md=external_summary_md(manifest, target_column),
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
    ctx: RunContext, plan: StrategyPlan, proposal, protocol, step, revision_of: str | None = None
) -> tuple[CandidateRecord, CVResult]:
    """Register, write, and evaluate one proposal through the harness."""
    params = {"roadmap_step": step.title, "priority": step.priority}
    if revision_of:
        params["revision_of"] = revision_of
    record = ctx.registry.add(
        ModelCandidate(
            model_type=proposal.model_family,
            notes=(proposal.description or step.title)[:200],
            params=params,
            parent_id=revision_of,
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
    ctx.say(
        f"  {record.id}: evaluating ({protocol.k_folds} folds, "
        f"budget {ctx.run.harness_timeout_s}s) — silence during CV is normal"
    )
    result = run_candidate(
        ctx.run,
        record.id,
        protocol,
        plan.problem_type,
        plan.target_column,
        timeout_s=ctx.run.harness_timeout_s,
    )
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
    done_steps, attempts = _step_stats(ctx)
    if step.title in done_steps:
        ctx.say(f"  baseline for '{step.title}' already exists — skipping")
        return
    if attempts.get(step.title, 0) >= MAX_STEP_ATTEMPTS:
        raise CandidateRunError(
            f"baseline step '{step.title}' exhausted {MAX_STEP_ATTEMPTS} attempts — "
            f"inspect {ctx.run.workspace('models')}/ and the strategy before retrying"
        )
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
        time_budget_s=ctx.run.harness_timeout_s,
        external_md=external_summary_md(load_manifest(ctx), plan.target_column),
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


def run_spend(ctx: RunContext) -> float:
    """LLM spend for the whole run: journal sum (survives resumes) or live client, if higher."""
    cost = getattr(ctx.llm, "cost", None)
    live = cost.total_usd if cost is not None else 0.0
    return max(llm_spend(ctx.journal), live)


def _budget_out(ctx: RunContext) -> bool:
    """Check the run's LLM budget; journal a warning and return True when exhausted."""
    used = run_spend(ctx)
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


MAX_STEP_ATTEMPTS = 3


def _step_stats(ctx: RunContext) -> tuple[set[str], dict[str, int]]:
    """Return (steps with a non-failed candidate, total attempts per step)."""
    done: set[str] = set()
    counts: dict[str, int] = {}
    for record in ctx.registry.list():
        title = record.params.get("roadmap_step")
        if not title:
            continue
        counts[title] = counts.get(title, 0) + 1
        if record.status != "failed":
            done.add(title)
    return done, counts


def _failure_head(result) -> str:
    """First line of a harness failure, truncated for terminal narration."""
    lines = (result.error or "").strip().splitlines()
    if not lines:
        return result.status
    return lines[0][:90]


def _reviewed_ids(ctx: RunContext) -> set[str]:
    """Candidate ids that already received a reviewer verdict (any decision)."""
    return {
        str(event.data.get("candidate_id"))
        for event in ctx.journal.read(type=EventType.AGENT_OUTPUT, agent="reviewer")
        if event.data.get("candidate_id")
    }


def _pending_revision(ctx: RunContext, higher_is_better: bool = True):
    """Return (record, verdict) for the best scored candidate with an unactioned REVISE.

    Unactioned = no candidate records it as `revision_of` yet. Best-first, so the
    strongest REVISE'd candidate is fixed before weaker ones (one per ITERATE pass).
    """
    revised = {
        r.params.get("revision_of") for r in ctx.registry.list() if r.params.get("revision_of")
    }
    latest: dict[str, dict] = {}
    for event in ctx.journal.read(type=EventType.AGENT_OUTPUT, agent="reviewer"):
        if event.data.get("candidate_id"):
            latest[str(event.data["candidate_id"])] = event.data
    pending = []
    for candidate_id, verdict in latest.items():
        if verdict.get("decision") != "revise" or candidate_id in revised:
            continue
        try:
            record = ctx.registry.get(candidate_id)
        except KeyError:
            continue
        if record.status == "scored" and record.cv_score is not None:
            pending.append((record, verdict))
    if not pending:
        return None
    pending.sort(key=lambda item: item[0].cv_score, reverse=higher_is_better)
    return pending[0]


def _run_revision(ctx: RunContext, plan: StrategyPlan, protocol, propose) -> None:
    """Act once on the latest REVISE feedback: propose, evaluate, re-review, re-gate."""
    revision = _pending_revision(ctx, protocol.higher_is_better)
    if revision is None:
        return
    record, verdict = revision
    if _budget_out(ctx):
        return
    step = next(
        (s for s in plan.roadmap if s.title == record.params.get("roadmap_step")), None
    )
    if step is None:
        return
    pipeline_path = ctx.run.workspace("models") / record.id / "pipeline.py"
    code = pipeline_path.read_text(encoding="utf-8") if pipeline_path.exists() else ""
    ctx.say(f"  acting on REVISE feedback for {record.id}")
    extra = (
        f"The reviewer REVISED candidate {record.id} and it must be fixed. "
        f"Reviewer reasoning: {verdict.get('reasoning', '')} "
        f"Risk flags: {verdict.get('risk_flags', [])}. "
        f"The current pipeline code:\n```python\n{code}\n```\n"
        "Produce a corrected version implementing the same roadmap step."
    )
    proposal = propose(step, extra=extra)
    new_record, result = _evaluate_proposal(
        ctx, plan, proposal, protocol, step, revision_of=record.id
    )
    if result.status != "ok":
        ctx.say(f"  revision {new_record.id} failed ({result.status})")
        return
    ctx.say(
        f"  revision {new_record.id} scored: {result.mean_score:.5f} "
        f"+/- {result.std_score:.5f}"
    )
    review_phase(ctx)
    submit_phase(ctx)


def iterate_phase(ctx: RunContext) -> None:
    """ITERATE: one candidate per pending roadmap step, with repair-retry and budget stops."""
    if ctx.llm is None or ctx.agent_configs is None:
        raise AnalysisError("iterate requires the LLM client — check OPENROUTER_API_KEY")
    plan, competition = _load_plan(ctx)
    protocol = build_cv_protocol(competition.get("evaluation_metric") or "", plan.cv)

    done_steps, attempts = _step_stats(ctx)
    pending = [
        s
        for s in sorted(plan.roadmap, key=lambda s: s.priority)
        if s.title not in done_steps and attempts.get(s.title, 0) < MAX_STEP_ATTEMPTS
    ]

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
            time_budget_s=ctx.run.harness_timeout_s,
            external_md=external_summary_md(load_manifest(ctx), plan.target_column),
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

    budget_hit = False
    for step in pending:
        if _budget_out(ctx):
            budget_hit = True
            break
        ctx.say(f"  iterating step {step.priority}: {step.title}")
        proposal = propose(step)
        record, result = _evaluate_proposal(ctx, plan, proposal, protocol, step)
        if result.status != "ok":
            ctx.say(
                f"  {record.id} failed ({result.status}: {_failure_head(result)}) "
                "— one repair attempt"
            )
            if _budget_out(ctx):
                budget_hit = True
                break
            failure_note = f"Your previous attempt failed:\n{(result.error or '')[-600:]}"
            proposal = propose(step, extra=failure_note)
            record, result = _evaluate_proposal(ctx, plan, proposal, protocol, step)
            if result.status != "ok":
                ctx.say(
                    f"  {record.id} failed again ({_failure_head(result)}) "
                    f"— step '{step.title}' exhausted, moving to next step"
                )
                continue
        ctx.say(
            f"  {record.id} scored: {result.mean_score:.5f} +/- {result.std_score:.5f}"
        )
        review_phase(ctx)
        submit_phase(ctx)

    if not pending:
        ctx.say("  no pending roadmap steps")
    if not budget_hit:
        _run_revision(ctx, plan, protocol, propose)

    from slayer.loop.blending import blend_phase  # lazy: blending imports this module

    blend_phase(ctx)
    submit_phase(ctx)


def review_phase(ctx: RunContext) -> None:
    """REVIEW: the reviewer agent judges the best scored, unreviewed candidate."""
    if ctx.llm is None or ctx.agent_configs is None:
        raise AnalysisError("review requires the LLM client — check OPENROUTER_API_KEY")
    plan, competition = _load_plan(ctx)
    protocol = build_cv_protocol(competition.get("evaluation_metric") or "", plan.cv)

    reviewed = _reviewed_ids(ctx)
    scored = [
        r
        for r in ctx.registry.list()
        if r.cv_score is not None
        and r.status not in ("approved", "rejected", "submitted")
        and r.id not in reviewed
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


def _fold_scores(ctx: RunContext, candidate_id: str) -> list[float]:
    """Per-fold CV scores from a candidate's cv_results.json (empty if unavailable)."""
    path = ctx.run.workspace("models") / candidate_id / "cv_results.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [float(f["score"]) for f in data.get("folds") or [] if "score" in f]


def _improvement_gate(
    ctx: RunContext, record, best, higher_is_better: bool, min_gain: float
) -> tuple[bool, str]:
    """CV gate: mean gain >= min_gain, or a consistent paired-fold gain.

    All candidates share the same folds, so the per-fold difference is far less
    noisy than one model's fold std. A gain counts as real when it is positive on all
    folds but at most one AND its mean exceeds 2x the std of the fold differences.
    """
    sign = 1 if higher_is_better else -1
    delta = (record.cv_score - best.cv_score) * sign
    if delta >= min_gain:
        return True, f"CV improvement {delta:.6f} >= gate {min_gain}"
    ours, theirs = _fold_scores(ctx, record.id), _fold_scores(ctx, best.id)
    if len(ours) == len(theirs) >= 3:
        diffs = [(a - b) * sign for a, b in zip(ours, theirs, strict=True)]
        mean = sum(diffs) / len(diffs)
        std = (sum((d - mean) ** 2 for d in diffs) / len(diffs)) ** 0.5
        wins = sum(1 for d in diffs if d > 0)
        if mean > 0 and wins >= len(diffs) - 1 and mean > 2 * std:
            return True, (
                f"paired gain {mean:.6f} on {wins}/{len(diffs)} folds "
                f"(fold-diff std {std:.6f}) vs {best.id}"
            )
        return False, (
            f"CV improvement {delta:.6f} < gate {min_gain}; paired gain {mean:.6f} on "
            f"{wins}/{len(diffs)} folds (fold-diff std {std:.6f}) not consistent"
        )
    return False, f"CV improvement {delta:.6f} < gate {min_gain}"


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
        passed, reason = _improvement_gate(
            ctx, record, best_submitted[0], protocol.higher_is_better, plan.min_cv_improvement
        )
        if not passed:
            ctx.say(f"  skipping {record.id}: {reason}")
            return
        ctx.say(f"  {record.id} passes the CV gate: {reason}")

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


def _parse_score(raw: str | None) -> float | None:
    """Parse a Kaggle score string into a float, tolerating None/blank."""
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _higher_is_better(ctx: RunContext, rows) -> bool:
    """Metric direction from the strategy's CV protocol; fall back to leaderboard order."""
    try:
        plan, competition = _load_plan(ctx)
        return build_cv_protocol(
            competition.get("evaluation_metric") or "", plan.cv
        ).higher_is_better
    except Exception:
        ranked = sorted((r for r in rows if r.score is not None), key=lambda r: r.rank)
        return len(ranked) < 2 or ranked[0].score >= ranked[-1].score


def _target_score(rows, target_percentile: int) -> float | None:
    """Score of the leaderboard entry sitting exactly at the target percentile line."""
    ranked = sorted((r for r in rows if r.score is not None), key=lambda r: r.rank)
    if not ranked:
        return None
    index = max(1, math.floor(len(ranked) * target_percentile / 100)) - 1
    return ranked[index].score


def _best_own_score(ctx: RunContext, higher: bool) -> tuple[float | None, str]:
    """Our best real LB score (basis 'lb'), else best CV as a proxy (basis 'cv')."""
    lb = [
        r.lb_score
        for r in ctx.registry.list()
        if r.lb_score is not None and not (r.submission_id or "").startswith("mock-")
    ]
    if lb:
        return (max(lb) if higher else min(lb)), "lb"
    cv = [r.cv_score for r in ctx.registry.list() if r.cv_score is not None]
    if cv:
        return (max(cv) if higher else min(cv)), "cv"
    return None, "none"


def _report_position(ctx: RunContext, rows, username: str) -> None:
    """Report our position vs target (listed, LB-estimated, or CV-estimated) plus the gap."""
    target = ctx.run.target_percentile
    total = len(rows)
    higher = _higher_is_better(ctx, rows)
    target_score = _target_score(rows, target)
    username_lower = str(username).strip().lower()
    our = next((row for row in rows if username_lower in row.member_usernames), None)

    if our is not None:
        rank, field, best, basis, ties = our.rank, total, our.score, "listed", None
    else:
        best, basis = _best_own_score(ctx, higher)
        scores = [row.score for row in rows if row.score is not None]
        if best is None or not scores:
            ctx.say(
                f"  leaderboard: username '{username}' not found among {total} entries "
                "and no own score to estimate from"
            )
            ctx.journal.append(
                agent="loop",
                type=EventType.MONITOR_POSITION,
                data={"entries_fetched": total, "our_entry": False},
                run=ctx.run.run_id,
            )
            return
        better = sum(1 for s in scores if (s > best if higher else s < best))
        ties = sum(1 for s in scores if s == best)
        rank, field = better + 1, total + 1

    percentile = round(rank / field * 100, 2) if field else None
    met = percentile is not None and percentile <= target
    gap = None
    if target_score is not None and best is not None:
        gap = round((target_score - best) if higher else (best - target_score), 6)

    if basis == "listed":
        head = f"rank {rank}/{field} (top {percentile}%)"
    else:
        source = "best LB score" if basis == "lb" else "best CV (no real LB score yet)"
        head = (
            f"not listed — {source} {best:.5f} would rank ~{rank}/{field} "
            f"(top {percentile}%), tie band of {ties}"
        )
    gap_text = (
        f"; target line {target_score:.5f}, gap {gap:+.5f}" if gap is not None else ""
    )
    ctx.say(
        f"  leaderboard: {head} — target top {target}%: "
        + ("MET" if met else "NOT met")
        + gap_text
    )
    data = {
        "our_entry": basis == "listed",
        "basis": basis,
        "rank": rank,
        "total": total,
        "percentile": percentile,
        "target_percentile": target,
        "met": met,
        "best_score": best,
        "target_score": target_score,
        "gap": gap,
    }
    if basis != "listed":
        data["estimated"] = True
    ctx.journal.append(
        agent="loop", type=EventType.MONITOR_POSITION, data=data, run=ctx.run.run_id
    )


def monitor_phase(ctx: RunContext) -> None:
    """MONITOR: collect real LB scores for submitted candidates; report position vs target."""
    if ctx.kaggle is None:
        raise AnalysisError("monitor requires the kaggle client")
    submissions = ctx.kaggle.submissions(ctx.run.competition)
    by_ref = {s.ref: s for s in submissions}
    by_description = {}
    for submission in submissions:
        found = re.search(r"slayer (cand-\d+):", submission.description or "")
        if found:
            by_description[found.group(1)] = submission

    for record in ctx.registry.list():
        if record.status != "submitted" or record.lb_score is not None:
            continue
        if (record.submission_id or "").startswith("mock-"):
            continue
        submission = by_ref.get(record.submission_id) or by_description.get(record.id)
        if submission is None:
            continue
        score = _parse_score(submission.public_score)
        if score is None:
            ctx.say(
                f"  {record.id}: submission {submission.ref} "
                f"status={submission.status} — no LB score yet"
            )
            continue
        ctx.registry.record_lb(record.id, score, submission.ref)
        ctx.say(f"  {record.id}: LB score {score:.5f} (submission {submission.ref})")

    username = ctx.kaggle.ping()
    rows = ctx.kaggle.full_leaderboard(
        ctx.run.competition, target_dir=ctx.run.workspace("reports")
    )
    _report_position(ctx, rows, username)


HANDLERS: dict[Phase, Handler] = {
    Phase.INIT: init_phase,
    Phase.DATA_COLLECTION: data_collection_phase,
    Phase.ANALYSIS: analysis_phase,
    Phase.PLANNING: planning_phase,
    Phase.BASELINE: baseline_phase,
    Phase.ITERATE: iterate_phase,
    Phase.REVIEW: review_phase,
    Phase.SUBMIT: submit_phase,
    Phase.MONITOR: monitor_phase,
}
