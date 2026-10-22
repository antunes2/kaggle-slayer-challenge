from __future__ import annotations

from slayer.agents.base import Agent
from slayer.schemas import RoadmapExtension, StrategyPlan


class PlannerAgent(Agent):
    """Turns analysis artifacts into the per-competition StrategyPlan."""

    name = "planner"
    prompt_file = "planner"

    def plan(
        self, competition_json: str, data_report_json: str, feature_plan_json: str
    ) -> StrategyPlan:
        """Produce the strategy from competition context, data report, and features."""
        user_message = (
            "Competition context (JSON):\n"
            f"{competition_json}\n\n"
            "Data analyst report (JSON):\n"
            f"{data_report_json}\n\n"
            "Feature plan (JSON):\n"
            f"{feature_plan_json}\n\n"
            "Produce the strategy plan."
        )
        _, plan = self.run_structured(user_message, StrategyPlan)
        return plan


class PlannerExtensionAgent(Agent):
    """The planner in re-entry mode: extends the roadmap from the experiment history."""

    name = "planner"
    prompt_file = "planner_extend"

    def extend(
        self,
        strategy_md: str,
        history_md: str,
        position_md: str,
        constraints_md: str,
        round_number: int,
    ) -> RoadmapExtension:
        """Diagnose the shortfall and propose the next roadmap segment (or stop)."""
        user_message = (
            f"Re-entry round {round_number}. The run is short of its target.\n\n"
            "Current strategy:\n"
            f"{strategy_md}\n\n"
            "Full experiment history (every candidate so far):\n"
            f"{history_md}\n\n"
            "Leaderboard position and gap:\n"
            f"{position_md}\n\n"
            "Constraints for this round:\n"
            f"{constraints_md}\n\n"
            "Produce the roadmap extension."
        )
        _, extension = self.run_structured(user_message, RoadmapExtension)
        return extension


def render_strategy_plan_md(plan: StrategyPlan) -> str:
    """Render the strategy plan as human-readable markdown."""
    lines = [
        f"# Strategy — {plan.problem_type} on `{plan.target_column}`",
        "",
        plan.summary,
        "",
        "## Cross-validation",
        f"- folds: {plan.cv.k_folds} (stratified: {plan.cv.stratified}, seed: {plan.cv.seed})",
        f"- submission gate: CV improvement >= {plan.min_cv_improvement}",
        "",
        "## Roadmap",
        "| # | title | family | feature ideas |",
        "|---|---|---|---|",
    ]
    for step in sorted(plan.roadmap, key=lambda s: s.priority):
        ideas = ", ".join(step.feature_ideas) or "-"
        lines.append(f"| {step.priority} | {step.title} | {step.model_family} | {ideas} |")
    lines.append("")
    if plan.guidance:
        lines.append("## Guidance for the researcher")
        lines.extend(f"- {g}" for g in plan.guidance)
        lines.append("")
    return "\n".join(lines) + "\n"
