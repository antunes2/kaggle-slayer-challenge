from __future__ import annotations

from slayer.agents.base import Agent
from slayer.schemas import PipelineProposal, RoadmapStep


class MLResearcherAgent(Agent):
    """Writes candidate pipeline.py plugins from the strategy roadmap."""

    name = "ml_researcher"
    prompt_file = "ml_researcher"

    def propose(
        self,
        profile_md: str,
        data_report_json: str,
        feature_plan_json: str,
        guidance: list[str],
        roadmap_step: RoadmapStep,
        iteration_context: str = "",
        time_budget_s: int | None = None,
        external_md: str = "",
    ) -> PipelineProposal:
        """Generate the pipeline code for one roadmap step, with prior-run feedback."""
        guidance_text = "\n".join(f"- {g}" for g in guidance) or "- (none)"
        context_section = (
            f"Iteration context (previous candidates, scores, feedback — obey it):\n"
            f"{iteration_context}\n\n"
            if iteration_context
            else ""
        )
        budget_section = ""
        if time_budget_s:
            per_fold_hint = max(1, time_budget_s // 6)
            budget_section = (
                f"Time budget: the harness must complete all CV folds plus one final "
                f"fit within {time_budget_s}s total — a pipeline exceeding it FAILS. "
                f"Keep a single fold well under ~{per_fold_hint}s; calibrate "
                "n_estimators and preprocessing cost accordingly.\n\n"
            )
        user_message = (
            "Dataset profile (exact column names and dtypes):\n"
            f"{profile_md}\n\n"
            "Data analyst report (JSON):\n"
            f"{data_report_json}\n\n"
            "Feature plan (JSON):\n"
            f"{feature_plan_json}\n\n"
            "Strategy guidance (mandatory):\n"
            f"{guidance_text}\n\n"
            f"{budget_section}"
            f"{external_md + chr(10) + chr(10) if external_md else ''}"
            f"{context_section}"
            "Current roadmap step to implement (JSON):\n"
            f"{roadmap_step.model_dump_json(indent=2)}\n\n"
            "Write the pipeline.py for this step."
        )
        _, proposal = self.run_structured(user_message, PipelineProposal)
        return proposal
