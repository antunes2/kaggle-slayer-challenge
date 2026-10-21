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
    ) -> PipelineProposal:
        """Generate the pipeline code for one roadmap step."""
        guidance_text = "\n".join(f"- {g}" for g in guidance) or "- (none)"
        user_message = (
            "Dataset profile (exact column names and dtypes):\n"
            f"{profile_md}\n\n"
            "Data analyst report (JSON):\n"
            f"{data_report_json}\n\n"
            "Feature plan (JSON):\n"
            f"{feature_plan_json}\n\n"
            "Strategy guidance (mandatory):\n"
            f"{guidance_text}\n\n"
            "Current roadmap step to implement (JSON):\n"
            f"{roadmap_step.model_dump_json(indent=2)}\n\n"
            "Write the pipeline.py for this step."
        )
        _, proposal = self.run_structured(user_message, PipelineProposal)
        return proposal
