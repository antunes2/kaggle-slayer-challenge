from __future__ import annotations

from slayer.agents.base import Agent
from slayer.schemas import ReviewVerdict


class ReviewerAgent(Agent):
    """Adversarially reviews candidate pipelines before submission."""

    name = "reviewer"
    prompt_file = "reviewer"

    def review(
        self,
        pipeline_code: str,
        cv_results_json: str,
        strategy_guidance: list[str],
        data_report_json: str,
    ) -> ReviewVerdict:
        """Judge one candidate against the leakage/quality checklist."""
        guidance_text = "\n".join(f"- {g}" for g in strategy_guidance) or "- (none)"
        user_message = (
            "Candidate pipeline.py code:\n"
            f"```python\n{pipeline_code}\n```\n\n"
            "CV results (JSON):\n"
            f"{cv_results_json}\n\n"
            "Strategy guidance (mandatory):\n"
            f"{guidance_text}\n\n"
            "Data analyst report (JSON):\n"
            f"{data_report_json}\n\n"
            "Review this candidate."
        )
        _, verdict = self.run_structured(user_message, ReviewVerdict)
        return verdict
