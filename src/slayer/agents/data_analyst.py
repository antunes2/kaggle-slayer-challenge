from __future__ import annotations

from slayer.agents.base import Agent
from slayer.schemas import DataAnalystOutput


class DataAnalystAgent(Agent):
    """Interprets the dataset profile into a DataReport and FeaturePlan."""

    name = "data_analyst"
    prompt_file = "data_analyst"

    def analyze(self, profile_md: str, competition_json: str) -> DataAnalystOutput:
        """Produce the analysis output from a profile and the competition context."""

        user_message = (
            "Competition context (JSON):\n"
            f"{competition_json}\n\n"
            "Dataset profile (deterministic, computed from the real data):\n"
            f"{profile_md}\n\n"
            "Interpret the profile and produce your report and feature plan."
        )
        _, output = self.run_structured(user_message, DataAnalystOutput)
        return output
