from pydantic import BaseModel, Field


class PipelineProposal(BaseModel):
    """The researcher's generated candidate pipeline."""

    model_family: str
    description: str
    pipeline_code: str
    feature_ideas_used: list[str] = Field(default_factory=list)
    notes: str | None = None
