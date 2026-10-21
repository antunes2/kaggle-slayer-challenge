from pydantic import BaseModel, Field

from slayer.schemas.analysis import ProblemType


class RoadmapStep(BaseModel):
    """One planned modelling step for the researcher to execute."""

    title: str
    description: str
    model_family: str
    feature_ideas: list[str] = Field(default_factory=list)
    priority: int = 1


class CVSettings(BaseModel):
    """Planner-chosen CV shape (metric + direction are resolved deterministically)."""

    k_folds: int = Field(default=5, ge=2, le=10)
    seed: int = 42
    stratified: bool = True


class StrategyPlan(BaseModel):
    """The planner's per-competition strategy for the whole run."""

    problem_type: ProblemType
    target_column: str
    cv: CVSettings
    roadmap: list[RoadmapStep] = Field(min_length=1)
    min_cv_improvement: float = Field(default=0.0005, ge=0.0)
    guidance: list[str] = Field(default_factory=list)
    summary: str
