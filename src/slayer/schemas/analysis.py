from typing import Literal

from pydantic import BaseModel, Field

ProblemType = Literal["binary_classification", "multiclass_classification", "regression"]


class DataReport(BaseModel):
    problem_type: ProblemType
    target_column: str
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    data_quality_issues: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class FeatureIdea(BaseModel):
    name: str
    rationale: str
    expected_value: Literal["low", "medium", "high"]
    based_on_columns: list[str] = Field(default_factory=list)


class FeaturePlan(BaseModel):
    ideas: list[FeatureIdea] = Field(default_factory=list)


class DataAnalystOutput(BaseModel):
    report: DataReport
    feature_plan: FeaturePlan
