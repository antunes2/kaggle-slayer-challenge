from pydantic import BaseModel, Field


class CVProtocol(BaseModel):
    """Cross-validation settings decided by the planner and enforced by the harness."""

    metric: str
    k_folds: int = Field(default=5, ge=2, le=10)
    seed: int = 42
    stratified: bool = True
    higher_is_better: bool = True


class FoldScore(BaseModel):
    """A single fold's validation score."""

    fold: int
    score: float


class CVResult(BaseModel):
    """Structured outcome of one candidate evaluation through the CV harness."""

    candidate_id: str
    metric: str
    status: str
    folds: list[FoldScore] = Field(default_factory=list)
    mean_score: float | None = None
    std_score: float | None = None
    higher_is_better: bool = True
    train_rows: int = 0
    elapsed_s: float = 0.0
    pipeline_sha256: str | None = None
    submission_sha256: str | None = None
    error: str | None = None
