from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    """Per-column statistics for the dataset profile."""

    name: str
    dtype: str
    missing_pct: float
    nunique: int
    numeric: bool
    facts: dict[str, str] = Field(default_factory=dict)


class DatasetProfile(BaseModel):
    """Aggregated dataset profile consumed by the data analyst agent."""

    competition: str
    train_rows: int = 0
    train_cols: int = 0
    test_rows: int = 0
    target_column: str | None = None
    problem_type_guess: str | None = None
    columns: list[ColumnProfile] = Field(default_factory=list)
    target_summary: dict[str, str] = Field(default_factory=dict)
    top_correlations: list[tuple[str, float]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
