from typing import Literal

from pydantic import BaseModel, Field


class ReviewVerdict(BaseModel):
    """The reviewer's decision on one candidate."""

    decision: Literal["approve", "reject", "revise"]
    reasoning: str
    risk_flags: list[str] = Field(default_factory=list)
