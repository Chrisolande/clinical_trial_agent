from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class MatchTier(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    DISQUALIFIED = "disqualified"


class ScoredTrial(BaseModel):
    trial_id: str
    brief_title: str
    phase: str | None = None
    overall_status: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    tier: MatchTier
    meets_count: int = 0
    fails_count: int = 0
    uncertain_count: int = 0
    hard_exclusion_failures: int = 0
    major_criteria_assessable: bool = False
    key_concern: str = ""
    critical_missing_info: list[str] = Field(default_factory=list)
    rationale: str = ""
