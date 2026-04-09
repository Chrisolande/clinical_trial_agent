from typing import Literal

from pydantic import BaseModel, Field, model_validator

from config import TIER_ORDER


class JudgeVerdict(BaseModel):
    match_score: float = Field(ge=0.0, le=1.0)
    match_tier: Literal["strong", "moderate", "weak", "disqualified"]
    major_criteria_assessable: bool
    inclusion_met: list[str] = Field(default_factory=list)
    inclusion_failed: list[str] = Field(default_factory=list)
    inclusion_uncertain: list[str] = Field(default_factory=list)
    exclusion_triggered: list[str] = Field(default_factory=list)
    exclusion_uncertain: list[str] = Field(default_factory=list)
    critical_missing_info: list[str] = Field(default_factory=list)
    key_concern: str
    rationale: str

    @model_validator(mode="after")
    def _coerce_disqualified_score(self) -> "JudgeVerdict":
        if self.match_tier == "disqualified" and self.match_score != 0.0:
            self.match_score = 0.0
        return self

    @model_validator(mode="after")
    def _coerce_major_criteria_uncertainty(self) -> "JudgeVerdict":
        if not self.major_criteria_assessable and self.match_score > 0.55:
            self.match_score = 0.55
        if (
            not self.major_criteria_assessable
            and TIER_ORDER[self.match_tier] > TIER_ORDER["moderate"]
        ):
            self.match_tier = "moderate"
        return self

    @model_validator(mode="after")
    def _coerce_high_uncertainty_tier(self) -> "JudgeVerdict":
        uncertain_count = len(self.inclusion_uncertain) + len(self.exclusion_uncertain)
        total = (
            len(self.inclusion_met)
            + len(self.inclusion_failed)
            + len(self.inclusion_uncertain)
            + len(self.exclusion_triggered)
            + len(self.exclusion_uncertain)
        )
        if total > 0 and (uncertain_count / total) > 0.5 and self.match_tier == "strong":
            self.match_tier = "moderate"
        return self
