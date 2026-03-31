from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Verdict(StrEnum):
    MEETS = "MEETS"
    FAILS = "FAILS"
    UNCERTAIN = "UNCERTAIN"


class MatchTier(StrEnum):
    STRONG = "strong_match"
    POSSIBLE = "possible_match"
    UNLIKELY = "unlikely_match"
    DISQUALIFIED = "disqualified"


class CriterionVerdict(BaseModel):
    model_config = ConfigDict(extra="allow")

    criterion_id: str
    verdict: Verdict
    justification: str

    @field_validator("justification", mode="before")
    @classmethod
    def _normalize_justification(cls, value: Any) -> str:
        text = str(value or "").strip()
        return text or "Insufficient data to assess."


class EligibilityVerdictList(BaseModel):
    verdicts: list[CriterionVerdict]


class ScoredTrial(BaseModel):
    nct_id: str
    trial_title: str
    phase: str
    status: str
    score: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    tier: MatchTier
    hard_excluded: bool
    data_sufficiency: bool
    verdicts: list[CriterionVerdict] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)


class EligibilityReasonerInput(BaseModel):
    patient_summary: str
    nct_id: str
    trial_title: str
    criteria_text: str


class ExecutiveSummaryInput(BaseModel):
    patient_summary: str
    strong_count: int
    possible_count: int
    unlikely_count: int
    total: int
    top_trials: str
    missing_info: str


class ExecutiveSummaryOutput(BaseModel):
    executive_summary: str
    patient_summary: str


class TrialEligibilityResult(BaseModel):
    trial_id: str
    verdicts: list[CriterionVerdict] = Field(default_factory=list)
    meets_count: int = 0
    fails_count: int = 0
    uncertain_count: int = 0
    hard_exclusion_failures: int = 0

    @model_validator(mode="after")
    def _sync_counts(self) -> TrialEligibilityResult:
        meets = fails = uncertain = hard_excl = 0
        for v in self.verdicts:
            if v.verdict == Verdict.MEETS:
                meets += 1
            elif v.verdict == Verdict.FAILS:
                fails += 1
                if bool(getattr(v, "is_hard_exclusion", False)):
                    hard_excl += 1
            elif v.verdict == Verdict.UNCERTAIN:
                uncertain += 1
        self.meets_count = meets
        self.fails_count = fails
        self.uncertain_count = uncertain
        self.hard_exclusion_failures = hard_excl
        return self
