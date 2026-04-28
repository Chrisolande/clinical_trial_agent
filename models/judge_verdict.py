from typing import Literal

from pydantic import BaseModel, Field, model_validator

from clinical_trial_agent.config import TIER_ORDER

MIN_ASSESSED_CRITERIA_FOR_STRONG = 3
MIN_TOTAL_CRITERIA_FOR_STRONG = 4
MIN_MAJOR_MET_FOR_STRONG = 1
MAX_UNCERTAIN_RATIO_FOR_STRONG = 0.34
MAJOR_CRITERION_KEYWORDS = (
    "diagnosis",
    "histology",
    "pathology",
    "biomarker",
    "mutation",
    "pd-l1",
    "braf",
    "egfr",
    "kras",
    "stage",
    "metast",
    "performance status",
    "ecog",
    "karnofsky",
    "prior treatment",
    "line of therapy",
    "measurable disease",
    "recist",
)


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

    @model_validator(mode="after")
    def _coerce_strong_tier_requirements(self) -> "JudgeVerdict":
        if self.match_tier != "strong":
            return self

        assessed_count = (
            len(self.inclusion_met) + len(self.inclusion_failed) + len(self.exclusion_triggered)
        )
        total_count = assessed_count + len(self.inclusion_uncertain) + len(self.exclusion_uncertain)
        uncertain_ratio = (
            (len(self.inclusion_uncertain) + len(self.exclusion_uncertain)) / total_count
            if total_count
            else 1.0
        )
        major_met_count = sum(
            1
            for criterion in self.inclusion_met
            if any(keyword in criterion.lower() for keyword in MAJOR_CRITERION_KEYWORDS)
        )

        strong_requirements_met = (
            not self.exclusion_triggered
            and not self.exclusion_uncertain
            and self.major_criteria_assessable
            and major_met_count >= MIN_MAJOR_MET_FOR_STRONG
            and assessed_count >= MIN_ASSESSED_CRITERIA_FOR_STRONG
            and total_count >= MIN_TOTAL_CRITERIA_FOR_STRONG
            and uncertain_ratio <= MAX_UNCERTAIN_RATIO_FOR_STRONG
            and len(self.critical_missing_info) == 0
            and len(self.inclusion_failed) == 0
        )
        if strong_requirements_met:
            return self

        if self.exclusion_triggered:
            self.match_tier = "weak"
            if self.match_score > 0.35:
                self.match_score = 0.35
            return self

        self.match_tier = "moderate"
        if self.match_score > 0.74:
            self.match_score = 0.74
        return self
