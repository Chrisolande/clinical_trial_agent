from typing import Literal

from pydantic import BaseModel, Field

CriterionCategory = Literal[
    "age", "lab", "biomarker", "diagnosis", "medication", "performance", "other"
]


class EligibilityCriterion(BaseModel):
    text: str = Field(description="Exact Criterion text, cleaned and atomic")
    is_hard_exclusion: bool = Field(
        description="True for all exclusion criteria and critical safety exclusions."
    )
    category: CriterionCategory = Field(description="Clinical category of the criterion.")


class ParsedEligibilityCriterion(BaseModel):
    inclusion_criteria: list[EligibilityCriterion] = Field(default_factory=list)
    exclusion_criteria: list[EligibilityCriterion] = Field(default_factory=list)


class CriterionAssessment(BaseModel):
    criterion_id: str = Field(description="Unique Criterion ID (e.g., NCT00000000_inc_0)")
    verdict: Literal["MEETS", "FAILS", "UNCERTAIN"] = Field(
        description="The assessment verdict for this specific criterion."
    )
    justification: str = Field(description="A one-sentence clinical justification for the verdict.")


class EligibilityAssessmentList(BaseModel):
    results: list[CriterionAssessment] = Field(
        description="A list containing the assessment for every criteria evaluated."
    )
