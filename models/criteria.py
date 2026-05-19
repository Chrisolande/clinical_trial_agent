from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

CriterionCategory = Literal[
    "age", "lab", "biomarker", "diagnosis", "medication", "performance", "other"
]


class EligibilityCriterion(BaseModel):
    text: str = Field(description="Exact Criterion text, cleaned and atomic")
    is_hard_exclusion: bool = Field(
        description="True for all exclusion criteria and critical safety exclusions."
    )
    category: CriterionCategory = Field(description="Clinical category of the criterion.")

    @model_validator(mode="before")
    @classmethod
    def normalize_text_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("text"):
            return data

        for alias in ("description", "criterion", "criteria", "requirement"):
            value = data.get(alias)
            if isinstance(value, str) and value.strip():
                return {**data, "text": value.strip()}

        return data


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
