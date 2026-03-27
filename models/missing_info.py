from typing import Literal

from pydantic import BaseModel, Field


class CompletenessAssessment(BaseModel):
    field: str = Field(
        description="The specific clinical field missing (e.g., 'EGFR mutation status', 'creatinine clearance')"
    )
    description: str = Field(
        description="Clinical rationale for why this information is needed and what specific verdicts it would clarify."
    )
    affected_trial_ids: list[str] = Field(
        description="List of NCT IDs for the trials where this missing data caused an UNCERTAIN verdict."
    )
    priority: Literal["high", "medium", "low"] = Field(
        description="Priority level based on how many trials or strong matches are affected."
    )


class CompletenessAssessmentList(BaseModel):
    results: list[CompletenessAssessment] = Field(
        description="List of the top missing data items, limited to the 10 most impactful."
    )
