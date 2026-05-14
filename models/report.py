from typing import Literal

from pydantic import BaseModel, Field

ReportTier = Literal["strong", "moderate", "weak", "disqualified"]
GapPriority = Literal["high", "medium", "low"]


class TrialReportCard(BaseModel):
    nct_id: str
    title: str
    tier: ReportTier
    score: float
    phase: str | None = None
    status: str | None = None

    recommendation: str = Field(
        description="One concise clinician-facing recommendation for this trial."
    )
    why_it_matches: list[str] = Field(
        description="Specific patient-trial fit reasons grounded in supplied evidence."
    )
    main_blockers: list[str] = Field(
        description="Known blockers or likely barriers. Empty if none known."
    )
    key_uncertainties: list[str] = Field(
        description="Important missing confirmations needed before enrollment."
    )
    next_action: str = Field(
        description="The most useful next clinical action for this trial."
    )
    evidence_summary: str = Field(
        description="Short evidence-based explanation of the match assessment."
    )


class ReportGap(BaseModel):
    item: str
    priority: GapPriority
    reason: str
    affects_trials: list[str]
    action: str
    applicable_to_patient: bool = True


class ReportPlan(BaseModel):
    patient_summary: str
    executive_summary: str
    bottom_line: str

    strong_matches: list[TrialReportCard]
    moderate_matches: list[TrialReportCard]

    information_gaps: list[ReportGap]
    recommended_actions: list[ReportGap]

    excluded_summary: str
    limitations: list[str]
