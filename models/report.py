from __future__ import annotations

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
    why_it_matches_evidence: list[ReportClaimEvidence] = Field(default_factory=list)
    main_blockers: list[str] = Field(
        description="Known blockers or likely barriers. Empty if none known."
    )
    main_blockers_evidence: list[ReportClaimEvidence] = Field(default_factory=list)
    key_uncertainties: list[str] = Field(
        description="Important missing confirmations needed before enrollment."
    )
    key_uncertainties_evidence: list[ReportClaimEvidence] = Field(default_factory=list)
    next_action: str = Field(description="The most useful next clinical action for this trial.")
    evidence_summary: str = Field(
        description="Short evidence-based explanation of the match assessment."
    )
    evidence_summary_refs: list[EvidenceRef] = Field(default_factory=list)


class EvidenceRef(BaseModel):
    source_type: Literal[
        "parsed_inclusion",
        "parsed_exclusion",
        "trial_metadata",
        "not_enough_evidence",
    ]
    trial_id: str
    criterion_id: str | None = None
    criterion_text: str | None = None
    metadata_field: str | None = None
    metadata_value: str | None = None
    verdict: Literal["MEETS", "FAILS", "UNCERTAIN"] | None = None
    note: str | None = None


class ReportClaimEvidence(BaseModel):
    claim_text: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


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
