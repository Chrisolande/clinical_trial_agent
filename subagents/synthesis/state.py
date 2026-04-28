"""State schemas for the Synthesis sub-agent."""

import operator
from typing import Annotated, Any, TypedDict


class QAIssue(TypedDict):
    code: str
    severity: str
    message: str


class SynthesisInput(TypedDict):
    """Keys the supervisor passes INTO the synthesis sub-agent."""

    patient_profile: dict[str, Any] | None
    trial_scores: list[dict[str, Any]] | None
    eligibility_verdicts: dict[str, dict[str, Any]] | None
    missing_info_recommendations: list[dict[str, Any]] | None
    trials_raw: list[dict[str, Any]]  # for report context
    search_queries: list[str]  # for report methodology
    decision_history: list[str]  # full history (for report, read-only)
    retrieval_errors: list[str] | None
    trials_with_criteria: list[dict[str, Any]] | None


class SynthesisOutput(TypedDict):
    """Keys the synthesis sub-agent writes BACK to supervisor state."""

    qa_issues: list[QAIssue]
    qa_passed: bool
    report_json: dict[str, Any] | None
    report_text: str | None
    synthesis_needs_re_evaluation: bool  # signal to supervisor
    synthesis_retry_retrieval: bool
    qa_remediation: dict[str, Any]
    decision_history: list[str]  # NEW entries only (parent accumulates)
    errors: list[str]  # new errors (parent accumulates)


class SynthesisState(TypedDict):
    """Full internal state for the synthesis sub-agent."""

    # From input
    patient_profile: dict[str, Any] | None
    trial_scores: list[dict[str, Any]] | None
    eligibility_verdicts: dict[str, dict[str, Any]] | None
    missing_info_recommendations: list[dict[str, Any]] | None
    trials_raw: list[dict[str, Any]]
    search_queries: list[str]
    decision_history: list[str]  # full history (read-only internally)
    retrieval_errors: list[str] | None
    trials_with_criteria: list[dict[str, Any]] | None

    # Internal state
    qa_fix_attempts: int

    # Output fields
    qa_issues: list[QAIssue]
    qa_passed: bool
    report_json: dict[str, Any] | None
    report_text: str | None
    synthesis_needs_re_evaluation: bool
    synthesis_retry_retrieval: bool
    qa_remediation: dict[str, Any] | None
    qa_unresolved_issues: list[QAIssue]
    qa_remediation_actions: Annotated[list[dict[str, str]], operator.add]
    new_decision_entries: Annotated[list[str], operator.add]  # internal accumulator
    new_errors: Annotated[list[str], operator.add]  # internal accumulator
