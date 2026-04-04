import operator
from typing import Annotated, Any, TypedDict


class EligibilityInput(TypedDict):
    patient_profile: dict[str, Any] | None
    trials_deduplicated: list[dict[str, Any]] | None
    eligibility_verdicts: dict[str, dict[str, Any]] | None


class EligibilityOutput(TypedDict):
    trials_with_criteria: list[dict[str, Any]]
    eligibility_verdicts: dict[str, dict[str, Any]]
    trial_scores: list[dict[str, Any]]
    viable_trial_count: int
    missing_info_recommendations: list[dict[str, Any]]
    retrieval_needs_broadening: bool
    decision_history: list[str]
    errors: list[str]


class TrialWorkerState(TypedDict):
    patient_profile: dict[str, Any]
    trial: dict[str, Any]


class EligibilityState(TypedDict):
    patient_profile: dict[str, Any] | None
    trials_deduplicated: list[dict[str, Any]] | None
    eligibility_verdicts: dict[str, dict[str, Any]] | None
    trials_to_evaluate: list[dict[str, Any]] | None
    processed_trials_with_criteria: Annotated[list[dict[str, Any]], operator.add]
    processed_verdicts: Annotated[list[dict[str, Any]], operator.add]

    trials_with_criteria: list[dict[str, Any]]
    trial_scores: list[dict[str, Any]]
    viable_trial_count: int
    missing_info_recommendations: list[dict[str, Any]]
    retrieval_needs_broadening: bool
    decision_history: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
