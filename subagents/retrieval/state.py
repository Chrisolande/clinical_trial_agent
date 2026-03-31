from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class RetrievalInput(TypedDict):
    normalized_terms: dict[str, Any] | None
    patient_profile: dict[str, Any] | None
    retry_count: int
    trials_raw: list[dict[str, Any]]


class RetrievalOutput(TypedDict):
    trials_raw: list[dict[str, Any]]
    trials_deduplicated: list[dict[str, Any]]
    search_queries: list[str]
    decision_history: list[str]
    errors: list[str]


class RetrievalState(TypedDict):
    normalized_terms: dict[str, Any] | None
    patient_profile: dict[str, Any] | None
    retry_count: int
    existing_nct_ids: list[str]
    trials_raw: list[dict[str, Any]]
    current_queries: list[dict[str, Any]]  # queries being executed
    include_not_yet_recruiting: bool
    internal_retry_count: int

    fetched_trials: Annotated[list[dict[str, Any]], operator.add]
    executed_query_strings: Annotated[list[str], operator.add]
    internal_decisions: Annotated[list[str], operator.add]
    internal_errors: Annotated[list[str], operator.add]

    trials_deduplicated: list[dict[str, Any]]  # full deduped list
    search_queries: list[str]
    decision_history: list[str]
    errors: list[str]
