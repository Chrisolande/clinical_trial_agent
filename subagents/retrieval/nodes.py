from typing import Any

from agents import search_refiner, trial_search
from loguru import logger

from clinical_trial_agent.config import get_settings

from .state import RetrievalState


def _get_new_unique_trials(
    fetched: list[dict[str, Any]], existing_ids: set[str] | list[str]
) -> list[dict[str, Any]]:
    existing = set(existing_ids)
    return list(
        {
            str(nct_id): t
            for t in fetched
            if (nct_id := t.get("nct_id")) and str(nct_id) not in existing
        }.values()
    )


async def initialize_retrieval(state: RetrievalState) -> dict[str, Any]:
    retry = state.get("retry_count", 0)
    normalized_terms = state.get("normalized_terms") or {}
    patient_profile = state.get("patient_profile") or {}

    existing_nct_ids = [t["nct_id"] for t in state.get("trials_raw") or [] if t.get("nct_id")]
    include_nyr = retry >= 1

    if retry >= 1:
        strategy = search_refiner.refine_search_strategy(
            normalized_terms=normalized_terms,
            patient_profile=patient_profile,
            retry_count=retry - 1,
            current_trial_count=0,
        )
        effective_terms = strategy["refined_terms"]
        include_nyr = strategy.get("include_not_yet_recruiting", include_nyr)
        decision_note = strategy["decision_note"]
    else:
        effective_terms = normalized_terms
        decision_note = "Initial search with primary terms."

    queries = trial_search.build_search_queries(
        effective_terms, patient_profile, include_not_yet_recruiting=include_nyr
    )

    return {
        "existing_nct_ids": existing_nct_ids,
        "current_queries": queries,
        "include_not_yet_recruiting": include_nyr,
        "internal_retry_count": 0,
        "normalized_terms": effective_terms,
        "internal_decisions": [
            f"Retrieval initialized (supervisor_retry={retry}): {decision_note}"
        ],
    }


async def execute_searches(state: RetrievalState) -> dict[str, Any]:
    """Execute all queries against ClinicalTrials.gov"""
    queries = state.get("current_queries", [])

    try:
        fetched = await trial_search._run_search_queries(
            queries, get_settings().max_trials_per_query
        )
        return {
            "fetched_trials": fetched,
            "executed_query_strings": [str(q) for q in queries],
            "internal_decisions": [
                f"Executed {len(queries)} queries, got {len(fetched)} raw results."
            ],
        }
    except Exception as exc:
        logger.error("execute_searches failed: {}", exc)
        return {
            "fetched_trials": [],
            "executed_query_strings": [],
            "internal_errors": [f"execute_searches: {exc}"],
            "internal_decisions": [f"Search execution failed: {exc}"],
        }


async def assess_and_finalize(state: RetrievalState) -> dict[str, Any]:
    """Build final output for the soupervisor"""

    fetched = state.get("fetched_trials") or []
    existing_ids = set(state.get("existing_nct_ids") or [])

    new_trials = _get_new_unique_trials(fetched, existing_ids)
    decisions = list(state.get("internal_decisions") or [])
    decisions.append(
        f"Retrieval complete: {len(new_trials)} new unique trials "
        f"(fetched: {len(fetched)}, within-batch dupes: {len(fetched) - len(new_trials)})."
    )
    return {
        "trials_raw": new_trials,
        "trials_deduplicated": new_trials,
        "search_queries": list(state.get("executed_query_strings") or []),
        "decision_history": decisions,
        "errors": list(state.get("internal_errors") or []),
    }


def should_retry_search(state: RetrievalState) -> str:
    """Route: if too few unique new results and internal retries available, try again."""
    if get_settings().one_pass_mode:
        return "finalize"
    fetched = state.get("fetched_trials") or []
    existing_ids = set(state.get("existing_nct_ids") or [])

    unique_new_count = len(_get_new_unique_trials(fetched, existing_ids))
    if (
        unique_new_count < 3
        and state.get("internal_retry_count", 0) < get_settings().retrieval_internal_max_retries
    ):
        return "retry_search"
    return "finalize"


async def broaden_and_retry(state: RetrievalState) -> dict[str, Any]:
    """Broaden search parameters and re-prepare queries for retry"""
    internal_retry = state.get("internal_retry_count", 0)
    strategy = search_refiner.refine_search_strategy(
        normalized_terms=state.get("normalized_terms") or {},
        patient_profile=state.get("patient_profile") or {},
        retry_count=internal_retry,
        current_trial_count=len(state.get("fetched_trials") or []),
    )

    return {
        "current_queries": trial_search.build_search_queries(
            strategy["refined_terms"],
            state.get("patient_profile") or {},
            include_not_yet_recruiting=strategy.get("include_not_yet_recruiting", False),
        ),
        "internal_retry_count": internal_retry + 1,
        "normalized_terms": strategy["refined_terms"],
        "internal_decisions": [f"Internal retry {internal_retry + 1}: {strategy['decision_note']}"],
    }
