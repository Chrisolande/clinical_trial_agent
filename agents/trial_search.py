import asyncio
import itertools
from typing import Any

from loguru import logger

from agents import query_helpers as _query_helpers
from clinical_trial_agent.clinical_trials import parse_trial_from_response, search_trials


def _extract_studies_from_results(results: list[Any]) -> list[dict[str, Any]]:
    """Normalize search results into flat trial dicts.

    `search_trials` already returns parsed trials under `studies`, but this helper
    also supports raw CT.gov study payloads for compatibility.
    """
    flat_items = itertools.chain.from_iterable(
        r.get("studies", []) for r in results if isinstance(r, dict)
    )
    trials: list[dict[str, Any]] = []
    for item in flat_items:
        if not isinstance(item, dict):
            continue
        if item.get("protocolSection"):
            trials.append(parse_trial_from_response(item))
            continue
        # Already normalized trial row from clinical_trials.search_trials
        if item.get("nct_id"):
            trials.append(item)
    return trials


async def _run_search_queries(
    queries: list[dict[str, Any]], page_size: int
) -> list[dict[str, Any]]:
    """Run multiple search queries concurrently and return parsed, flat trial dicts."""
    tasks = [
        search_trials(
            condition=query.get("condition"),
            intervention=query.get("intervention"),
            term=query.get("term"),
            status=query.get("status"),
            page_size=page_size,
        )
        for query in queries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.warning("Error in query {}: {}", i, res)
    results_dicts = [r for r in results if isinstance(r, dict)]

    trials = _extract_studies_from_results(results_dicts)
    trials = [t for t in trials if t.get("nct_id")]
    return trials


def build_search_queries(
    normalized_terms: dict[str, Any],
    patient_profile: dict[str, Any],
    include_not_yet_recruiting: bool = False,
) -> list[dict[str, Any]]:
    return _query_helpers.build_search_queries(
        normalized_terms, patient_profile, include_not_yet_recruiting
    )
