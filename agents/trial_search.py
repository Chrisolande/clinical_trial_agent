import asyncio
import itertools
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from langchain_tavily import TavilySearch
from loguru import logger

from agents import query_helpers as _query_helpers
from clinical_trial_agent.clinical_trials import parse_trial_from_response, search_trials
from clinical_trial_agent.config import get_settings

settings = get_settings()


@lru_cache(maxsize=8)
def _get_tavily_tool(api_key: str, max_results: int) -> TavilySearch:
    """Cached TavilySearch instance keyed by active get_settings()."""
    return TavilySearch(
        api_key=api_key,
        max_results=max_results,
        include_domains=["clinicaltrials.gov"],
        search_depth="advanced",
    )


def _needs_ctgov_supplement(trial: dict[str, Any]) -> bool:
    raw = trial.get("eligibility_criteria_raw")
    if raw is None:
        return True
    return not bool(str(raw).strip())


def _build_tavily_query(trial: dict[str, Any]) -> str:
    nct_id = str(trial.get("nct_id", "")).strip()
    title = str(trial.get("brief_title", "")).strip()
    if nct_id:
        return f"{nct_id} clinicaltrials.gov eligibility criteria"
    return f"{title} clinical trial eligibility criteria clinicaltrials.gov"


def _extract_eligibility_snippet(search_result: Any) -> str | None:
    if isinstance(search_result, dict):
        content = search_result.get("content")
        if isinstance(content, str):
            cleaned = content.strip()
            if len(cleaned) >= 40:
                return cleaned[: settings.criteria_text_max_chars]
    if isinstance(search_result, str):
        cleaned = search_result.strip()
        if len(cleaned) >= 40:
            return cleaned[: get_settings().criteria_text_max_chars]
    return None


def _run_tavily_search(query: str) -> Any:
    """Invoke Tavily synchronously - intended to be called via asyncio.to_thread."""
    api_key = (
        settings.tavily_api_key.get_secret_value()
        if hasattr(settings.tavily_api_key, "get_secret_value")
        else str(settings.tavily_api_key)
    )
    return _get_tavily_tool(api_key, settings.tavily_max_results).invoke({"query": query})


def _extract_snippet_from_response_items(response: Any) -> str | None:
    items: Sequence[Any] = response if isinstance(response, list) else [response]
    return next((s for s in (_extract_eligibility_snippet(i) for i in items) if s), None)


def _merge_trial_with_snippet(trial: dict[str, Any], snippet: str) -> dict[str, Any]:
    merged = dict(trial)
    merged["eligibility_criteria_raw"] = snippet
    merged["criteria_source"] = "tavily_snippet"
    merged["criteria_source_verified"] = False
    merged["criteria_retrieved_at"] = datetime.now(UTC).isoformat()
    merged["criteria_completeness"] = "partial"
    return merged


def _build_supplement_candidates(
    trials: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    return [(idx, trial) for idx, trial in enumerate(trials) if _needs_ctgov_supplement(trial)][
        : settings.tavily_max_trials_to_enrich
    ]


async def _run_tavily_queries(
    candidates: list[tuple[int, dict[str, Any]]],
) -> list[Any]:
    queries = [_build_tavily_query(trial) for _, trial in candidates]
    results = await asyncio.gather(
        *[asyncio.to_thread(_run_tavily_search, query) for query in queries],
        return_exceptions=True,
    )
    return list(results)


def _apply_tavily_response(
    trial: dict[str, Any],
    response: Any,
) -> tuple[dict[str, Any] | None, bool]:
    if isinstance(response, Exception):
        logger.warning(
            "Tavily supplement failed for {}: {}",
            trial.get("nct_id", "unknown"),
            response,
        )
        return None, False

    snippet = _extract_snippet_from_response_items(response)
    if not snippet:
        return None, False
    return _merge_trial_with_snippet(trial, snippet), True


async def _supplement_trials_from_tavily(
    trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    api_key = (
        settings.tavily_api_key.get_secret_value()
        if hasattr(settings.tavily_api_key, "get_secret_value")
        else str(settings.tavily_api_key)
    )
    if (not settings.tavily_enable_ctgov_supplement) or not api_key:
        return trials, 0

    candidates = _build_supplement_candidates(trials)
    if not candidates:
        return trials, 0

    responses = await _run_tavily_queries(candidates)

    updated = list(trials)
    enriched_count = 0
    for (index, trial), response in zip(candidates, responses, strict=False):
        merged, enriched = _apply_tavily_response(trial, response)
        if merged:
            updated[index] = merged
        if enriched:
            enriched_count += 1

    return updated, enriched_count


def _log_query_exception(index: int, error: Exception) -> None:
    logger.warning("Error in query {}: {}", index, error)


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
            _log_query_exception(i, res)

    # Parse raw CT.gov v2 study objects before any downstream processing.
    # This ensures nct_id, eligibility_criteria_raw etc. are present for
    # _needs_ctgov_supplement and _build_tavily_query to work correctly.
    trials = _extract_studies_from_results(list(results))
    trials = [t for t in trials if t.get("nct_id")]

    supplemented_trials, supplemented_count = await _supplement_trials_from_tavily(trials)
    if supplemented_count:
        logger.info(
            "Tavily supplemented missing CT.gov eligibility text for {} trial(s).",
            supplemented_count,
        )
    return supplemented_trials


def build_search_queries(
    normalized_terms: dict[str, Any],
    patient_profile: dict[str, Any],
    include_not_yet_recruiting: bool = False,
) -> list[dict[str, Any]]:
    return _query_helpers.build_search_queries(
        normalized_terms, patient_profile, include_not_yet_recruiting
    )
