from __future__ import annotations

import asyncio
import itertools
from collections.abc import Sequence
from typing import Any

from clinical_trials import search_trials
from config import settings
from langchain_tavily import TavilySearch
from loguru import logger


def _needs_ctgov_supplement(trial: dict[str, Any]) -> bool:
    return not bool(trial.get("eligibility_criteria_raw"))


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
            if len(cleaned) >= 80:
                return cleaned[: settings.criteria_text_max_chars]
    if isinstance(search_result, str):
        cleaned = search_result.strip()
        if len(cleaned) >= 80:
            return cleaned[: settings.criteria_text_max_chars]
    return None


def _run_tavily_search(query: str) -> Any:
    tool = TavilySearch(
        max_results=settings.tavily_max_results,
        include_domains=["clinicaltrials.gov"],
        search_depth="advanced",
    )
    return tool.invoke({"query": query, "api_key": settings.tavily_api_key})


async def _supplement_trials_from_tavily(
    trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    if not settings.tavily_enable_ctgov_supplement or not settings.tavily_api_key:
        return trials, 0

    candidates: list[tuple[int, dict[str, Any]]] = [
        (idx, trial) for idx, trial in enumerate(trials) if _needs_ctgov_supplement(trial)
    ][: settings.tavily_max_trials_to_enrich]
    if not candidates:
        return trials, 0

    queries = [_build_tavily_query(trial) for _, trial in candidates]
    responses = await asyncio.gather(
        *[asyncio.to_thread(_run_tavily_search, query) for query in queries],
        return_exceptions=True,
    )

    updated = list(trials)
    enriched_count = 0
    for (index, trial), response in zip(candidates, responses, strict=False):
        if isinstance(response, Exception):
            logger.warning(
                "Tavily supplement failed for {}: {}",
                trial.get("nct_id", "unknown"),
                response,
            )
            continue

        items: Sequence[Any] = response if isinstance(response, list) else [response]

        snippet = next((s for s in (_extract_eligibility_snippet(i) for i in items) if s), None)
        if snippet:
            merged = dict(trial)
            merged["eligibility_criteria_raw"] = snippet
            updated[index] = merged
            enriched_count += 1

    return updated, enriched_count


async def _run_search_queries(
    queries: list[dict[str, Any]], page_size: int
) -> list[dict[str, Any]]:
    """Run multiple search queries concurrently and return a flat list of trial dicts."""
    tasks = [
        search_trials(
            condition=query.get("condition"),
            intervention=query.get("intervention"),
            status=query.get("status"),
            page_size=page_size,
        )
        for query in queries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 1. log any API Exceptions
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.warning("Error in query {}: {}", i, res)

    # 2. search_trials already parsed the data, so just flatten the studies
    flat_studies = itertools.chain.from_iterable(
        r.get("studies", []) for r in results if isinstance(r, dict)
    )

    # 3. Filter out any empty results to ensure we only return valid trials
    trials = [trial for trial in flat_studies if trial.get("nct_id")]

    supplemented_trials, supplemented_count = await _supplement_trials_from_tavily(trials)
    if supplemented_count:
        logger.info(
            "Tavily supplemented missing CT.gov eligibility text for {} trial(s).",
            supplemented_count,
        )
    return supplemented_trials


def _resolve_status(include_nyr: bool) -> list[str]:
    return ["RECRUITING", "NOT_YET_RECRUITING"] if include_nyr else ["RECRUITING"]


def _primary_condition(normalized: dict[str, Any], profile: dict[str, Any]) -> str | None:
    terms = normalized.get("primary_search_terms", [])
    conds = profile.get("conditions", [])

    return (
        (terms[0] if terms else None)
        or profile.get("primary_condition")
        or (conds[0] if conds else None)
    )


def _build_primary_query(
    normalized: dict[str, Any], primary: str | None, statuses: list[str]
) -> dict[str, Any] | None:
    terms = normalized.get("primary_search_terms", [])
    if not primary:
        return None
    return {"condition": terms[0] if terms else primary, "status": statuses}


def _build_intervention_query(
    normalized: dict[str, Any], primary: str | None, statuses: list[str]
) -> dict[str, Any] | None:
    itrms = normalized.get("intervention_search_terms", [])
    if not itrms:
        return None
    base = {"intervention": itrms[0], "status": statuses}
    if primary:
        base["condition"] = primary
    return base


def _build_fallback_query(
    normalized: dict[str, Any], profile: dict[str, Any], primary: str | None, statuses: list[str]
) -> dict[str, Any] | None:
    terms = normalized.get("primary_search_terms", [])
    conds = profile.get("conditions", [])
    if len(terms) > 1:
        return {"condition": terms[1], "status": statuses}
    if len(conds) > 1:
        return {"condition": conds[1], "status": statuses}
    if primary:
        return {
            "condition": primary,
            "status": ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"],
        }
    return None


def build_search_queries(
    normalized_terms: dict[str, Any],
    patient_profile: dict[str, Any],
    include_not_yet_recruiting: bool = False,
) -> list[dict[str, Any]]:
    """Build 3 complementary search queries from normalised terminology."""
    statuses = _resolve_status(include_not_yet_recruiting)
    primary = _primary_condition(normalized_terms, patient_profile)
    candidates = [
        _build_primary_query(normalized_terms, primary, statuses),
        _build_intervention_query(normalized_terms, primary, statuses),
        _build_fallback_query(normalized_terms, patient_profile, primary, statuses),
    ]
    return [q for q in candidates if q is not None][:3]
