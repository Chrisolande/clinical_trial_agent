from __future__ import annotations

import asyncio
import itertools
from typing import Any

from loguru import logger

# Ensure this import matches the name of your tool file
from tools.clinical_trial import search_trials


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
            logger.warning(f"Error in query {i}: {res}")

    # 2. search_trials already parsed the data, so just flatten the studies
    flat_studies = itertools.chain.from_iterable(
        r.get("studies", []) for r in results if isinstance(r, dict)
    )

    # 3. Filter out any empty results to ensure we only return valid trials
    return [trial for trial in flat_studies if trial.get("nct_id")]


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
