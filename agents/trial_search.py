from __future__ import annotations

import asyncio
import itertools
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, cast

from clinical_trials import parse_trial_from_response, search_trials
from config import settings
from langchain_tavily import TavilySearch
from loguru import logger


@lru_cache(maxsize=8)
def _get_tavily_tool(api_key: str, max_results: int) -> TavilySearch:
    """Cached TavilySearch instance keyed by active settings."""
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
            return cleaned[: settings.criteria_text_max_chars]
    return None


def _run_tavily_search(query: str) -> Any:
    """Invoke Tavily synchronously - intended to be called via asyncio.to_thread."""
    return _get_tavily_tool(settings.tavily_api_key, settings.tavily_max_results).invoke(
        {"query": query}
    )


def _extract_snippet_from_response_items(response: Any) -> str | None:
    items: Sequence[Any] = response if isinstance(response, list) else [response]
    return next((s for s in (_extract_eligibility_snippet(i) for i in items) if s), None)


def _merge_trial_with_snippet(trial: dict[str, Any], snippet: str) -> dict[str, Any]:
    merged = dict(trial)
    merged["eligibility_criteria_raw"] = snippet
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
    return cast(
        "list[Any]",
        await asyncio.gather(
            *[asyncio.to_thread(_run_tavily_search, query) for query in queries],
            return_exceptions=True,
        ),
    )


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
    if not settings.tavily_enable_ctgov_supplement or not settings.tavily_api_key:
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


def _as_term(value: Any) -> str:
    return str(value).strip()


def _collect_intervention_terms(
    normalized: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    terms: list[str] = []
    for value in normalized.get("intervention_search_terms", []):
        term = _as_term(value)
        if term:
            terms.append(term)

    for medication in profile.get("medications", []):
        if isinstance(medication, dict):
            term = _as_term(medication.get("name", ""))
        else:
            term = _as_term(medication)
        if term:
            terms.append(term)

    for treatment in profile.get("prior_treatments", []):
        term = _as_term(treatment)
        if term:
            terms.append(term)
    return list(dict.fromkeys(terms))


def _collect_biomarker_terms(profile: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for biomarker in profile.get("biomarkers", []):
        if isinstance(biomarker, dict):
            name = _as_term(biomarker.get("name", ""))
            result = _as_term(biomarker.get("result", ""))
            text = " ".join(part for part in [name, result] if part)
        else:
            text = _as_term(biomarker)
        if text:
            terms.append(text)
    return list(dict.fromkeys(terms))


def _condition_variants(normalized: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    base_terms = [_as_term(t) for t in normalized.get("primary_search_terms", []) if _as_term(t)]
    primary = _as_term(profile.get("primary_condition", ""))
    if primary:
        base_terms.append(primary)
    base_terms.extend(_as_term(c) for c in profile.get("conditions", []) if _as_term(c))
    deduped_base = list(dict.fromkeys(base_terms))
    if not deduped_base:
        return []

    biomarkers = _collect_biomarker_terms(profile)
    variants: list[str] = list(deduped_base)
    # Build focused condition variants (e.g., "colorectal adenocarcinoma MSI-H").
    for base in deduped_base[:2]:
        for marker in biomarkers[:2]:
            variants.append(f"{base} {marker}")
    return list(dict.fromkeys(variants))


def _build_primary_query(
    normalized: dict[str, Any],
    profile: dict[str, Any],
    primary: str | None,
    statuses: list[str],
) -> dict[str, Any] | None:
    condition_terms = _condition_variants(normalized, profile)
    if not condition_terms and not primary:
        return None
    interventions = _collect_intervention_terms(normalized, profile)
    query: dict[str, Any] = {
        "condition": condition_terms[0] if condition_terms else _as_term(primary),
        "status": statuses,
    }
    # Bias first query toward treatment trials when intervention signals exist.
    if interventions:
        query["intervention"] = interventions[0]
    return query


def _build_intervention_query(
    normalized: dict[str, Any],
    profile: dict[str, Any],
    primary: str | None,
    statuses: list[str],
) -> dict[str, Any] | None:
    itrms = _collect_intervention_terms(normalized, profile)
    if not itrms:
        return None
    condition_terms = _condition_variants(normalized, profile)
    preferred_intervention = itrms[1] if len(itrms) > 1 else itrms[0]
    base: dict[str, Any] = {"intervention": preferred_intervention, "status": statuses}
    if condition_terms:
        base["condition"] = condition_terms[0]
    elif primary:
        base["condition"] = _as_term(primary)
    return base


def _build_fallback_query(
    normalized: dict[str, Any],
    profile: dict[str, Any],
    primary: str | None,
    statuses: list[str],
) -> dict[str, Any] | None:
    terms = _condition_variants(normalized, profile)
    conds = profile.get("conditions", [])
    if len(terms) > 1:
        return {"condition": terms[1], "status": statuses}
    if len(conds) > 1:
        return {"condition": _as_term(conds[1]), "status": statuses}
    if primary:
        return {
            "condition": _as_term(primary),
            "status": ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"],
        }
    return None


def build_search_queries(
    normalized_terms: dict[str, Any],
    patient_profile: dict[str, Any],
    include_not_yet_recruiting: bool = False,
) -> list[dict[str, Any]]:
    """Build focused retrieval queries using condition + intervention + biomarker signals."""
    statuses = _resolve_status(include_not_yet_recruiting)
    primary = _primary_condition(normalized_terms, patient_profile)
    candidates = [
        _build_primary_query(normalized_terms, patient_profile, primary, statuses),
        _build_intervention_query(normalized_terms, patient_profile, primary, statuses),
        _build_fallback_query(normalized_terms, patient_profile, primary, statuses),
    ]
    return [q for q in candidates if q is not None][:3]
