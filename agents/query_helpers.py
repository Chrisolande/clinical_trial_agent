"""Shared helpers for constructing broad, disease-agnostic trial search queries."""

from __future__ import annotations

import json
from typing import Any


DEFAULT_STATUSES = ["RECRUITING"]
BROAD_STATUSES = ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"]

MISSING_MARKERS = {
    "",
    "unknown",
    "not reported",
    "not yet performed",
    "missing",
    "not available",
    "pending",
}

NEGATIVE_PREFIXES = (
    "no ",
    "negative for ",
    "absence of ",
)


def _as_term(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = " ".join(_as_term(value).split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def _resolve_status(include_nyr: bool) -> list[str]:
    return BROAD_STATUSES if include_nyr else DEFAULT_STATUSES


def _looks_searchable_condition(term: str) -> bool:
    lowered = term.lower().strip()
    if not lowered:
        return False
    if lowered.startswith(NEGATIVE_PREFIXES):
        return False
    if "no known" in lowered or "not pregnant" in lowered:
        return False
    if len(lowered) > 140:
        return False
    return True


def _collect_condition_terms(
    normalized: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    terms: list[str] = []

    for value in normalized.get("primary_search_terms", []):
        term = _as_term(value)
        if _looks_searchable_condition(term):
            terms.append(term)

    primary = _as_term(profile.get("primary_condition"))
    if _looks_searchable_condition(primary):
        terms.append(primary)

    for condition in profile.get("conditions", []):
        term = _as_term(condition)
        if _looks_searchable_condition(term):
            terms.append(term)

    return _dedupe(terms)


def _collect_biomarker_terms(profile: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    for biomarker in profile.get("biomarkers", []):
        if not isinstance(biomarker, dict):
            term = _as_term(biomarker)
            if term:
                terms.append(term)
            continue

        name = _as_term(biomarker.get("name"))
        result = _as_term(biomarker.get("result"))
        result_key = result.lower()

        if not name or result_key in MISSING_MARKERS:
            continue

        # Keep clinically useful positive/actionable markers.
        text = " ".join(part for part in [name, result] if part)
        if text:
            terms.append(text)

    return _dedupe(terms)


def _collect_preference_terms(profile: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    for preference in profile.get("trial_preferences", []):
        text = _as_term(preference)
        if not text:
            continue

        lowered = text.lower()
        if "interested in " in lowered:
            terms.append(text.split("interested in ", 1)[-1])
        elif "prefers " in lowered:
            terms.append(text.split("prefers ", 1)[-1])
        else:
            terms.append(text)

    return _dedupe(terms)


def _query_key(query: dict[str, Any]) -> str:
    return json.dumps(query, sort_keys=True)


def _dedupe_queries(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for query in queries:
        cleaned = {k: v for k, v in query.items() if v not in (None, "", [])}
        key = _query_key(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)

    return out


def build_search_queries(
    normalized_terms: dict[str, Any],
    patient_profile: dict[str, Any],
    include_not_yet_recruiting: bool = False,
) -> list[dict[str, Any]]:
    """Build broad retrieval queries.

    Design rule:
    - Conditions and biomarkers belong in retrieval.
    - Current medications, borderline labs, allergies, and contraindications belong in eligibility.
    - Do not over-constrain the first query.
    """

    statuses = _resolve_status(include_not_yet_recruiting)

    conditions = _collect_condition_terms(normalized_terms, patient_profile)
    biomarkers = _collect_biomarker_terms(patient_profile)
    preferences = _collect_preference_terms(patient_profile)

    queries: list[dict[str, Any]] = []

    # 1. Broad disease-only queries. These are the most important.
    for condition in conditions[:3]:
        queries.append(
            {
                "condition": condition,
                "status": statuses,
            }
        )

    # 2. Disease + actionable biomarker free-text queries.
    for condition in conditions[:2]:
        for marker in biomarkers[:4]:
            queries.append(
                {
                    "term": f"{condition} {marker}",
                    "status": statuses,
                }
            )

    # 3. Disease + trial preference queries.
    for condition in conditions[:2]:
        for preference in preferences[:3]:
            queries.append(
                {
                    "term": f"{condition} {preference}",
                    "status": statuses,
                }
            )

    # 4. Last-resort biomarker-only search.
    for marker in biomarkers[:2]:
        queries.append(
            {
                "term": marker,
                "status": statuses,
            }
        )

    return _dedupe_queries(queries)[:10]