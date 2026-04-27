"""Shared helpers for constructing trial search queries."""

from typing import Any


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
    base: dict[str, Any] = {
        "intervention": itrms[1] if len(itrms) > 1 else itrms[0],
        "status": statuses,
    }
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
