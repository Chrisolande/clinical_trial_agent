from typing import Any

from config import get_settings
from loguru import logger


def refine_search_strategy(
    normalized_terms: dict[str, Any],
    patient_profile: dict[str, Any],
    retry_count: int,
    current_trial_count: int,
) -> dict[str, Any]:
    """Determine refined search strategy based on retry number.

    Retry 1: Broaden condition terms (add synonyms, broader terms)
    Retry 2: Include NOT_YET_RECRUITING trials
    Retry 3: Use related/comorbid conditions
    """
    strategy: dict[str, Any] = {
        "include_not_yet_recruiting": False,
        "use_broader_terms": False,
        "use_related_conditions": False,
        "refined_terms": dict(normalized_terms),
        "decision_note": "",
    }

    attempt_num = retry_count + 1
    trial_count_note = f" after {current_trial_count} trial(s) found so far"

    if retry_count == 0:
        # First retry: broaden condition terms using synonyms
        strategy["use_broader_terms"] = True
        broader_terms = _get_broader_terms(normalized_terms, patient_profile)
        strategy["refined_terms"]["primary_search_terms"] = broader_terms
        strategy["decision_note"] = (
            f"Refinement attempt {attempt_num}: broadening search with synonym terms "
            f"{broader_terms[:3]}{trial_count_note}"
        )
        logger.info("Refine strategy #1: broader terms {}", broader_terms[:3])

    elif retry_count == 1:
        # Second retry: include NOT_YET_RECRUITING trials
        strategy["include_not_yet_recruiting"] = True
        strategy["decision_note"] = (
            f"Refinement attempt {attempt_num}: including trials not yet recruiting "
            f"{trial_count_note}"
        )
        logger.info("Refine strategy #2: include not yet recruiting trials")
    else:
        # Third retry: try related conditions
        strategy["use_related_conditions"] = True
        related = _get_related_conditions(normalized_terms, patient_profile)
        if related:
            existing = strategy["refined_terms"].get("primary_search_terms", [])
            strategy["refined_terms"]["primary_search_terms"] = existing + related
            strategy["decision_note"] = (
                f"Refinement attempt {attempt_num}: searching related conditions as fallback"
                f"{trial_count_note}"
            )
            logger.info("Refine strategy #3: related conditions {}", related)

    return strategy


def _get_broader_terms(
    normalized_terms: dict[str, Any], patient_profile: dict[str, Any]
) -> list[str]:
    terms: list[str] = []
    for cond in normalized_terms.get("conditions", {}).values():
        if isinstance(cond, dict):
            for key in ("broader_terms", "synonyms", "search_terms"):
                terms.extend([str(t) for t in cond.get(key, [])[:2]])  # Take up to 2 per category

    # Fall back to profile if terms is empty
    profile_conditions = [str(c) for c in patient_profile.get("conditions", [])[:3]]
    terms = terms or profile_conditions

    return list(dict.fromkeys(filter(None, terms)))[:5]


def _get_related_conditions(
    normalized_terms: dict[str, Any], patient_profile: dict[str, Any]
) -> list[str]:
    related: list[str] = []

    for cond_data in normalized_terms.get("conditions", {}).values():
        if isinstance(cond_data, dict):
            narrower = cond_data.get("narrower_terms", [])
            related.extend(narrower[:2])

    history = [str(h) for h in patient_profile.get("medical_history", [])]
    related.extend(history[:2])
    seen: set[str] = set()
    unique: list[str] = []
    for t in related:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:3]


def should_continue_refining(retry_count: int) -> bool:
    """Return True if more refinement attempts are available."""
    return retry_count < get_settings().max_retry_attempts
