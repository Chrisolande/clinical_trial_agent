from typing import Any

import openai
from loguru import logger
from models.judge_verdict import JudgeVerdict
from pydantic import ValidationError

from clinical_trial_agent.constants import ELIGIBILITY_TIMEOUT_FALLBACK_MESSAGE

from .eligibility_fallback_rules import (
    append_outcome,
    assess_exclusion,
    assess_inclusion,
    append_outcome as _append_outcome_impl,
    assess_exclusion as _assess_exclusion_impl,
    _assess_inclusion_age as _assess_inclusion_age_impl,
    _assess_inclusion_biomarker as _assess_inclusion_biomarker_impl,
    _assess_inclusion_melanoma as _assess_inclusion_melanoma_impl,
    _assess_inclusion_performance as _assess_inclusion_performance_impl,
    _extract_age_bound as _extract_age_bound_impl,
    evaluate_single_criterion,
    profile_blob,
    select_timeout_tier_and_score,
)

FALLBACK_VERDICT: dict[str, Any] = {
    "match_tier": "weak",
    "match_score": 0.1,
    "major_criteria_assessable": False,
    "inclusion_met": [],
    "inclusion_failed": [],
    "inclusion_uncertain": [],
    "exclusion_triggered": [],
    "exclusion_uncertain": [],
    "critical_missing_info": ["Eligibility could not be confidently assessed from available criteria."],
    "key_concern": "Eligibility assessment inconclusive",
    "rationale": "Eligibility assessment was inconclusive; conservative weak tier assigned.",
    "internal_fallback_used": True,
}

TIMEOUT_EXCEPTIONS = (
    TimeoutError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


def validate_verdict(verdict_dict: dict[str, Any], trial_id: str) -> JudgeVerdict:
    try:
        return JudgeVerdict.model_validate(verdict_dict)
    except (ValidationError, TypeError, ValueError):
        logger.warning("Invalid judge verdict schema for {}", trial_id)
        return JudgeVerdict.model_validate(dict(FALLBACK_VERDICT))


def _make_fallback(key_concern: str, rationale: str, missing: str) -> dict[str, Any]:
    fallback = dict(FALLBACK_VERDICT)
    fallback["key_concern"] = key_concern
    fallback["rationale"] = rationale
    fallback["critical_missing_info"] = [missing]
    return fallback


def deterministic_timeout_verdict(
    patient_profile: dict[str, Any],
    all_criteria: list[dict[str, Any]],
    trial_id: str,
) -> JudgeVerdict:
    profile_blob_text = profile_blob(patient_profile)
    inclusion_met: list[str] = []
    inclusion_failed: list[str] = []
    inclusion_uncertain: list[str] = []
    exclusion_triggered: list[str] = []
    exclusion_uncertain: list[str] = []
    missing: list[str] = []

    for crit in all_criteria:
        text = str(crit.get("text", "")).strip()
        if not text:
            continue
        ctype = str(crit.get("criteria_type", "inclusion")).lower()
        verdict, reason = evaluate_single_criterion(text, ctype, patient_profile, profile_blob_text)
        append_outcome(
            verdict,
            reason,
            text,
            is_exclusion=(ctype == "exclusion"),
            inclusion_met=inclusion_met,
            inclusion_failed=inclusion_failed,
            inclusion_uncertain=inclusion_uncertain,
            exclusion_triggered=exclusion_triggered,
            exclusion_uncertain=exclusion_uncertain,
            missing=missing,
        )

    total = (
        len(inclusion_met)
        + len(inclusion_failed)
        + len(inclusion_uncertain)
        + len(exclusion_triggered)
        + len(exclusion_uncertain)
    )
    tier, score = select_timeout_tier_and_score(exclusion_triggered, inclusion_failed, total)
    verdict_dict: dict[str, Any] = {
        "match_score": score,
        "match_tier": tier,
        "major_criteria_assessable": bool(inclusion_met),
        "inclusion_met": inclusion_met,
        "inclusion_failed": inclusion_failed,
        "inclusion_uncertain": inclusion_uncertain,
        "exclusion_triggered": exclusion_triggered,
        "exclusion_uncertain": exclusion_uncertain,
        "critical_missing_info": list(dict.fromkeys(missing))[:10]
        or ["Additional trial-specific data required."],
        "key_concern": ELIGIBILITY_TIMEOUT_FALLBACK_MESSAGE,
        "rationale": f"Eligibility assessment for {trial_id} timed out; conservative rule-based triage applied.",
        "internal_fallback_used": True,
    }
    return validate_verdict(verdict_dict, trial_id)


def fallback_verdict_for_exception(
    exc: Exception,
    trial_id: str,
    patient_profile: dict[str, Any],
    all_criteria: list[dict[str, Any]],
) -> JudgeVerdict:
    if isinstance(exc, TIMEOUT_EXCEPTIONS):
        logger.warning("Eligibility assessment timed out for {} ({})", trial_id, type(exc).__name__)
        return deterministic_timeout_verdict(patient_profile, all_criteria, trial_id)

    logger.opt(exception=False).error(
        "Eligibility assessment failed for {} ({}): {}",
        trial_id,
        type(exc).__name__,
        exc,
    )
    return validate_verdict(
        _make_fallback(
            "LLM response parsing failed",
            "LLM response parsing failed; conservative weak tier assigned.",
            "Additional clinical review recommended.",
        ),
        trial_id,
    )


def _extract_age_bound(text: str) -> tuple[int | None, int | None]:
    return _extract_age_bound_impl(text)


def _assess_inclusion_age(lowered: str, age: Any) -> tuple[str, str] | None:
    return _assess_inclusion_age_impl(lowered, age)


def _assess_inclusion_melanoma(lowered: str, profile_blob_text: str) -> tuple[str, str] | None:
    return _assess_inclusion_melanoma_impl(lowered, profile_blob_text)


def _assess_inclusion_biomarker(lowered: str, patient_profile: dict[str, Any]) -> tuple[str, str] | None:
    return _assess_inclusion_biomarker_impl(lowered, patient_profile)


def _assess_inclusion_performance(lowered: str, patient_profile: dict[str, Any]) -> tuple[str, str] | None:
    return _assess_inclusion_performance_impl(lowered, patient_profile)


def _assess_exclusion(text: str, patient_profile: dict[str, Any], profile_blob_text: str) -> tuple[str, str]:
    return _assess_exclusion_impl(text, patient_profile, profile_blob_text)


def _append_outcome(
    verdict: str,
    reason: str,
    text: str,
    *,
    is_exclusion: bool,
    inclusion_met: list[str],
    inclusion_failed: list[str],
    inclusion_uncertain: list[str],
    exclusion_triggered: list[str],
    exclusion_uncertain: list[str],
    missing: list[str],
) -> None:
    _append_outcome_impl(
        verdict,
        reason,
        text,
        is_exclusion=is_exclusion,
        inclusion_met=inclusion_met,
        inclusion_failed=inclusion_failed,
        inclusion_uncertain=inclusion_uncertain,
        exclusion_triggered=exclusion_triggered,
        exclusion_uncertain=exclusion_uncertain,
        missing=missing,
    )


def _select_timeout_tier_and_score(
    exclusion_triggered: list[str], inclusion_failed: list[str], total: int
) -> tuple[str, float]:
    return select_timeout_tier_and_score(exclusion_triggered, inclusion_failed, total)
