import re
from typing import Any

import openai
from constants import ELIGIBILITY_TIMEOUT_FALLBACK_MESSAGE
from loguru import logger
from models.judge_verdict import JudgeVerdict
from pydantic import ValidationError

FALLBACK_VERDICT: dict[str, Any] = {
    "match_tier": "weak",
    "match_score": 0.1,
    "major_criteria_assessable": False,
    "inclusion_met": [],
    "inclusion_failed": [],
    "inclusion_uncertain": [],
    "exclusion_triggered": [],
    "exclusion_uncertain": [],
    "critical_missing_info": ["Insufficient structured verdict from judge model."],
    "key_concern": "LLM response parsing failed",
    "rationale": "Could not parse model verdict; conservative weak fallback applied.",
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


def _to_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _profile_blob(patient_profile: dict[str, Any]) -> str:
    keys = [
        "primary_condition",
        "conditions",
        "stage",
        "medical_history",
        "medications",
        "biomarkers",
        "ecog_performance_status",
    ]
    return " ".join(_to_text(patient_profile.get(k, "")) for k in keys).lower()


def _extract_age_bound(text: str) -> tuple[int | None, int | None]:
    lowered = text.lower()
    min_age: int | None = None
    max_age: int | None = None

    ge_match = re.search(r"(?:>=|≥|at least)\s*(\d{1,3})", lowered)
    if ge_match:
        min_age = int(ge_match.group(1))

    le_match = re.search(r"(?:<=|≤|at most|up to)\s*(\d{1,3})", lowered)
    if le_match:
        max_age = int(le_match.group(1))

    between_match = re.search(r"between\s+(\d{1,3})\s+(?:and|to)\s+(\d{1,3})", lowered)
    if between_match:
        min_age = int(between_match.group(1))
        max_age = int(between_match.group(2))

    return min_age, max_age


def _keyword_reason_mappings() -> tuple[tuple[tuple[str, ...], str], ...]:
    return (
        (
            ("ecog", "karnofsky", "performance status"),
            "Missing ECOG/performance status documentation",
        ),
        (("creatinine", "egfr", "renal"), "Missing renal function labs (creatinine/eGFR)"),
        (("ast", "alt", "bilirubin", "liver"), "Missing liver function labs (AST/ALT/bilirubin)"),
        (
            ("anc", "hemoglobin", "platelet", "cbc", "hematolog"),
            "Missing CBC/hematology labs (ANC/Hb/platelets)",
        ),
        (("ldh",), "Missing LDH value"),
        (("braf", "pd-l1", "biomarker", "mutation"), "Missing molecular biomarker result"),
        (
            ("recist", "measurable lesion", "imaging", "measurable disease"),
            "Missing baseline imaging/measurable disease assessment",
        ),
        (("histology", "pathology", "diagnosis"), "Missing pathology/diagnosis confirmation"),
        (("bleeding", "hemorrhage"), "Missing recent bleeding history"),
        (("cardiac", "ecg", "echo"), "Missing cardiac history/assessment"),
        (("retinal", "ophthalm"), "Missing ophthalmologic/retinal assessment"),
    )


def _missing_reason_from_criterion(text: str, *, exclusion: bool = False) -> str:
    lowered = text.lower()
    for keywords, message in _keyword_reason_mappings():
        if any(keyword in lowered for keyword in keywords):
            return message
    return (
        "Missing trial-specific clinical detail"
        if not exclusion
        else "Missing exclusion-history detail"
    )


def _assess_inclusion(
    text: str, patient_profile: dict[str, Any], profile_blob: str
) -> tuple[str, str]:
    lowered = text.lower()
    age_result = _assess_inclusion_age(lowered, patient_profile.get("age"))
    if age_result is not None:
        return age_result

    melanoma_result = _assess_inclusion_melanoma(lowered, profile_blob)
    if melanoma_result is not None:
        return melanoma_result

    biomarker_result = _assess_inclusion_biomarker(lowered, profile_blob)
    if biomarker_result is not None:
        return biomarker_result

    performance_result = _assess_inclusion_performance(lowered, patient_profile)
    if performance_result is not None:
        return performance_result

    return "UNCERTAIN", _missing_reason_from_criterion(text, exclusion=False)


def _assess_inclusion_age(lowered: str, age: Any) -> tuple[str, str] | None:
    if "age" not in lowered or not isinstance(age, int | float):
        return None
    min_age, max_age = _extract_age_bound(lowered)
    if min_age is not None and age < min_age:
        return "FAILS", "Age below required threshold"
    if max_age is not None and age > max_age:
        return "FAILS", "Age above allowed threshold"
    return "MEETS", "Age appears within stated bounds"


def _assess_inclusion_melanoma(lowered: str, profile_blob: str) -> tuple[str, str] | None:
    if "melanoma" not in lowered:
        return None
    if "melanoma" in profile_blob:
        return "MEETS", "Diagnosis mentions melanoma and patient condition includes melanoma"
    return "UNCERTAIN", "Melanoma diagnosis not explicit in profile"


def _assess_inclusion_biomarker(lowered: str, profile_blob: str) -> tuple[str, str] | None:
    biomarker_keywords = ("braf", "pd-l1", "biomarker", "mutation")
    if not any(keyword in lowered for keyword in biomarker_keywords):
        return None
    if any(keyword in profile_blob for keyword in ("braf", "pd-l1", "mutation")):
        return "MEETS", "Relevant biomarker evidence exists in profile"
    return "UNCERTAIN", "Required biomarker information missing"


def _assess_inclusion_performance(
    lowered: str, patient_profile: dict[str, Any]
) -> tuple[str, str] | None:
    if not any(keyword in lowered for keyword in ("ecog", "performance status", "karnofsky")):
        return None
    if patient_profile.get("ecog_performance_status") is not None:
        return "MEETS", "Performance status present in profile"
    return "UNCERTAIN", "Performance status missing"


def _assess_exclusion(
    text: str, patient_profile: dict[str, Any], profile_blob: str
) -> tuple[str, str]:
    lowered = text.lower()

    if "uveal melanoma" in lowered:
        if "uveal" in profile_blob:
            return "FAILS", "Profile indicates uveal melanoma exclusion"
        return "MEETS", "No explicit uveal melanoma evidence"

    if "pregnan" in lowered and str(patient_profile.get("sex", "")).lower() == "male":
        return "MEETS", "Pregnancy exclusion not applicable to male patient"

    if any(k in lowered for k in ["autoimmune", "organ transplant", "active infection"]):
        if any(k in profile_blob for k in ["autoimmune", "transplant", "infection"]):
            return "FAILS", "Potential exclusion condition appears in profile"
        return "UNCERTAIN", _missing_reason_from_criterion(text, exclusion=True)

    return "UNCERTAIN", _missing_reason_from_criterion(text, exclusion=True)


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
    if is_exclusion:
        if verdict == "FAILS":
            exclusion_triggered.append(text)
        elif verdict == "UNCERTAIN":
            exclusion_uncertain.append(text)
            missing.append(reason)
        return
    if verdict == "MEETS":
        inclusion_met.append(text)
    elif verdict == "FAILS":
        inclusion_failed.append(text)
    else:
        inclusion_uncertain.append(text)
        missing.append(reason)


def _evaluate_single_criterion(
    text: str,
    ctype: str,
    patient_profile: dict[str, Any],
    profile_blob: str,
) -> tuple[str, str]:
    if ctype == "exclusion":
        return _assess_exclusion(text, patient_profile, profile_blob)
    return _assess_inclusion(text, patient_profile, profile_blob)


def _select_timeout_tier_and_score(
    exclusion_triggered: list[str], inclusion_failed: list[str], total: int
) -> tuple[str, float]:
    if exclusion_triggered:
        return "disqualified", 0.0
    if inclusion_failed:
        return "weak", 0.3
    return "weak", 0.25 if total == 0 else 0.42


def deterministic_timeout_verdict(
    patient_profile: dict[str, Any],
    all_criteria: list[dict[str, Any]],
    trial_id: str,
) -> JudgeVerdict:
    profile_blob = _profile_blob(patient_profile)

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
        verdict, reason = _evaluate_single_criterion(text, ctype, patient_profile, profile_blob)
        _append_outcome(
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

    tier, score = _select_timeout_tier_and_score(exclusion_triggered, inclusion_failed, total)

    major_criteria_assessable = bool(inclusion_met)

    verdict_dict: dict[str, Any] = {
        "match_score": score,
        "match_tier": tier,
        "major_criteria_assessable": major_criteria_assessable,
        "inclusion_met": inclusion_met,
        "inclusion_failed": inclusion_failed,
        "inclusion_uncertain": inclusion_uncertain,
        "exclusion_triggered": exclusion_triggered,
        "exclusion_uncertain": exclusion_uncertain,
        "critical_missing_info": list(dict.fromkeys(missing))[:10]
        or ["Additional trial-specific data required."],
        "key_concern": ELIGIBILITY_TIMEOUT_FALLBACK_MESSAGE,
        "rationale": f"Eligibility judge timed out for {trial_id}; deterministic rule-based triage applied.",
    }
    return validate_verdict(verdict_dict, trial_id)


def fallback_verdict_for_exception(
    exc: Exception,
    trial_id: str,
    patient_profile: dict[str, Any],
    all_criteria: list[dict[str, Any]],
) -> JudgeVerdict:
    if isinstance(exc, TIMEOUT_EXCEPTIONS):
        logger.warning("Eligibility judge timed out for {} ({})", trial_id, type(exc).__name__)
        return deterministic_timeout_verdict(patient_profile, all_criteria, trial_id)

    logger.opt(exception=False).error(
        "Eligibility judge failed for {} ({}): {}",
        trial_id,
        type(exc).__name__,
        exc,
    )
    return validate_verdict(
        _make_fallback(
            f"Eligibility judge error ({type(exc).__name__})",
            "Eligibility judge execution failed; conservative weak fallback applied.",
            f"Eligibility judge raised {type(exc).__name__}.",
        ),
        trial_id,
    )
