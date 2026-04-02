from __future__ import annotations

import re
from typing import Any, cast

import openai
from config import get_llm
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from models.judge_verdict import JudgeVerdict
from prompts.eligibility import build_eligibility_prompt
from pydantic import ValidationError
from tools.retry import llm_retry

_FALLBACK_VERDICT: dict[str, Any] = {
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

_TIMEOUT_EXCEPTIONS = (
    TimeoutError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


def _log_warning(msg: str, *args: Any) -> None:
    logger.warning(msg, *args)


def _log_error(msg: str, *args: Any) -> None:
    logger.opt(exception=False).error(msg, *args)


def _format_patient_summary(patient_profile: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in patient_profile.items():
        if value in (None, "", []):
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _format_trial_summary(trial: dict[str, Any], criteria: list[dict[str, Any]]) -> str:
    trial_meta = [
        f"nct_id: {trial.get('nct_id', '')}",
        f"title: {trial.get('brief_title', '')}",
        f"phase: {trial.get('phase', '')}",
        f"status: {trial.get('overall_status', '')}",
    ]
    criteria_lines = [
        f"[{str(c.get('criteria_type', 'inclusion')).lower()}] {c.get('text', '')!s}"
        for c in criteria
    ]
    return "\n".join([*trial_meta, "", "criteria:", *criteria_lines])


def _build_judge_messages(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    criteria: list[dict[str, Any]],
) -> list[BaseMessage]:
    prompt = ChatPromptTemplate.from_template(build_eligibility_prompt())
    return cast(
        "list[BaseMessage]",
        prompt.format_messages(
            patient_summary=_format_patient_summary(patient_profile),
            trial_summary=_format_trial_summary(trial, criteria),
        ),
    )


def _validate_verdict(verdict_dict: dict[str, Any], trial_id: str) -> JudgeVerdict:
    try:
        return cast("JudgeVerdict", JudgeVerdict.model_validate(verdict_dict))
    except (ValidationError, TypeError, ValueError):
        _log_warning("Invalid judge verdict schema for {}", trial_id)
        return cast("JudgeVerdict", JudgeVerdict.model_validate(dict(_FALLBACK_VERDICT)))


def _make_fallback(key_concern: str, rationale: str, missing: str) -> dict[str, Any]:
    fallback = dict(_FALLBACK_VERDICT)
    fallback["key_concern"] = key_concern
    fallback["rationale"] = rationale
    fallback["critical_missing_info"] = [missing]
    return fallback


def _fallback_verdict_for_exception(
    exc: Exception,
    trial_id: str,
    patient_profile: dict[str, Any],
    all_criteria: list[dict[str, Any]],
) -> JudgeVerdict:
    if isinstance(exc, _TIMEOUT_EXCEPTIONS):
        _log_warning("Eligibility judge timed out for {} ({})", trial_id, type(exc).__name__)
        return _deterministic_timeout_verdict(patient_profile, all_criteria, trial_id)

    _log_error(
        "Eligibility judge failed for {} ({}): {}",
        trial_id,
        type(exc).__name__,
        exc,
    )
    return _validate_verdict(
        _make_fallback(
            f"Eligibility judge error ({type(exc).__name__})",
            "Eligibility judge execution failed; conservative weak fallback applied.",
            f"Eligibility judge raised {type(exc).__name__}.",
        ),
        trial_id,
    )


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


def _assess_inclusion(
    text: str, patient_profile: dict[str, Any], profile_blob: str
) -> tuple[str, str]:
    lowered = text.lower()
    age = patient_profile.get("age")

    if "age" in lowered and isinstance(age, (int | float)):
        min_age, max_age = _extract_age_bound(lowered)
        if min_age is not None and age < min_age:
            return "FAILS", "Age below required threshold"
        if max_age is not None and age > max_age:
            return "FAILS", "Age above allowed threshold"
        return "MEETS", "Age appears within stated bounds"

    if "melanoma" in lowered:
        return (
            (
                "MEETS",
                "Diagnosis mentions melanoma and patient condition includes melanoma",
            )
            if "melanoma" in profile_blob
            else ("UNCERTAIN", "Melanoma diagnosis not explicit in profile")
        )

    if any(k in lowered for k in ["braf", "pd-l1", "biomarker", "mutation"]):
        if any(k in profile_blob for k in ["braf", "pd-l1", "mutation"]):
            return "MEETS", "Relevant biomarker evidence exists in profile"
        return "UNCERTAIN", "Required biomarker information missing"

    if any(k in lowered for k in ["ecog", "performance status", "karnofsky"]):
        if patient_profile.get("ecog_performance_status") is not None:
            return "MEETS", "Performance status present in profile"
        return "UNCERTAIN", "Performance status missing"

    return "UNCERTAIN", _missing_reason_from_criterion(text, exclusion=False)


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


def _missing_reason_from_criterion(text: str, *, exclusion: bool = False) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ["ecog", "karnofsky", "performance status"]):
        return "Missing ECOG/performance status documentation"
    if any(k in lowered for k in ["creatinine", "egfr", "renal"]):
        return "Missing renal function labs (creatinine/eGFR)"
    if any(k in lowered for k in ["ast", "alt", "bilirubin", "liver"]):
        return "Missing liver function labs (AST/ALT/bilirubin)"
    if any(k in lowered for k in ["anc", "hemoglobin", "platelet", "cbc", "hematolog"]):
        return "Missing CBC/hematology labs (ANC/Hb/platelets)"
    if any(k in lowered for k in ["ldh"]):
        return "Missing LDH value"
    if any(k in lowered for k in ["braf", "pd-l1", "biomarker", "mutation"]):
        return "Missing molecular biomarker result"
    if any(k in lowered for k in ["recist", "measurable lesion", "imaging", "measurable disease"]):
        return "Missing baseline imaging/measurable disease assessment"
    if any(k in lowered for k in ["histology", "pathology", "diagnosis"]):
        return "Missing pathology/diagnosis confirmation"
    if any(k in lowered for k in ["bleeding", "hemorrhage"]):
        return "Missing recent bleeding history"
    if any(k in lowered for k in ["cardiac", "ecg", "echo"]):
        return "Missing cardiac history/assessment"
    if any(k in lowered for k in ["retinal", "ophthalm"]):
        return "Missing ophthalmologic/retinal assessment"
    return (
        "Missing trial-specific clinical detail"
        if not exclusion
        else "Missing exclusion-history detail"
    )


def _deterministic_timeout_verdict(
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

        if ctype == "exclusion":
            verdict, reason = _assess_exclusion(text, patient_profile, profile_blob)
            if verdict == "FAILS":
                exclusion_triggered.append(text)
            elif verdict == "UNCERTAIN":
                exclusion_uncertain.append(text)
                missing.append(reason)
        else:
            verdict, reason = _assess_inclusion(text, patient_profile, profile_blob)
            if verdict == "MEETS":
                inclusion_met.append(text)
            elif verdict == "FAILS":
                inclusion_failed.append(text)
            else:
                inclusion_uncertain.append(text)
                missing.append(reason)

    total = (
        len(inclusion_met)
        + len(inclusion_failed)
        + len(inclusion_uncertain)
        + len(exclusion_triggered)
        + len(exclusion_uncertain)
    )
    uncertain = len(inclusion_uncertain) + len(exclusion_uncertain)

    if exclusion_triggered:
        tier = "disqualified"
        score = 0.0
    elif inclusion_failed:
        tier = "weak"
        score = 0.3
    elif total > 0 and uncertain <= max(2, total // 3):
        tier = "moderate"
        score = 0.62
    elif total > 0 and uncertain < total:
        tier = "weak"
        score = 0.42
    else:
        tier = "weak"
        score = 0.25

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
        "key_concern": "Deterministic fallback used after eligibility judge timeout",
        "rationale": f"Eligibility judge timed out for {trial_id}; deterministic rule-based triage applied.",
    }
    return _validate_verdict(verdict_dict, trial_id)


@llm_retry
async def _judge_trial(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    criteria: list[dict[str, Any]],
) -> JudgeVerdict:
    llm = get_llm()
    if hasattr(llm, "with_structured_output"):
        llm = llm.with_structured_output(JudgeVerdict)

    messages = _build_judge_messages(patient_profile, trial, criteria)
    response = await llm.ainvoke(
        messages,
        config={"run_name": "eligibility_judge", "tags": ["eligibility", "judge"]},
    )
    trial_id = str(trial.get("nct_id", "unknown"))

    if isinstance(response, JudgeVerdict):
        return response
    if isinstance(response, dict):
        return _validate_verdict(cast("dict[str, Any]", response), trial_id)

    content = getattr(response, "content", None)
    if isinstance(content, dict):
        return _validate_verdict(cast("dict[str, Any]", content), trial_id)

    _log_warning(
        "Eligibility judge returned non-structured response for {}: {}",
        trial_id,
        type(response).__name__,
    )
    return _validate_verdict(dict(_FALLBACK_VERDICT), trial_id)


def _build_verdict_rows(verdict: JudgeVerdict) -> list[dict[str, Any]]:
    def rows(texts: list[str], verdict_label: str, kind: str, hard: bool) -> list[dict[str, Any]]:
        return [
            {
                "criterion_text": t,
                "verdict": verdict_label,
                "criterion_type": kind,
                "is_hard_exclusion": hard,
            }
            for t in texts
        ]

    return [
        *rows(verdict.inclusion_met, "MEETS", "inclusion", False),
        *rows(verdict.inclusion_failed, "FAILS", "inclusion", False),
        *rows(verdict.inclusion_uncertain, "UNCERTAIN", "inclusion", False),
        *rows(verdict.exclusion_triggered, "FAILS", "exclusion", True),
        *rows(verdict.exclusion_uncertain, "UNCERTAIN", "exclusion", True),
    ]


def _is_meets(verdict_row: dict[str, Any]) -> bool:
    return verdict_row.get("verdict") == "MEETS"


def _is_soft_fail(verdict_row: dict[str, Any]) -> bool:
    return verdict_row.get("verdict") == "FAILS" and not bool(verdict_row.get("is_hard_exclusion"))


def _is_hard_fail(verdict_row: dict[str, Any]) -> bool:
    return verdict_row.get("verdict") == "FAILS" and bool(verdict_row.get("is_hard_exclusion"))


def _is_uncertain(verdict_row: dict[str, Any]) -> bool:
    return verdict_row.get("verdict") == "UNCERTAIN"


def _summarize_verdict_counts(
    verdicts: list[dict[str, Any]],
) -> tuple[int, int, int, int]:
    inclusion_meets = [v for v in verdicts if _is_meets(v)]
    inclusion_fails = [v for v in verdicts if _is_soft_fail(v)]
    exclusion_triggered = [v for v in verdicts if _is_hard_fail(v)]
    uncertain = [v for v in verdicts if _is_uncertain(v)]
    return (
        len(inclusion_meets),
        len(inclusion_fails) + len(exclusion_triggered),
        len(uncertain),
        len(exclusion_triggered),
    )


def _build_batch_result(
    trial_id: str,
    verdict: JudgeVerdict,
    verdicts: list[dict[str, Any]],
) -> dict[str, Any]:
    meets_count, fails_count, uncertain_count, hard_exclusion_failures = _summarize_verdict_counts(
        verdicts
    )
    return {
        "trial_id": trial_id,
        "match_score": verdict.match_score,
        "match_tier": verdict.match_tier,
        "major_criteria_assessable": verdict.major_criteria_assessable,
        "critical_missing_info": verdict.critical_missing_info,
        "key_concern": verdict.key_concern,
        "rationale": verdict.rationale,
        "verdicts": verdicts,
        "meets_count": meets_count,
        "fails_count": fails_count,
        "uncertain_count": uncertain_count,
        "hard_exclusion_failures": hard_exclusion_failures,
    }


async def evaluate_criteria_batch(
    patient_profile: dict[str, Any],
    trial: dict[str, Any],
    all_criteria: list[dict[str, Any]],
) -> dict[str, Any]:
    trial_id = str(trial.get("nct_id", "unknown"))

    try:
        verdict = await _judge_trial(patient_profile, trial, all_criteria)
    except Exception as exc:
        verdict = _fallback_verdict_for_exception(exc, trial_id, patient_profile, all_criteria)

    verdicts = _build_verdict_rows(verdict)
    return _build_batch_result(trial_id, verdict, verdicts)
