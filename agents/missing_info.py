"""Missing information identification node."""

import asyncio
import os
from typing import Any

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from loguru import logger
from models.missing_info import CompletenessAssessmentList
from prompts.missinginfo import build_missing_info_human_prompt, build_missing_info_system_prompt

from config import external_llm_requires_consent, get_llm, get_settings


def _assert_external_llm_consent() -> None:
    if not external_llm_requires_consent():
        return
    consent = os.environ.get("CLINICAL_DATA_EXTERNAL_LLM_CONSENT", "false").strip().lower()
    if consent != "true":
        raise RuntimeError(
            "CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true is required before sending patient data to external LLMs."
        )


_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(build_missing_info_system_prompt()),
        HumanMessagePromptTemplate.from_template(build_missing_info_human_prompt()),
    ]
)


def _get_chain() -> Any:
    """Build a fresh chain per call.

    Caching this chain globally can bind transports to a stale asyncio loop under
    uvicorn/pytest startup behavior.
    """
    return _PROMPT | get_llm().with_structured_output(CompletenessAssessmentList)


def _build_uncertain_summary(
    eligibility_verdicts: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[str]], str]:
    uncertain_by_theme: dict[str, list[str]] = {}
    for trial_id, verdict_data in eligibility_verdicts.items():
        for v in verdict_data.get("verdicts", []):
            if v["verdict"] == "UNCERTAIN":
                text = v["criterion_text"][:100]
                uncertain_by_theme.setdefault(text, []).append(trial_id)
    summary = "\n".join(
        f"- [{', '.join(ids[:3])}] {text}" for text, ids in list(uncertain_by_theme.items())[:20]
    )
    return uncertain_by_theme, summary


def _format_profile_summary(profile: dict[str, Any]) -> str:
    redacted_fields = {
        "age",
        "sex",
        "conditions",
        "primary_condition",
        "biomarkers",
        "medications",
        "prior_treatments",
    }
    available: list[str] = []
    for key, value in profile.items():
        if value in (None, "", []):
            continue
        if key in redacted_fields:
            available.append(f"{key}: [redacted]")
        else:
            available.append(f"{key}: {str(value)[:60]}")
    return "\n".join(available[:20])


async def _invoke_missing_info_llm(
    patient_profile: dict[str, Any],
    uncertain_summary: str,
) -> CompletenessAssessmentList:
    """Invoke the missing-info chain. Timeout is delegated to RunnableConfig."""
    _assert_external_llm_consent()
    result = await _get_chain().ainvoke(
        {
            "patient_profile": _format_profile_summary(patient_profile),
            "trial_verdicts": uncertain_summary,
        },
        config={
            "run_name": "missing_info",
            "tags": ["eligibility", "missing-data"],
            "timeout": get_settings().llm_call_timeout_seconds,
        },
    )
    if isinstance(result, CompletenessAssessmentList):
        return result
    if isinstance(result, dict):
        return CompletenessAssessmentList.model_validate(result)
    raise TypeError(f"Unexpected missing-info output type: {type(result)!r}")


async def identify_missing_info(
    patient_profile: dict[str, Any],
    eligibility_verdicts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    uncertain_by_theme, uncertain_summary = _build_uncertain_summary(eligibility_verdicts)
    if not uncertain_by_theme:
        return []

    try:
        result = await _invoke_missing_info_llm(patient_profile, uncertain_summary)
        if result and result.results:
            return [item.model_dump() for item in result.results]
        logger.info("Missing info model returned no items; using deterministic fallback.")
        return _fallback_missing_info_recommendations(uncertain_by_theme)

    except TimeoutError:
        logger.warning("Missing info identification timed out; using deterministic fallback.")
        return _fallback_missing_info_recommendations(uncertain_by_theme)
    except asyncio.CancelledError:
        logger.warning("Missing info identification cancelled; using deterministic fallback.")
        return _fallback_missing_info_recommendations(uncertain_by_theme)

    except (ValueError, TypeError, RuntimeError) as exc:
        logger.exception("Missing info identification failed: {}", exc)
        return _fallback_missing_info_recommendations(uncertain_by_theme)


_ACTIONABLE_FIELD_MAP: dict[str, tuple[str, str]] = {
    "ecog": (
        "ECOG performance status",
        "Document ECOG 0-4 from current oncology assessment.",
    ),
    "performance": (
        "ECOG performance status",
        "Document ECOG 0-4 from current oncology assessment.",
    ),
    "karnofsky": (
        "Performance status (Karnofsky/ECOG)",
        "Record functional status scale used by the treating oncology team.",
    ),
    "creatinine": (
        "Renal function labs",
        "Order serum creatinine and calculate creatinine clearance/eGFR.",
    ),
    "egfr": (
        "EGFR mutation status required",
        "Confirm EGFR mutation result from pathology/molecular report.",
    ),
    "ast": (
        "Liver function labs",
        "Order AST/ALT and total bilirubin to verify hepatic eligibility criteria.",
    ),
    "alt": (
        "Liver function labs",
        "Order AST/ALT and total bilirubin to verify hepatic eligibility criteria.",
    ),
    "bilirubin": (
        "Liver function labs",
        "Order AST/ALT and total bilirubin to verify hepatic eligibility criteria.",
    ),
    "platelet": (
        "Hematology labs (CBC)",
        "Obtain CBC with differential including ANC, hemoglobin, and platelet count.",
    ),
    "hemoglobin": (
        "Hematology labs (CBC)",
        "Obtain CBC with differential including ANC, hemoglobin, and platelet count.",
    ),
    "anc": (
        "Hematology labs (CBC)",
        "Obtain CBC with differential including ANC, hemoglobin, and platelet count.",
    ),
    "ldh": (
        "LDH value",
        "Obtain serum LDH and compare against protocol threshold.",
    ),
    "braf": (
        "BRAF mutation status",
        "Confirm BRAF mutation result from pathology/molecular report.",
    ),
    "pd-l1": (
        "PD-L1 status",
        "Confirm PD-L1 assay result and method from pathology report.",
    ),
    "recist": (
        "Measurable disease assessment (RECIST)",
        "Document baseline measurable lesions per RECIST v1.1 on imaging.",
    ),
    "measurable lesion": (
        "Measurable disease assessment (RECIST)",
        "Document baseline measurable lesions per RECIST v1.1 on imaging.",
    ),
    "bleeding": (
        "Recent bleeding history",
        "Document Grade >=3 bleeding/hemorrhage history within protocol time window.",
    ),
    "cardiac": (
        "Cardiac history/assessment",
        "Document relevant cardiac history and recent ECG/echo if protocol requires.",
    ),
    "retinal": (
        "Ophthalmologic history",
        "Document retinal/ocular history and exam findings if required by protocol.",
    ),
    "histology": (
        "Pathology confirmation",
        "Attach pathology report confirming diagnosis and disease subtype.",
    ),
    "diagnosis": (
        "Pathology confirmation",
        "Attach pathology report confirming diagnosis and disease subtype.",
    ),
}


def _to_actionable_field(raw_text: str) -> tuple[str, str]:
    lowered = raw_text.lower()
    for key, mapped in _ACTIONABLE_FIELD_MAP.items():
        if key in lowered:
            return mapped
    cleaned = raw_text.strip().rstrip(".")
    fallback_field = cleaned[:80] if cleaned else "Additional clinical detail"
    return (
        fallback_field,
        "Capture this missing clinical detail in structured form (note/lab/pathology).",
    )


def _fallback_missing_info_recommendations(
    uncertain_by_theme: dict[str, list[str]],
) -> list[dict[str, Any]]:
    if not uncertain_by_theme:
        return []

    items = sorted(
        uncertain_by_theme.items(),
        key=lambda kv: len(set(kv[1])),
        reverse=True,
    )[:10]

    recommendations: list[dict[str, Any]] = []
    for text, ids in items:
        field, description = _to_actionable_field(text)
        recommendations.append(
            {
                "field": field,
                "description": description,
                "affected_trial_ids": list(dict.fromkeys(ids)),
                "priority": _priority_from_impact(len(set(ids))),
            }
        )
    return recommendations


def _priority_from_impact(affected_count: int) -> str:
    if affected_count >= 4:
        return "high"
    if affected_count >= 2:
        return "medium"
    return "low"
