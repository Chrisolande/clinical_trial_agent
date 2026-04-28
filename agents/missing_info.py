"""Missing information identification node."""

import asyncio
import re
from typing import Any

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from loguru import logger
from models.missing_info import CompletenessAssessmentList
from prompts.missinginfo import build_missing_info_human_prompt, build_missing_info_system_prompt

from agents.consent import assert_external_llm_consent
from clinical_trial_agent.config import get_llm, get_settings

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
) -> tuple[dict[str, dict[str, Any]], str]:
    uncertain_by_field: dict[str, dict[str, Any]] = {}
    for trial_id, verdict_data in eligibility_verdicts.items():
        for v in verdict_data.get("verdicts", []):
            if v["verdict"] == "UNCERTAIN":
                text = str(v.get("criterion_text", "")).strip()
                canonical = _to_actionable_field(text)
                field_id = canonical["field_id"]
                if field_id not in uncertain_by_field:
                    uncertain_by_field[field_id] = {
                        **canonical,
                        "affected_trial_ids": [],
                        "examples": [],
                    }
                if trial_id not in uncertain_by_field[field_id]["affected_trial_ids"]:
                    uncertain_by_field[field_id]["affected_trial_ids"].append(trial_id)
                if text and text not in uncertain_by_field[field_id]["examples"]:
                    uncertain_by_field[field_id]["examples"].append(text[:120])
    sorted_items = sorted(
        uncertain_by_field.values(),
        key=lambda item: (-len(item["affected_trial_ids"]), item["field_id"]),
    )
    summary = "\n".join(
        (
            f"- {item['field_id']} ({item['display_name']}) "
            f"[{', '.join(item['affected_trial_ids'][:3])}] "
            f"why_needed={item['why_needed']} "
            f"example={item['examples'][0] if item['examples'] else ''}"
        )
        for item in sorted_items[:20]
    )
    return uncertain_by_field, summary


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
    assert_external_llm_consent()
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
    uncertain_by_field, uncertain_summary = _build_uncertain_summary(eligibility_verdicts)
    if not uncertain_by_field:
        return []

    try:
        result = await _invoke_missing_info_llm(patient_profile, uncertain_summary)
        if result and result.results:
            return _enrich_with_trial_context(
                [item.model_dump() for item in result.results], uncertain_by_field
            )
        logger.info("Missing info model returned no items; using deterministic fallback.")
        return _fallback_missing_info_recommendations(uncertain_by_field)

    except TimeoutError:
        logger.warning("Missing info identification timed out; using deterministic fallback.")
        return _fallback_missing_info_recommendations(uncertain_by_field)
    except asyncio.CancelledError:
        logger.warning("Missing info identification cancelled; using deterministic fallback.")
        return _fallback_missing_info_recommendations(uncertain_by_field)

    except (ValueError, TypeError, RuntimeError) as exc:
        logger.exception("Missing info identification failed: {}", exc)
        return _fallback_missing_info_recommendations(uncertain_by_field)


def _enrich_with_trial_context(
    items: list[dict[str, Any]], uncertain_by_field: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in items:
        field_id = str(item.get("field_id", "")).strip()
        if not field_id:
            source = str(item.get("display_name") or item.get("field") or "").strip().lower()
            field_id = re.sub(r"[^a-z0-9]+", "_", source).strip("_") or "additional_clinical_detail"
        context = uncertain_by_field.get(field_id, {})

        display_name = (
            str(
                item.get("display_name") or item.get("field") or context.get("display_name") or ""
            ).strip()
            or field_id.replace("_", " ").title()
        )
        rationale = str(
            item.get("why_needed")
            or item.get("evidence_text")
            or item.get("description")
            or context.get("why_needed")
            or ""
        ).strip()
        item_ids = [str(tid) for tid in (item.get("affected_trial_ids") or [])]
        context_ids = [str(tid) for tid in (context.get("affected_trial_ids") or [])]
        affected_ids = list(dict.fromkeys(item_ids + context_ids))
        priority = str(item.get("priority", "")).lower()
        if priority not in {"high", "medium", "low"}:
            priority = _priority_from_impact(len(set(affected_ids)))

        enriched.append(
            {
                "field_id": field_id,
                "display_name": display_name,
                "category": str(item.get("category") or context.get("category") or "clinical"),
                "field": display_name,
                "why_needed": rationale,
                "evidence_text": rationale,
                "description": rationale,
                "affected_trial_ids": affected_ids,
                "priority": priority,
            }
        )
    return enriched


_ACTIONABLE_FIELD_MAP: dict[str, dict[str, str]] = {
    "ecog": {
        "field_id": "ecog_performance_status",
        "display_name": "ECOG performance status",
        "category": "clinical_assessment",
        "why_needed": "Document ECOG 0-4 from current oncology assessment to resolve functional eligibility criteria.",
    },
    "performance": {
        "field_id": "ecog_performance_status",
        "display_name": "ECOG performance status",
        "category": "clinical_assessment",
        "why_needed": "Document ECOG 0-4 from current oncology assessment to resolve functional eligibility criteria.",
    },
    "karnofsky": {
        "field_id": "performance_status_scale",
        "display_name": "Performance status (Karnofsky/ECOG)",
        "category": "clinical_assessment",
        "why_needed": "Record the functional status scale used by the oncology team to confirm baseline performance criteria.",
    },
    "creatinine": {
        "field_id": "renal_function_labs",
        "display_name": "Renal function labs",
        "category": "labs",
        "why_needed": "Order serum creatinine and calculate creatinine clearance/eGFR to verify renal eligibility thresholds.",
    },
    "egfr": {
        "field_id": "egfr_mutation_status",
        "display_name": "EGFR mutation status",
        "category": "pathology",
        "why_needed": "Confirm EGFR mutation result from molecular pathology because biomarker-stratified trials require it.",
    },
    "ast": {
        "field_id": "liver_function_labs",
        "display_name": "Liver function labs",
        "category": "labs",
        "why_needed": "Order AST/ALT and bilirubin to verify hepatic eligibility criteria.",
    },
    "alt": {
        "field_id": "liver_function_labs",
        "display_name": "Liver function labs",
        "category": "labs",
        "why_needed": "Order AST/ALT and bilirubin to verify hepatic eligibility criteria.",
    },
    "bilirubin": {
        "field_id": "liver_function_labs",
        "display_name": "Liver function labs",
        "category": "labs",
        "why_needed": "Order AST/ALT and bilirubin to verify hepatic eligibility criteria.",
    },
    "platelet": {
        "field_id": "cbc_hematology",
        "display_name": "Hematology labs (CBC)",
        "category": "labs",
        "why_needed": "Obtain CBC with differential including ANC, hemoglobin, and platelets to resolve hematologic criteria.",
    },
    "hemoglobin": {
        "field_id": "cbc_hematology",
        "display_name": "Hematology labs (CBC)",
        "category": "labs",
        "why_needed": "Obtain CBC with differential including ANC, hemoglobin, and platelets to resolve hematologic criteria.",
    },
    "anc": {
        "field_id": "cbc_hematology",
        "display_name": "Hematology labs (CBC)",
        "category": "labs",
        "why_needed": "Obtain CBC with differential including ANC, hemoglobin, and platelets to resolve hematologic criteria.",
    },
    "ldh": {
        "field_id": "ldh_value",
        "display_name": "LDH value",
        "category": "labs",
        "why_needed": "Obtain serum LDH and compare against protocol threshold.",
    },
    "braf": {
        "field_id": "braf_mutation_status",
        "display_name": "BRAF mutation status",
        "category": "pathology",
        "why_needed": "Confirm BRAF mutation result from pathology because mutation-specific trial arms depend on it.",
    },
    "pd-l1": {
        "field_id": "pd_l1_status",
        "display_name": "PD-L1 status",
        "category": "pathology",
        "why_needed": "Confirm PD-L1 assay result and method to determine biomarker eligibility cutoffs.",
    },
    "recist": {
        "field_id": "recist_measurable_disease",
        "display_name": "Measurable disease assessment (RECIST)",
        "category": "imaging",
        "why_needed": "Document baseline measurable lesions per RECIST v1.1 to confirm measurable-disease inclusion criteria.",
    },
    "measurable lesion": {
        "field_id": "recist_measurable_disease",
        "display_name": "Measurable disease assessment (RECIST)",
        "category": "imaging",
        "why_needed": "Document baseline measurable lesions per RECIST v1.1 to confirm measurable-disease inclusion criteria.",
    },
    "brain mri": {
        "field_id": "brain_mri_assessment",
        "display_name": "Brain MRI assessment",
        "category": "imaging",
        "why_needed": "Provide brain MRI findings to assess CNS metastases and protocol-specific neurologic eligibility.",
    },
    "mediastinal node sampling": {
        "field_id": "mediastinal_node_sampling",
        "display_name": "Mediastinal node sampling",
        "category": "procedural",
        "why_needed": "Document mediastinal node sampling results when nodal staging is required by protocol.",
    },
    "mdt evaluation": {
        "field_id": "mdt_evaluation",
        "display_name": "Multidisciplinary team (MDT) evaluation",
        "category": "clinical_assessment",
        "why_needed": "Capture MDT evaluation status when protocol requires multidisciplinary confirmation before enrollment.",
    },
    "radiation field": {
        "field_id": "prior_radiation_fields",
        "display_name": "Prior radiation fields",
        "category": "treatment_history",
        "why_needed": "Detail prior radiation treatment fields to evaluate overlap/exclusion constraints in protocol.",
    },
    "prior radiation": {
        "field_id": "prior_radiation_fields",
        "display_name": "Prior radiation fields",
        "category": "treatment_history",
        "why_needed": "Detail prior radiation treatment fields to evaluate overlap/exclusion constraints in protocol.",
    },
    "tsh": {
        "field_id": "thyroid_stimulating_hormone",
        "display_name": "TSH level",
        "category": "labs",
        "why_needed": "Obtain thyroid-stimulating hormone (TSH) value when endocrine function criteria are specified.",
    },
    "pft": {
        "field_id": "pulmonary_function_tests",
        "display_name": "Pulmonary function tests (PFTs)",
        "category": "pulmonary",
        "why_needed": "Provide pulmonary function tests to verify respiratory reserve thresholds.",
    },
    "pulmonary function": {
        "field_id": "pulmonary_function_tests",
        "display_name": "Pulmonary function tests (PFTs)",
        "category": "pulmonary",
        "why_needed": "Provide pulmonary function tests to verify respiratory reserve thresholds.",
    },
    "bleeding": {
        "field_id": "recent_bleeding_history",
        "display_name": "Recent bleeding history",
        "category": "clinical_assessment",
        "why_needed": "Document grade ≥3 bleeding/hemorrhage history within protocol time window.",
    },
    "cardiac": {
        "field_id": "cardiac_history_assessment",
        "display_name": "Cardiac history/assessment",
        "category": "clinical_assessment",
        "why_needed": "Document relevant cardiac history and recent ECG/echo findings when required.",
    },
    "retinal": {
        "field_id": "ophthalmologic_history",
        "display_name": "Ophthalmologic history",
        "category": "clinical_assessment",
        "why_needed": "Document retinal/ocular history and exam findings for vision-related exclusions.",
    },
    "histology": {
        "field_id": "pathology_confirmation",
        "display_name": "Pathology confirmation",
        "category": "pathology",
        "why_needed": "Attach pathology report confirming diagnosis and disease subtype.",
    },
    "diagnosis": {
        "field_id": "pathology_confirmation",
        "display_name": "Pathology confirmation",
        "category": "pathology",
        "why_needed": "Attach pathology report confirming diagnosis and disease subtype.",
    },
}


def _to_actionable_field(raw_text: str) -> dict[str, str]:
    lowered = raw_text.lower()
    for key, mapped in _ACTIONABLE_FIELD_MAP.items():
        if key in lowered:
            return dict(mapped)
    cleaned = raw_text.strip().rstrip(".")
    fallback_display = cleaned[:80] if cleaned else "Additional clinical detail"
    fallback_field_id = re.sub(r"[^a-z0-9]+", "_", fallback_display.lower()).strip("_")
    if not fallback_field_id:
        fallback_field_id = "additional_clinical_detail"
    return {
        "field_id": fallback_field_id,
        "display_name": fallback_display,
        "category": "clinical",
        "why_needed": "Capture this missing clinical detail in structured form (note/lab/pathology).",
    }


def _fallback_missing_info_recommendations(
    uncertain_by_field: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not uncertain_by_field:
        return []

    items = sorted(
        uncertain_by_field.values(),
        key=lambda item: (-len(set(item["affected_trial_ids"])), item["field_id"]),
    )[:10]

    recommendations: list[dict[str, Any]] = []
    for item in items:
        ids = list(dict.fromkeys(item.get("affected_trial_ids", [])))
        rationale = str(item.get("why_needed", "")).strip() or (
            "Capture this missing clinical detail in structured form (note/lab/pathology)."
        )
        display_name = str(item.get("display_name", "")).strip() or "Additional clinical detail"
        field_id = str(item.get("field_id", "")).strip() or "additional_clinical_detail"
        recommendations.append(
            {
                "field_id": field_id,
                "display_name": display_name,
                "category": str(item.get("category", "clinical")),
                "field": display_name,
                "why_needed": rationale,
                "evidence_text": rationale,
                "description": rationale,
                "affected_trial_ids": ids,
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
