import re
from typing import Any

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


def to_actionable_field(raw_text: str) -> dict[str, str]:
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


def priority_from_impact(affected_count: int) -> str:
    if affected_count >= 4:
        return "high"
    if affected_count >= 2:
        return "medium"
    return "low"


def fallback_missing_info_recommendations(
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
                "priority": priority_from_impact(len(set(ids))),
            }
        )
    return recommendations
