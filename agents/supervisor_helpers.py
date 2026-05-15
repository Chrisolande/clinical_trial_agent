import json
from typing import Any

from clinical_trial_agent.config import TIER_ORDER, get_settings

PatientSummary = dict[str, Any]
TrialSummary = dict[str, Any]
ScoredTrialSummary = dict[str, Any]


def apply_feedback_adjustments(
    scored_trials: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    boosts: dict[str, int] = {}
    for row in feedback_rows:
        nct = str(row.get("nct_id", "")).strip()
        verdict = str(row.get("verdict", "")).strip().lower()
        if not nct:
            continue
        if verdict == "confirmed":
            boosts[nct] = boosts.get(nct, 0) + 1
        elif verdict == "rejected":
            boosts[nct] = boosts.get(nct, 0) - 1

    adjusted = [dict(trial) for trial in scored_trials]
    for trial in adjusted:
        trial_id = str(trial.get("trial_id", "")).strip()
        delta = boosts.get(trial_id, 0)
        if delta == 0:
            continue
        trial["score"] = max(0.0, min(1.0, float(trial.get("score", 0.0)) + 0.05 * delta))

    adjusted.sort(
        key=lambda x: (
            TIER_ORDER.get(str(x.get("tier", "weak")), 0),
            float(x.get("score", 0.0)),
        ),
        reverse=True,
    )
    for idx, trial in enumerate(adjusted, 1):
        trial["rank"] = idx
    return adjusted


def compute_tier_counts(scored_trials: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"strong": 0, "moderate": 0, "weak": 0, "disqualified": 0}
    for trial in scored_trials:
        tier = str(trial.get("tier", "weak"))
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def unwrap_synthesis_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"report_text": str(result)}

    if isinstance(result.get("report_json"), dict):
        return result

    report_text = result.get("report_text", "")
    if isinstance(report_text, str) and report_text.strip().startswith("{"):
        try:
            parsed = json.loads(report_text)
            if isinstance(parsed, dict) and "report_json" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return result


def unwrap_report_json(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        nested = parsed.get("report_json")
        if isinstance(nested, dict):
            merged = dict(parsed)
            merged["report_json"] = nested
            return merged
        if "report_text" in parsed:
            return parsed
    return None


def extract_final_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        if isinstance(result.get("report_json"), dict):
            return result
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str):
                unwrapped = unwrap_report_json(content)
                if isinstance(unwrapped, dict):
                    return unwrapped
                return {"report_text": content}
    return {"report_text": str(result)}


def project_patient_summary(patient_profile: dict[str, Any]) -> PatientSummary:
    keys = (
        "age",
        "sex",
        "primary_condition",
        "conditions",
        "medical_history",
        "medications",
        "biomarkers",
        "lab_values",
        "prior_treatments",
        "contraindications",
        "ecog_performance_status",
        "smoking_status",
        "bmi",
    )
    return {k: patient_profile[k] for k in keys if patient_profile.get(k)}


def project_trial_summary(trial: dict[str, Any]) -> TrialSummary:
    settings = get_settings()
    criteria_text = str(trial.get("eligibility_criteria_raw", ""))[: settings.criteria_text_max_chars]
    return {
        "nct_id": str(trial.get("nct_id", "")),
        "brief_title": str(trial.get("brief_title", "")),
        "overall_status": str(trial.get("overall_status", "")),
        "phase": str(trial.get("phase", "")),
        "conditions": list(trial.get("conditions", [])),
        "interventions": list(trial.get("interventions", [])),
        "eligibility_criteria_raw": criteria_text,
        "locations": list(trial.get("locations", []))[:5],
        "primary_completion_date": str(trial.get("primary_completion_date", "")),
        "criteria_source": str(trial.get("criteria_source", "missing")),
        "criteria_source_verified": bool(trial.get("criteria_source_verified", False)),
        "criteria_retrieved_at": str(trial.get("criteria_retrieved_at", "")),
        "criteria_completeness": str(trial.get("criteria_completeness", "missing")),
    }


def compress_retrieval_output(retrieval: dict[str, Any]) -> dict[str, Any]:
    deduped = [
        project_trial_summary(t)
        for t in list(retrieval.get("trials_deduplicated", []))
        if isinstance(t, dict)
    ]
    raw = [
        project_trial_summary(t)
        for t in list(retrieval.get("trials_raw", []))
        if isinstance(t, dict)
    ]
    return {
        "trials_deduplicated": deduped,
        "trials_raw": raw,
        "search_queries": list(retrieval.get("search_queries", [])),
        "retrieval_failed": bool(retrieval.get("retrieval_failed", False)),
        "retrieval_errors": list(retrieval.get("retrieval_errors", retrieval.get("errors", []))),
    }


def compress_scored_trials(trials: list[dict[str, Any]]) -> list[ScoredTrialSummary]:
    keep = (
        "trial_id",
        "brief_title",
        "overall_status",
        "phase",
        "score",
        "tier",
        "major_criteria_assessable",
        "key_concern",
        "critical_missing_info",
        "rationale",
        "meets_count",
        "fails_count",
        "uncertain_count",
        "hard_exclusion_failures",
        "key_inclusion_passed",
        "key_exclusion_failed",
        "key_uncertain",
        "locations_summary",
        "criteria_source",
        "criteria_source_verified",
        "criteria_retrieved_at",
        "criteria_completeness",
        "internal_fallback_used",
        "sparse_evidence_cap_applied",
        "sparse_evidence_cap_reason",
    )
    return [{key: trial[key] for key in keep if trial.get(key) is not None} for trial in trials]


def compress_decision_history(decisions: list[str]) -> list[str]:
    return [d[:240] for d in decisions[-20:]]
