import re
from datetime import UTC, datetime
from typing import Any

from clinical_trial_agent.config import TIER_ORDER

from .report_formatter import build_text_report
from .report_synthesizer import generate_report_plan

_METHODOLOGY_NOTE = (
    "This report uses structured eligibility assessment with conservative tiering. "
    "Strong matches require confidence on major criteria and no disqualifying exclusion triggers. "
    "Weak/disqualified trials are excluded from the main body."
)

_NOT_APPLICABLE_MARKERS = ("not_applicable", "not applicable", "n/a", "non-applicable")
_INTERNAL_FAILURE_MARKERS = (
    "judge model",
    "structured verdict",
    "llm",
    "parser",
    "fallback",
    "tool failed",
    "qa issue",
    "qa check",
    "model returned none",
)
_LOW_VALUE_GAP_MARKERS = {
    "criterion requires details not present in profile",
    "missing trial-specific clinical detail",
    "missing exclusion-history detail",
    "additional clinical detail",
    "n/a",
    "unknown",
    "none",
}
_PUBLIC_LOW_PRIORITY_GAP_LIMIT = 2


def _normalize_qa_issues(qa_issues: list[dict[str, Any] | str]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for issue in qa_issues:
        if isinstance(issue, dict):
            normalized.append(
                {
                    "code": str(issue.get("code", "UNSPECIFIED")),
                    "severity": str(issue.get("severity", "medium")),
                    "message": str(issue.get("message", "")),
                }
            )
            continue
        message = str(issue).strip()
        if message:
            normalized.append(
                {
                    "code": "UNSPECIFIED",
                    "severity": "medium",
                    "message": message,
                }
            )
    return normalized


def _sanitize_qa_remediation(qa_remediation: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(qa_remediation, dict):
        return {"attempts": 0, "actions": [], "unresolved_issues": []}
    return {
        "attempts": int(qa_remediation.get("attempts", 0) or 0),
        "actions": list(qa_remediation.get("actions") or []),
        "unresolved_issues": _normalize_qa_issues(list(qa_remediation.get("unresolved_issues") or [])),
    }


def _severity_for_missing_item(item: str, tier: str) -> str:
    text = item.lower()
    high_markers = [
        "egfr",
        "alk",
        "ros1",
        "her2",
        "pd-l1",
        "stage",
        "ecog",
        "karnofsky",
        "diagnosis",
        "prior treatment",
        "measurable disease",
    ]
    if any(marker in text for marker in high_markers):
        return "high"
    if tier in {"weak", "disqualified"}:
        return "medium"
    return "low"


def _collect_information_gaps(scored_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for trial in scored_trials:
        for gap in _gaps_from_scored_trial(trial):
            _merge_gap(grouped, gap)

    priority_order = {"high": 3, "medium": 2, "low": 1}
    return sorted(
        grouped.values(), key=lambda x: priority_order.get(str(x["priority"]), 0), reverse=True
    )


def _gaps_from_scored_trial(trial: dict[str, Any]) -> list[dict[str, Any]]:
    trial_id = str(trial.get("trial_id", ""))
    tier = str(trial.get("tier", "weak"))
    gaps: list[dict[str, Any]] = []
    for item in trial.get("critical_missing_info", []) or []:
        raw_item = str(item).strip()
        if not raw_item or _is_not_applicable_gap(
            {"description": raw_item, "field": raw_item, "category": ""}
        ):
            continue
        normalized = _normalize_gap_item(raw_item)
        gaps.append(_gap_payload(normalized, raw_item, tier, trial_id))
    return gaps


def _gap_payload(
    normalized: dict[str, str], raw_item: str, tier: str, trial_id: str
) -> dict[str, Any]:
    return {
        "field_id": normalized["field_id"],
        "display_name": normalized["display_name"],
        "category": normalized["category"],
        "field": normalized["display_name"],
        "why_needed": normalized["why_needed"],
        "evidence_text": normalized["why_needed"],
        "description": normalized["why_needed"],
        "priority": _severity_for_missing_item(raw_item, tier),
        "affected_trial_ids": [trial_id] if trial_id else [],
    }


def _merge_gap(grouped: dict[str, dict[str, Any]], gap: dict[str, Any]) -> None:
    key = str(gap["field_id"])
    if key not in grouped:
        grouped[key] = gap
        return
    current = grouped[key]
    current["affected_trial_ids"] = sorted(
        set(current.get("affected_trial_ids", [])) | set(gap.get("affected_trial_ids", []))
    )
    if _priority_rank(str(gap.get("priority", "low"))) > _priority_rank(
        str(current.get("priority", "low"))
    ):
        current["priority"] = gap["priority"]


def _deduplicate_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        if _is_not_applicable_gap(gap):
            continue
        normalized = _normalized_gap(gap)
        if not normalized:
            continue
        field_id = str(normalized["field_id"])
        if field_id not in seen:
            seen[field_id] = normalized
            continue
        _merge_normalized_gap(seen[field_id], normalized)
    return list(seen.values())


def _contains_internal_failure_text(*parts: str) -> bool:
    text = " ".join(part for part in parts if part).strip().lower()
    return bool(text) and any(marker in text for marker in _INTERNAL_FAILURE_MARKERS)


def _is_low_value_gap(gap: dict[str, Any]) -> bool:
    field = str(gap.get("field") or gap.get("display_name") or "").strip().lower()
    desc = str(gap.get("description") or gap.get("why_needed") or "").strip().lower()
    if not field and not desc:
        return True
    if field in _LOW_VALUE_GAP_MARKERS or desc in _LOW_VALUE_GAP_MARKERS:
        return True
    return bool(field and field == desc and field in _LOW_VALUE_GAP_MARKERS)


def _limit_public_low_priority_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not gaps:
        return []
    output: list[dict[str, Any]] = []
    low_seen = 0
    for gap in gaps:
        if str(gap.get("priority", "medium")).lower() == "low":
            if low_seen >= _PUBLIC_LOW_PRIORITY_GAP_LIMIT:
                continue
            low_seen += 1
        output.append(gap)
    return output


def _sanitize_information_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for gap in _deduplicate_gaps(gaps):
        if _is_not_applicable_gap(gap):
            continue
        normalized = _normalized_gap(gap)
        if not normalized:
            continue
        if _contains_internal_failure_text(
            str(normalized.get("field", "")),
            str(normalized.get("display_name", "")),
            str(normalized.get("description", "")),
            str(normalized.get("why_needed", "")),
        ):
            continue
        if _is_low_value_gap(normalized):
            continue
        sanitized.append(normalized)

    ordered = sorted(
        sanitized,
        key=lambda x: (
            _priority_rank(str(x.get("priority", "low"))),
            len(str(x.get("description", ""))),
        ),
        reverse=True,
    )
    return _limit_public_low_priority_gaps(ordered)


def _normalized_gap(gap: dict[str, Any]) -> dict[str, Any] | None:
    field_id = str(gap.get("field_id", "")).strip() or _fallback_gap_field_id(gap)
    if not field_id:
        return None
    normalized = dict(gap)
    display_name = str(gap.get("display_name") or gap.get("field") or "").strip()
    normalized["field_id"] = field_id
    normalized["display_name"] = display_name or field_id.replace("_", " ").title()
    normalized["field"] = normalized["display_name"]
    rationale = _gap_rationale(normalized)
    normalized["why_needed"] = rationale
    normalized["evidence_text"] = rationale
    normalized["description"] = rationale
    normalized["affected_trial_ids"] = sorted(set(normalized.get("affected_trial_ids") or []))
    return normalized


def _gap_rationale(gap: dict[str, Any]) -> str:
    return str(gap.get("why_needed") or gap.get("evidence_text") or gap.get("description") or "")


def _merge_normalized_gap(current: dict[str, Any], normalized: dict[str, Any]) -> None:
    if len(str(normalized.get("description", ""))) > len(str(current.get("description", ""))):
        rationale = str(normalized.get("description", ""))
        current["why_needed"] = rationale
        current["evidence_text"] = rationale
        current["description"] = rationale
    if _priority_rank(str(normalized.get("priority", "low"))) > _priority_rank(
        str(current.get("priority", "low"))
    ):
        current["priority"] = str(normalized.get("priority", "low")).lower()
    current["affected_trial_ids"] = sorted(
        set(current.get("affected_trial_ids", [])) | set(normalized.get("affected_trial_ids", []))
    )


def _priority_rank(priority: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(priority.lower(), 0)


def _is_not_applicable_gap(gap: dict[str, Any]) -> bool:
    text = (
        " ".join(
            [
                str(gap.get("field_id", "")),
                str(gap.get("field", "")),
                str(gap.get("display_name", "")),
                str(gap.get("description", "")),
                str(gap.get("why_needed", "")),
                str(gap.get("category", "")),
            ]
        )
        .strip()
        .lower()
    )
    if not text:
        return False
    return any(marker in text for marker in _NOT_APPLICABLE_MARKERS)


def _fallback_gap_field_id(gap: dict[str, Any]) -> str:
    text = str(gap.get("field") or gap.get("display_name") or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slug or "additional_clinical_detail"


def _normalize_gap_item(raw_item: str) -> dict[str, str]:
    slug = re.sub(r"[^a-z0-9]+", "_", raw_item.lower()).strip("_")
    display_name = raw_item
    return {
        "field_id": slug or "additional_clinical_detail",
        "display_name": display_name,
        "category": "clinical",
        "why_needed": raw_item,
    }


def _merge_information_gaps(
    scored_trials: list[dict[str, Any]], missing_info: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = _collect_information_gaps(scored_trials)
    if missing_info:
        merged.extend(missing_info)
    return _sanitize_information_gaps(merged)


def _partition_trials(
    enriched_trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strong = [t for t in enriched_trials if str(t.get("tier", "weak")) == "strong"]
    moderate = [t for t in enriched_trials if str(t.get("tier", "weak")) == "moderate"]
    weak = [t for t in enriched_trials if str(t.get("tier", "weak")) == "weak"]
    disqualified = [t for t in enriched_trials if str(t.get("tier", "weak")) == "disqualified"]
    return strong, moderate, weak, disqualified


def _build_per_criterion_verdicts(verdict_details: dict[str, Any]) -> list[dict[str, str]]:
    mapping = {"MEETS": "eligible", "FAILS": "ineligible", "UNCERTAIN": "uncertain"}
    rows: list[dict[str, str]] = []
    for item in list(verdict_details.get("verdicts", [])):
        text = str(item.get("criterion_text", "")).strip()
        if not text:
            continue
        verdict = mapping.get(str(item.get("verdict", "UNCERTAIN")), "uncertain")
        rows.append(
            {
                "criterion_text": text,
                "verdict": verdict,
                "reasoning": str(item.get("reasoning", "")),
            }
        )
    return rows


def _build_trial_report_entry(
    trial: dict[str, Any], eligibility_verdicts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    tid = str(trial.get("trial_id", ""))
    enriched = dict(trial)
    details = eligibility_verdicts.get(tid, {})
    enriched["verdict_details"] = details
    enriched["per_criterion_verdicts"] = _build_per_criterion_verdicts(details)
    return enriched


def _sort_by_tier_then_score(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        trials,
        key=lambda t: (TIER_ORDER.get(str(t.get("tier", "weak")), 0), float(t.get("score", 0.0))),
        reverse=True,
    )


def _build_report_payload(
    *,
    report_plan: Any,
    enriched_trials: list[dict[str, Any]],
    information_gaps: list[dict[str, Any]],
    trials_raw: list[dict[str, Any]],
    search_queries: list[str],
    decision_history: list[str],
    qa_issues_internal: list[dict[str, str]],
    retrieval_errors: list[str],
    qa_remediation_internal: dict[str, Any],
) -> dict[str, Any]:
    strong, moderate, weak, disqualified = _partition_trials(enriched_trials)
    retrieval_failed = bool(retrieval_errors) and len(enriched_trials) == 0
    report_plan_dict = report_plan.model_dump() if hasattr(report_plan, "model_dump") else dict(report_plan)
    effective_summary = str(report_plan_dict.get("executive_summary", ""))
    patient_summary = str(report_plan_dict.get("patient_summary", ""))
    if retrieval_failed:
        effective_summary = (
            "Trial retrieval failed before eligibility fan-out completed. "
            "No matches are reported because zero trials were retrieved/evaluated."
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_plan": report_plan_dict,
        "patient_summary": patient_summary,
        "executive_summary": effective_summary,
        "total_trials_searched": len(trials_raw),
        "total_trials_evaluated": len(enriched_trials),
        "retrieval_failed": retrieval_failed,
        "retrieval_errors": list(retrieval_errors),
        "strong_matches": strong,
        "moderate_matches": moderate,
        "excluded_trial_count": len(weak) + len(disqualified),
        "excluded_trials": weak + disqualified,
        "information_gaps": information_gaps,
        "qa_issues_internal": qa_issues_internal,
        "qa_remediation_internal": qa_remediation_internal,
        "debug": False,
        "methodology_note": _METHODOLOGY_NOTE,
        "search_queries_used": list(dict.fromkeys(search_queries)),
        "decision_history": decision_history,
        "ranked_trials": enriched_trials,
    }


async def build_report(
    patient_profile: dict[str, Any],
    scored_trials: list[dict[str, Any]],
    missing_info: list[dict[str, Any]],
    eligibility_verdicts: dict[str, dict[str, Any]],
    trials_raw: list[dict[str, Any]],
    search_queries: list[str],
    decision_history: list[str],
    qa_issues: list[dict[str, Any] | str],
    retrieval_errors: list[str] | None = None,
    qa_remediation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranked = _sort_by_tier_then_score(scored_trials)
    enriched_trials = [_build_trial_report_entry(t, eligibility_verdicts) for t in ranked]
    merged_gaps = _merge_information_gaps(ranked, missing_info)
    qa_internal = _normalize_qa_issues(qa_issues)
    report_plan = await generate_report_plan(
        patient_profile=patient_profile,
        scored_trials=enriched_trials,
        eligibility_verdicts=eligibility_verdicts,
        missing_info=merged_gaps,
        qa_issues=qa_internal,
    )
    return _build_report_payload(
        report_plan=report_plan,
        enriched_trials=enriched_trials,
        information_gaps=merged_gaps,
        trials_raw=trials_raw,
        search_queries=search_queries,
        decision_history=decision_history,
        qa_issues_internal=qa_internal,
        retrieval_errors=list(retrieval_errors or []),
        qa_remediation_internal=_sanitize_qa_remediation(qa_remediation),
    )


__all__ = ["build_report", "build_text_report"]
