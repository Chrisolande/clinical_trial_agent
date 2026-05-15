from datetime import UTC, datetime
from typing import Any

from clinical_trial_agent.config import TIER_ORDER

from .report_formatter import build_text_report
from .report_generator_gaps import merge_information_gaps
from .report_synthesizer import generate_report_plan

_METHODOLOGY_NOTE = (
    "This report uses structured eligibility assessment with conservative tiering. "
    "Strong matches require confidence on major criteria and no disqualifying exclusion triggers. "
    "Weak/disqualified trials are excluded from the main body."
)


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
            normalized.append({"code": "UNSPECIFIED", "severity": "medium", "message": message})
    return normalized


def _sanitize_qa_remediation(qa_remediation: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(qa_remediation, dict):
        return {"attempts": 0, "actions": [], "unresolved_issues": []}
    return {
        "attempts": int(qa_remediation.get("attempts", 0) or 0),
        "actions": list(qa_remediation.get("actions") or []),
        "unresolved_issues": _normalize_qa_issues(
            list(qa_remediation.get("unresolved_issues") or [])
        ),
    }


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
        rows.append(
            {
                "criterion_text": text,
                "verdict": mapping.get(str(item.get("verdict", "UNCERTAIN")), "uncertain"),
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
    report_plan_dict = (
        report_plan.model_dump() if hasattr(report_plan, "model_dump") else dict(report_plan)
    )
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
    merged_gaps = merge_information_gaps(ranked, missing_info)
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


def _merge_information_gaps(
    scored_trials: list[dict[str, Any]], missing_info: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return merge_information_gaps(scored_trials, missing_info)


__all__ = ["build_report", "build_text_report"]
