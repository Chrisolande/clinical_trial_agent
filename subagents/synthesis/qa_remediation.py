from typing import Any, Literal

from clinical_trial_agent.config import TIER_ORDER

from .state import QAIssue, SynthesisState

_SEVERITY_ORDER = {"critical": 3, "high": 2, "medium": 1, "low": 0}
_RETRIEVAL_RETRY_CODES = {"RETRIEVAL_FAILED_EMPTY_RESULT"}
_NOT_APPLICABLE_MARKERS = ("not_applicable", "not applicable", "n/a", "non-applicable")
_INTERNAL_GAP_MARKERS = (
    "judge model",
    "structured verdict",
    "llm",
    "parser",
    "fallback",
    "tool failed",
    "qa issue",
)
_QA_ACTION_BY_CODE = {
    "HARD_EXCLUSION_RANKING_CONFLICT": "recompute_tier_from_verdicts",
    "ALL_UNCERTAIN_HIGH_TIER": "recompute_tier_from_verdicts",
    "STRONG_TOO_FEW_CRITERIA": "recompute_tier_from_verdicts",
    "AGE_VERDICT_CONTRADICTION": "reparse_age_criteria",
    "RETRIEVAL_FAILED_EMPTY_RESULT": "broaden_retrieval",
    "N_A_LEAKAGE_IN_GAPS": "sanitize_information_gaps",
    "DUPLICATE_MISSING_INFO": "sanitize_information_gaps",
    "GENERIC_BIOMARKER_INFERENCE": "sanitize_information_gaps",
    "EXEC_SUMMARY_CONTRADICTS_CRITICAL_GAPS": "regenerate_report_plan",
    "QA_CHECK_ERROR": "escalate_unfixable",
}

RemediationAction = Literal[
    "rerank_trials",
    "recompute_tier_from_verdicts",
    "reparse_age_criteria",
    "broaden_retrieval",
    "sanitize_information_gaps",
    "regenerate_report_plan",
    "escalate_unfixable",
]


def normalize_qa_issues(issues: Any) -> list[QAIssue]:
    normalized: list[QAIssue] = []
    if not isinstance(issues, list):
        return normalized
    for issue in issues:
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


def build_remediation_summary(state: SynthesisState) -> dict[str, Any]:
    actions = list(state.get("qa_remediation_actions") or [])
    unresolved = normalize_qa_issues(state.get("qa_unresolved_issues") or [])
    return {"attempts": int(state.get("qa_fix_attempts", 0)), "actions": actions, "unresolved_issues": unresolved}


def issue_requires_retrieval_retry(issue: QAIssue) -> bool:
    return str(issue.get("code", "")).strip() in _RETRIEVAL_RETRY_CODES


def _select_primary_issue(issues: list[QAIssue]) -> QAIssue | None:
    if not issues:
        return None
    return max(
        issues,
        key=lambda issue: (
            _SEVERITY_ORDER.get(str(issue.get("severity", "medium")).lower(), 1),
            str(issue.get("code", "")),
        ),
    )


def _action_for_issue(issue: QAIssue | None) -> RemediationAction:
    if issue is None:
        return "rerank_trials"
    mapped = _QA_ACTION_BY_CODE.get(str(issue.get("code", "")).strip(), "rerank_trials")
    return mapped  # type: ignore[return-value]


def _sort_trials(scored_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = [dict(trial) for trial in scored_trials]
    ordered.sort(
        key=lambda x: (TIER_ORDER.get(str(x.get("tier", "weak")), 0), float(x.get("score", 0.0))),
        reverse=True,
    )
    return ordered


def _recompute_tiers_from_verdicts(
    scored_trials: list[dict[str, Any]],
    eligibility_verdicts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    recomputed: list[dict[str, Any]] = []
    for trial in scored_trials:
        updated = dict(trial)
        trial_id = str(updated.get("trial_id", ""))
        verdict = eligibility_verdicts.get(trial_id, {})
        hard_failures = int(verdict.get("hard_exclusion_failures", 0))
        uncertain = int(verdict.get("uncertain_count", updated.get("uncertain_count", 0)))
        meets = int(verdict.get("meets_count", updated.get("meets_count", 0)))
        fails = int(verdict.get("fails_count", updated.get("fails_count", 0)))
        total = meets + fails + uncertain

        if hard_failures >= 1:
            updated["tier"] = "disqualified"
            updated["score"] = min(float(updated.get("score", 0.0)), 0.2)
        elif total > 0 and uncertain == total and TIER_ORDER.get(str(updated.get("tier", "weak")), 0) >= TIER_ORDER["moderate"]:
            updated["tier"] = "weak"
            updated["score"] = min(float(updated.get("score", 0.0)), 0.39)
        recomputed.append(updated)
    return _sort_trials(recomputed)


def _normalize_gap_key(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("field_id") or item.get("field") or item.get("display_name") or item.get("description") or "").strip().lower()
    return str(item).strip().lower()


def _gap_text(item: Any) -> str:
    if isinstance(item, dict):
        return " ".join(str(value) for value in item.values()).strip().lower()
    return str(item).strip().lower()


def _is_not_applicable_gap(item: Any) -> bool:
    text = _gap_text(item)
    return bool(text) and any(marker in text for marker in _NOT_APPLICABLE_MARKERS)


def _is_internal_failure_gap(item: Any) -> bool:
    text = _gap_text(item)
    return bool(text) and any(marker in text for marker in _INTERNAL_GAP_MARKERS)


def _sanitize_information_gaps(scored_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for trial in scored_trials:
        updated = dict(trial)
        cleaned: list[Any] = []
        seen: set[str] = set()
        for item in trial.get("critical_missing_info", []) or []:
            if _is_not_applicable_gap(item) or _is_internal_failure_gap(item):
                continue
            key = _normalize_gap_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned.append(item)
        updated["critical_missing_info"] = cleaned
        sanitized.append(updated)
    return sanitized


async def attempt_qa_fix(state: SynthesisState) -> dict[str, Any]:
    issues = normalize_qa_issues(state.get("qa_issues") or [])
    scored = list(state.get("trial_scores") or [])
    verdicts = state.get("eligibility_verdicts") or {}
    fix_attempts = int(state.get("qa_fix_attempts", 0))
    primary_issue = _select_primary_issue(issues)
    action = _action_for_issue(primary_issue)
    issue_code = str(primary_issue.get("code", "UNSPECIFIED")) if primary_issue else "UNSPECIFIED"
    remediation_record = {"attempt": str(fix_attempts + 1), "action": action, "issue_code": issue_code}

    if action == "recompute_tier_from_verdicts":
        patched = _recompute_tiers_from_verdicts(scored, verdicts)
        return {"trial_scores": patched, "qa_fix_attempts": fix_attempts + 1, "synthesis_needs_re_evaluation": False, "synthesis_retry_retrieval": False, "qa_remediation_actions": [remediation_record], "new_decision_entries": [f"QA remediation attempt {fix_attempts + 1}: recomputed tiers from verdicts for {issue_code}."]}
    if action == "sanitize_information_gaps":
        patched = _sanitize_information_gaps(scored)
        return {"trial_scores": patched, "qa_fix_attempts": fix_attempts + 1, "synthesis_needs_re_evaluation": False, "synthesis_retry_retrieval": False, "qa_remediation_actions": [remediation_record], "new_decision_entries": [f"QA remediation attempt {fix_attempts + 1}: sanitized information gaps for {issue_code}."]}
    if action == "regenerate_report_plan":
        return {"qa_fix_attempts": fix_attempts + 1, "synthesis_needs_re_evaluation": False, "synthesis_retry_retrieval": False, "qa_remediation_actions": [remediation_record], "new_decision_entries": [f"QA remediation attempt {fix_attempts + 1}: regenerated report plan for {issue_code}."]}
    if action == "rerank_trials":
        patched = _sort_trials(scored)
        return {"trial_scores": patched, "qa_fix_attempts": fix_attempts + 1, "synthesis_needs_re_evaluation": False, "synthesis_retry_retrieval": False, "qa_remediation_actions": [remediation_record], "new_decision_entries": [f"QA remediation attempt {fix_attempts + 1}: reranked trial scores for {issue_code}."]}
    if action in {"broaden_retrieval", "reparse_age_criteria"}:
        retrieval_retry = any(issue_requires_retrieval_retry(issue) for issue in issues)
        return {"qa_fix_attempts": fix_attempts + 1, "synthesis_needs_re_evaluation": True, "synthesis_retry_retrieval": retrieval_retry, "qa_remediation_actions": [remediation_record], "new_decision_entries": [f"QA remediation attempt {fix_attempts + 1}: escalated {issue_code} for upstream retry."]}
    unresolved = [primary_issue] if primary_issue is not None else []
    return {"qa_fix_attempts": fix_attempts + 1, "synthesis_needs_re_evaluation": False, "synthesis_retry_retrieval": False, "qa_unresolved_issues": unresolved, "qa_remediation_actions": [remediation_record], "new_decision_entries": [f"QA remediation attempt {fix_attempts + 1}: marked {issue_code} unresolved for escalation."]}


async def flag_re_evaluation(state: SynthesisState) -> dict[str, Any]:
    issues = normalize_qa_issues(state.get("qa_issues") or [])
    unresolved = normalize_qa_issues(state.get("qa_unresolved_issues") or [])
    all_issues = issues + [issue for issue in unresolved if issue not in issues]
    retry_retrieval = any(issue_requires_retrieval_retry(issue) for issue in all_issues)
    return {
        "synthesis_needs_re_evaluation": True,
        "synthesis_retry_retrieval": retry_retrieval,
        "qa_unresolved_issues": all_issues,
        "new_decision_entries": [f"Synthesis flagging retry due to persistent QA issues: {[issue.get('code', 'UNSPECIFIED') for issue in all_issues[:3]]}"],
    }


def collect_synthesis_inputs(state: SynthesisState) -> dict[str, Any]:
    return {
        "patient_profile": state.get("patient_profile") or {},
        "scored_trials": state.get("trial_scores") or [],
        "missing_info": state.get("missing_info_recommendations") or [],
        "eligibility_verdicts": state.get("eligibility_verdicts") or {},
        "trials_raw": state.get("trials_raw") or [],
        "search_queries": state.get("search_queries") or [],
        "decision_history": list(state.get("decision_history") or []) + list(state.get("new_decision_entries") or []),
        "qa_issues": normalize_qa_issues(state.get("qa_issues") or []),
        "retrieval_errors": list(state.get("retrieval_errors") or []),
        "qa_remediation": build_remediation_summary(state),
    }
