from typing import Any, Literal

from agents import qa_checker, report_generator
from loguru import logger

from clinical_trial_agent.config import TIER_ORDER

from .state import QAIssue, SynthesisState


def _exception_label(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    return exc.__class__.__name__


_SEVERITY_ORDER = {"critical": 3, "high": 2, "medium": 1, "low": 0}
_MAX_QA_FIX_ATTEMPTS = 2
_RETRIEVAL_RETRY_CODES = {"RETRIEVAL_FAILED_EMPTY_RESULT"}

_QA_ACTION_BY_CODE = {
    "HARD_EXCLUSION_RANKING_CONFLICT": "recompute_tier_from_verdicts",
    "ALL_UNCERTAIN_HIGH_TIER": "recompute_tier_from_verdicts",
    "AGE_VERDICT_CONTRADICTION": "reparse_age_criteria",
    "RETRIEVAL_FAILED_EMPTY_RESULT": "broaden_retrieval",
    "QA_CHECK_ERROR": "escalate_unfixable",
}

RemediationAction = Literal[
    "rerank_trials",
    "recompute_tier_from_verdicts",
    "reparse_age_criteria",
    "broaden_retrieval",
    "escalate_unfixable",
]


def _normalize_qa_issues(issues: Any) -> list[QAIssue]:
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
            normalized.append(
                {
                    "code": "UNSPECIFIED",
                    "severity": "medium",
                    "message": message,
                }
            )
    return normalized


def _issue_requires_retrieval_retry(issue: QAIssue) -> bool:
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
        key=lambda x: (
            TIER_ORDER.get(str(x.get("tier", "weak")), 0),
            float(x.get("score", 0.0)),
        ),
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
        elif (
            total > 0
            and uncertain == total
            and TIER_ORDER.get(str(updated.get("tier", "weak")), 0) >= TIER_ORDER["moderate"]
        ):
            updated["tier"] = "weak"
            updated["score"] = min(float(updated.get("score", 0.0)), 0.39)

        recomputed.append(updated)
    return _sort_trials(recomputed)


def _build_remediation_summary(state: SynthesisState) -> dict[str, Any]:
    actions = list(state.get("qa_remediation_actions") or [])
    unresolved = _normalize_qa_issues(state.get("qa_unresolved_issues") or [])
    return {
        "attempts": int(state.get("qa_fix_attempts", 0)),
        "actions": actions,
        "unresolved_issues": unresolved,
    }


async def run_qa_check(state: SynthesisState) -> dict[str, Any]:
    try:
        result = await qa_checker.run_qa_check(
            patient_profile=state.get("patient_profile") or {},
            eligibility_verdicts=state.get("eligibility_verdicts") or {},
            scored_trials=state.get("trial_scores") or [],
            retrieval_errors=list(state.get("retrieval_errors") or []),
        )
        qa_passed = result.get("qa_passed", True)
        issues = _normalize_qa_issues(result.get("qa_issues", []))
        return {
            "qa_passed": qa_passed,
            "qa_issues": issues,
            "new_decision_entries": [
                f"QA check {'passed' if qa_passed else 'found issues'}: {len(issues)} issue(s)."
            ],
        }
    except (ValueError, TypeError, TimeoutError, RuntimeError) as exc:
        label = _exception_label(exc)
        logger.error("run_qa_check failed (fail-closed, {}) : {}", label, exc)
        return {
            "qa_passed": False,
            "qa_issues": [
                {
                    "code": "QA_CHECK_ERROR",
                    "severity": "critical",
                    "message": f"QA check error ({label}): {exc}",
                }
            ],
            "new_errors": [f"qa_check[{label}]: {exc}"],
            "new_decision_entries": [f"QA check failed closed: {exc}"],
        }


def route_after_qa(
    state: SynthesisState,
) -> Literal["generate_report", "attempt_qa_fix", "flag_re_evaluation"]:
    """Route based on QA result and fix attempt count."""
    if state.get("qa_passed", True):
        return "generate_report"
    if bool(state.get("synthesis_needs_re_evaluation", False)):
        return "flag_re_evaluation"
    if state.get("qa_fix_attempts", 0) < _MAX_QA_FIX_ATTEMPTS:
        return "attempt_qa_fix"
    return "flag_re_evaluation"


async def attempt_qa_fix(state: SynthesisState) -> dict[str, Any]:
    """Attempt deterministic QA remediation based on issue codes."""
    issues = _normalize_qa_issues(state.get("qa_issues") or [])
    scored = list(state.get("trial_scores") or [])
    verdicts = state.get("eligibility_verdicts") or {}
    fix_attempts = int(state.get("qa_fix_attempts", 0))
    primary_issue = _select_primary_issue(issues)
    action = _action_for_issue(primary_issue)
    issue_code = str(primary_issue.get("code", "UNSPECIFIED")) if primary_issue else "UNSPECIFIED"

    remediation_record = {
        "attempt": str(fix_attempts + 1),
        "action": action,
        "issue_code": issue_code,
    }

    if action == "recompute_tier_from_verdicts":
        patched = _recompute_tiers_from_verdicts(scored, verdicts)
        return {
            "trial_scores": patched,
            "qa_fix_attempts": fix_attempts + 1,
            "synthesis_needs_re_evaluation": False,
            "synthesis_retry_retrieval": False,
            "qa_remediation_actions": [remediation_record],
            "new_decision_entries": [
                f"QA remediation attempt {fix_attempts + 1}: recomputed tiers from verdicts for {issue_code}."
            ],
        }

    if action == "rerank_trials":
        patched = _sort_trials(scored)
        return {
            "trial_scores": patched,
            "qa_fix_attempts": fix_attempts + 1,
            "synthesis_needs_re_evaluation": False,
            "synthesis_retry_retrieval": False,
            "qa_remediation_actions": [remediation_record],
            "new_decision_entries": [
                f"QA remediation attempt {fix_attempts + 1}: reranked trial scores for {issue_code}."
            ],
        }

    if action in {"broaden_retrieval", "reparse_age_criteria"}:
        retrieval_retry = any(_issue_requires_retrieval_retry(issue) for issue in issues)
        return {
            "qa_fix_attempts": fix_attempts + 1,
            "synthesis_needs_re_evaluation": True,
            "synthesis_retry_retrieval": retrieval_retry,
            "qa_remediation_actions": [remediation_record],
            "new_decision_entries": [
                f"QA remediation attempt {fix_attempts + 1}: escalated {issue_code} for upstream re-evaluation."
            ],
        }

    unresolved = [primary_issue] if primary_issue is not None else []
    return {
        "qa_fix_attempts": fix_attempts + 1,
        "synthesis_needs_re_evaluation": False,
        "synthesis_retry_retrieval": False,
        "qa_unresolved_issues": unresolved,
        "qa_remediation_actions": [remediation_record],
        "new_decision_entries": [
            f"QA remediation attempt {fix_attempts + 1}: marked {issue_code} unresolved for escalation."
        ],
    }


async def flag_re_evaluation(state: SynthesisState) -> dict[str, Any]:
    """Flag that the supervisor should re-run eligibility reasoning."""
    issues = _normalize_qa_issues(state.get("qa_issues") or [])
    unresolved = _normalize_qa_issues(state.get("qa_unresolved_issues") or [])
    all_issues = issues + [issue for issue in unresolved if issue not in issues]
    retry_retrieval = any(_issue_requires_retrieval_retry(issue) for issue in all_issues)
    return {
        "synthesis_needs_re_evaluation": True,
        "synthesis_retry_retrieval": retry_retrieval,
        "qa_unresolved_issues": all_issues,
        "new_decision_entries": [
            "Synthesis flagging re-evaluation needed due to persistent QA issues: "
            f"{[issue.get('code', 'UNSPECIFIED') for issue in all_issues[:3]]}"
        ],
    }


def _collect_synthesis_inputs(state: SynthesisState) -> dict[str, Any]:
    return {
        "patient_profile": state.get("patient_profile") or {},
        "scored_trials": state.get("trial_scores") or [],
        "missing_info": state.get("missing_info_recommendations") or [],
        "eligibility_verdicts": state.get("eligibility_verdicts") or {},
        "trials_raw": state.get("trials_raw") or [],
        "search_queries": state.get("search_queries") or [],
        "decision_history": (
            list(state.get("decision_history") or [])
            + list(state.get("new_decision_entries") or [])
        ),
        "qa_issues": _normalize_qa_issues(state.get("qa_issues") or []),
        "retrieval_errors": list(state.get("retrieval_errors") or []),
        "qa_remediation": _build_remediation_summary(state),
    }


async def generate_report_node(state: SynthesisState) -> dict[str, Any]:
    """Generate the final report."""
    try:
        ctx = _collect_synthesis_inputs(state)
        report = await report_generator.build_report(**ctx)
        return {
            "report_json": report,
            "report_text": report_generator.build_text_report(report),
            "synthesis_needs_re_evaluation": False,
            "synthesis_retry_retrieval": False,
            "new_decision_entries": ["Final report generated by synthesis sub-agent."],
        }
    except (ValueError, TypeError, TimeoutError, RuntimeError) as exc:
        label = _exception_label(exc)
        logger.error("generate_report failed ({}) : {}", label, exc)
        return {
            "report_json": {"error": str(exc), "error_type": label},
            "report_text": f"Report generation failed ({label}): {exc}",
            "synthesis_needs_re_evaluation": False,
            "synthesis_retry_retrieval": False,
            "new_errors": [f"generate_report[{label}]: {exc}"],
            "new_decision_entries": [f"Report generation failed: {exc}"],
        }


async def finalize_synthesis_output(state: SynthesisState) -> dict[str, Any]:
    """Collect new_decision_entries and new_errors into output keys."""
    return {
        "qa_issues": _normalize_qa_issues(state.get("qa_issues") or []),
        "qa_passed": bool(state.get("qa_passed", True)),
        "report_json": state.get("report_json"),
        "report_text": state.get("report_text"),
        "synthesis_needs_re_evaluation": bool(state.get("synthesis_needs_re_evaluation", False)),
        "synthesis_retry_retrieval": bool(state.get("synthesis_retry_retrieval", False)),
        "qa_remediation": _build_remediation_summary(state),
        "decision_history": list(state.get("new_decision_entries") or []),
        "errors": list(state.get("new_errors") or []),
    }
