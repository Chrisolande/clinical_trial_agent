from typing import Any, TypedDict

from clinical_trial_agent.config import TIER_ORDER

_KNOWN_BIOMARKERS = (
    "egfr",
    "alk",
    "ros1",
    "braf",
    "kras",
    "her2",
    "pd-l1",
    "pdl1",
    "msi",
    "tmb",
    "met",
    "ret",
    "ntrk",
)

_NO_CONCERN_PHRASES = (
    "no concerns",
    "no major concerns",
    "none identified",
    "no significant concerns",
)
_NOT_APPLICABLE_MARKERS = ("not_applicable", "not applicable", "n/a", "non-applicable")
_INTERNAL_LEAKAGE_PHRASES = (
    "judge model",
    "structured verdict",
    "llm",
    "parser",
    "fallback",
    "tool failed",
    "qa issue",
)
_STRONG_OVERSTATEMENT_PHRASES = (
    "fully meets all criteria",
    "meets all criteria",
    "all criteria met",
    "no concerns",
    "no major concerns",
    "none identified",
    "no significant concerns",
)
_LOW_PRIORITY_GAP_BLOAT_THRESHOLD = 5


class QAIssue(TypedDict):
    code: str
    severity: str
    message: str


def _issue(code: str, severity: str, message: str) -> QAIssue:
    return {"code": code, "severity": severity, "message": message}


async def run_qa_check(
    patient_profile: dict[str, Any],
    eligibility_verdicts: dict[str, dict[str, Any]],
    scored_trials: list[dict[str, Any]],
    retrieval_errors: list[str] | None = None,
) -> dict[str, Any]:
    issues: list[QAIssue] = []
    errors = [str(err).strip() for err in retrieval_errors or [] if str(err).strip()]
    if errors and not scored_trials:
        issues.append(
            _issue(
                "RETRIEVAL_FAILED_EMPTY_RESULT",
                "critical",
                "Retrieval stage produced errors and no scored trials were available for synthesis.",
            )
        )
    issues.extend(_check_score_verdict_alignment(eligibility_verdicts, scored_trials))
    issues.extend(_check_age_consistency(patient_profile, eligibility_verdicts))
    issues.extend(_check_additional_quality_rules(eligibility_verdicts, scored_trials))
    qa_passed = not any(_is_blocking_issue(issue) for issue in issues)
    return {"qa_passed": qa_passed, "qa_issues": issues}


def _is_blocking_issue(issue: QAIssue) -> bool:
    return str(issue.get("severity", "")).lower() == "critical"


def _check_age_consistency(
    patient_profile: dict[str, Any],
    eligibility_verdicts: dict[str, dict[str, Any]],
) -> list[QAIssue]:
    patient_age = patient_profile.get("age")
    if not isinstance(patient_age, int | float):
        return []
    age_verdicts = _extract_age_verdicts(eligibility_verdicts)
    return _detect_age_inconsistency(int(patient_age), age_verdicts)


def _detect_age_inconsistency(
    patient_age: int, age_verdicts: list[tuple[str, str]]
) -> list[QAIssue]:
    by_trial: dict[str, set[str]] = {}
    for trial_id, verdict in age_verdicts:
        by_trial.setdefault(trial_id, set()).add(verdict)
    contradictory_trials = [
        trial_id for trial_id, verdicts in by_trial.items() if {"MEETS", "FAILS"} <= verdicts
    ]
    if contradictory_trials:
        sample = contradictory_trials[:3]
        return [
            _issue(
                "AGE_VERDICT_CONTRADICTION",
                "critical",
                (
                    f"Age criterion inconsistency within trial(s) for patient age {patient_age}: "
                    f"{sample}. Review parsed age criteria for contradictory verdicts."
                ),
            )
        ]
    return []


def _extract_age_verdicts(
    eligibility_verdicts: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    age_verdicts: list[tuple[str, str]] = []
    for trial_id, verdict_data in eligibility_verdicts.items():
        for verdict in verdict_data.get("verdicts", []):
            text_lower = str(verdict.get("criterion_text", "")).lower()
            if "age" in text_lower and (
                "years" in text_lower or "≥" in text_lower or ">=" in text_lower
            ):
                age_verdicts.append((trial_id, str(verdict.get("verdict", "UNCERTAIN"))))
    return age_verdicts


def _check_score_verdict_alignment(
    eligibility_verdicts: dict[str, dict[str, Any]],
    scored_trials: list[dict[str, Any]],
) -> list[QAIssue]:
    issues: list[QAIssue] = []
    score_lookup = {str(trial["trial_id"]): trial for trial in scored_trials if "trial_id" in trial}
    for trial_id, verdict in eligibility_verdicts.items():
        scored = score_lookup.get(trial_id)
        if scored is None:
            continue

        hard_failures = int(verdict.get("hard_exclusion_failures", 0))
        tier = str(scored.get("tier", "weak"))
        if hard_failures >= 1 and TIER_ORDER.get(tier, 0) >= TIER_ORDER["moderate"]:
            issues.append(
                _issue(
                    "HARD_EXCLUSION_RANKING_CONFLICT",
                    "critical",
                    (
                        f"Inconsistency: {trial_id} has hard exclusion failures but tier={tier}. "
                        "Hard exclusions should strongly suppress ranking."
                    ),
                )
            )

        uncertain = int(verdict.get("uncertain_count", 0))
        total = int(verdict.get("meets_count", 0)) + int(verdict.get("fails_count", 0)) + uncertain
        if total > 0 and uncertain == total and TIER_ORDER.get(tier, 0) >= TIER_ORDER["moderate"]:
            issues.append(
                _issue(
                    "ALL_UNCERTAIN_HIGH_TIER",
                    "high",
                    f"Warning: {trial_id} has all UNCERTAIN verdicts but tier={tier}.",
                )
            )
    return issues


def _check_additional_quality_rules(
    eligibility_verdicts: dict[str, dict[str, Any]],
    scored_trials: list[dict[str, Any]],
) -> list[QAIssue]:
    issues: list[QAIssue] = []
    issues.extend(_check_public_report_internal_leakage(scored_trials))
    issues.extend(_check_strong_too_few_criteria(scored_trials))
    issues.extend(_check_strong_match_contradiction(scored_trials))
    issues.extend(_check_na_inference_errors(eligibility_verdicts))
    issues.extend(_check_duplicate_missing_info(scored_trials))
    issues.extend(_check_not_applicable_leakage(scored_trials))
    issues.extend(_check_generic_biomarker_inference(scored_trials))
    issues.extend(_check_low_priority_gap_bloat(scored_trials))
    issues.extend(_check_summary_language_vs_critical_gaps(scored_trials))
    return issues


def _trial_text_snippets(trial: dict[str, Any]) -> list[str]:
    snippets: list[str] = [str(trial.get("key_concern", "")), str(trial.get("rationale", ""))]
    for item in trial.get("critical_missing_info", []) or []:
        if isinstance(item, dict):
            snippets.extend(str(value) for value in item.values())
        else:
            snippets.append(str(item))
    return [snippet for snippet in snippets if snippet.strip()]


def _check_public_report_internal_leakage(scored_trials: list[dict[str, Any]]) -> list[QAIssue]:
    flagged: list[str] = []
    for trial in scored_trials:
        lowered_snippets = [snippet.lower() for snippet in _trial_text_snippets(trial)]
        if any(
            phrase in snippet
            for snippet in lowered_snippets
            for phrase in _INTERNAL_LEAKAGE_PHRASES
        ):
            flagged.append(str(trial.get("trial_id", "")))
    if not flagged:
        return []
    return [
        _issue(
            "PUBLIC_REPORT_INTERNAL_LEAKAGE",
            "critical",
            (
                "Internal processing language leaked into report-facing narrative in trial(s): "
                f"{flagged[:5]}."
            ),
        )
    ]


def _check_strong_too_few_criteria(scored_trials: list[dict[str, Any]]) -> list[QAIssue]:
    flagged: list[str] = []
    for trial in scored_trials:
        if str(trial.get("tier", "weak")) != "strong":
            continue
        assessed = int(trial.get("meets_count", 0)) + int(trial.get("fails_count", 0))
        if assessed < 3:
            flagged.append(str(trial.get("trial_id", "")))
    if not flagged:
        return []
    return [
        _issue(
            "STRONG_TOO_FEW_CRITERIA",
            "high",
            (
                "Strong-tier trial(s) have too few assessed criteria (<3), which may overstate confidence: "
                f"{flagged[:5]}."
            ),
        )
    ]


def _check_strong_match_contradiction(scored_trials: list[dict[str, Any]]) -> list[QAIssue]:
    flagged: list[str] = []
    for trial in scored_trials:
        if str(trial.get("tier", "weak")) != "strong":
            continue
        uncertain = int(trial.get("uncertain_count", 0))
        missing = [item for item in trial.get("critical_missing_info", []) or [] if str(item).strip()]
        if uncertain <= 0 and not missing:
            continue
        narrative = f"{trial.get('key_concern', '')} {trial.get('rationale', '')}".lower()
        if any(phrase in narrative for phrase in _STRONG_OVERSTATEMENT_PHRASES):
            flagged.append(str(trial.get("trial_id", "")))
    if not flagged:
        return []
    return [
        _issue(
            "STRONG_MATCH_CONTRADICTION",
            "high",
            (
                "Strong match narrative overstates certainty despite unresolved uncertainty or gaps in "
                f"trial(s): {flagged[:5]}."
            ),
        )
    ]


def _check_na_inference_errors(eligibility_verdicts: dict[str, dict[str, Any]]) -> list[QAIssue]:
    flagged: list[str] = []
    na_markers = ("n/a", "not applicable", "non-applicable")
    for trial_id, verdict_data in eligibility_verdicts.items():
        for verdict in verdict_data.get("verdicts", []):
            verdict_label = str(verdict.get("verdict", "")).upper()
            criterion_text = str(verdict.get("criterion_text", "")).lower()
            if verdict_label in {"MEETS", "FAILS"} and any(m in criterion_text for m in na_markers):
                flagged.append(str(trial_id))
                break
    if not flagged:
        return []
    return [
        _issue(
            "N_A_INFERENCE_ERROR",
            "high",
            (
                "Criteria marked as not applicable appear to have been inferred as MEETS/FAILS in "
                f"trial(s): {flagged[:5]}."
            ),
        )
    ]


def _check_duplicate_missing_info(scored_trials: list[dict[str, Any]]) -> list[QAIssue]:
    trials_with_duplicates: list[str] = []
    for trial in scored_trials:
        cleaned = [_missing_info_key(item) for item in trial.get("critical_missing_info", []) or []]
        cleaned = [value for value in cleaned if value]
        if cleaned and len(cleaned) != len(set(cleaned)):
            trials_with_duplicates.append(str(trial.get("trial_id", "")))
    if not trials_with_duplicates:
        return []
    return [
        _issue(
            "DUPLICATE_MISSING_INFO",
            "medium",
            (
                "Duplicate critical_missing_info entries detected in trial(s): "
                f"{trials_with_duplicates[:5]}. Deduplicate before synthesis."
            ),
        )
    ]


def _missing_info_key(item: Any) -> str:
    if isinstance(item, dict):
        return (
            str(item.get("field_id") or item.get("field") or item.get("display_name") or "")
            .strip()
            .lower()
        )
    return str(item).strip().lower()


def _check_not_applicable_leakage(scored_trials: list[dict[str, Any]]) -> list[QAIssue]:
    flagged: list[str] = []
    for trial in scored_trials:
        for item in trial.get("critical_missing_info", []) or []:
            text = str(item).strip().lower()
            if text and any(marker in text for marker in _NOT_APPLICABLE_MARKERS):
                flagged.append(str(trial.get("trial_id", "")))
                break
    if not flagged:
        return []
    return [
        _issue(
            "N_A_LEAKAGE_IN_GAPS",
            "high",
            (
                "Not-applicable items leaked into critical_missing_info in trial(s): "
                f"{flagged[:5]}. Suppress these from information gaps and actions."
            ),
        )
    ]


def _check_generic_biomarker_inference(scored_trials: list[dict[str, Any]]) -> list[QAIssue]:
    flagged: list[str] = []
    for trial in scored_trials:
        texts = [
            str(trial.get("key_concern", "")),
            str(trial.get("rationale", "")),
            *[str(item) for item in trial.get("critical_missing_info", []) or []],
        ]
        for text in texts:
            lowered = text.lower()
            if "biomarker" not in lowered:
                continue
            if not any(marker in lowered for marker in _KNOWN_BIOMARKERS):
                flagged.append(str(trial.get("trial_id", "")))
                break
    if not flagged:
        return []
    return [
        _issue(
            "GENERIC_BIOMARKER_INFERENCE",
            "high",
            (
                "Generic biomarker language without specific marker detected in trial(s): "
                f"{flagged[:5]}. Use marker-specific evidence where possible."
            ),
        )
    ]


def _check_summary_language_vs_critical_gaps(scored_trials: list[dict[str, Any]]) -> list[QAIssue]:
    flagged: list[str] = []
    for trial in scored_trials:
        missing = [
            str(item).strip()
            for item in trial.get("critical_missing_info", []) or []
            if str(item).strip()
        ]
        if not missing:
            continue
        narrative = " ".join(
            [
                str(trial.get("key_concern", "")),
                str(trial.get("rationale", "")),
            ]
        ).lower()
        if any(phrase in narrative for phrase in _NO_CONCERN_PHRASES):
            flagged.append(str(trial.get("trial_id", "")))
    if not flagged:
        return []
    return [
        _issue(
            "EXEC_SUMMARY_CONTRADICTS_CRITICAL_GAPS",
            "high",
            (
                "Reassuring language conflicts with critical missing information in trial narrative(s): "
                f"{flagged[:5]}."
            ),
        )
    ]


def _check_low_priority_gap_bloat(scored_trials: list[dict[str, Any]]) -> list[QAIssue]:
    flagged: list[str] = []
    for trial in scored_trials:
        low_priority_count = 0
        for item in trial.get("critical_missing_info", []) or []:
            if isinstance(item, dict):
                priority = str(item.get("priority", "")).strip().lower()
                if priority == "low":
                    low_priority_count += 1
                    continue
                descriptor = " ".join(
                    str(item.get(key, "")) for key in ("field_id", "field", "description")
                ).lower()
                if "low priority" in descriptor:
                    low_priority_count += 1
            else:
                text = str(item).lower()
                if "low priority" in text or text.endswith("(low)"):
                    low_priority_count += 1
        if low_priority_count >= _LOW_PRIORITY_GAP_BLOAT_THRESHOLD:
            flagged.append(str(trial.get("trial_id", "")))
    if not flagged:
        return []
    return [
        _issue(
            "LOW_PRIORITY_GAP_BLOAT",
            "medium",
            (
                "Low-priority information gaps are excessively enumerated in trial(s): "
                f"{flagged[:5]}. Consolidate low-priority gaps."
            ),
        )
    ]
