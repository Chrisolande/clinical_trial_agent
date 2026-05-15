from collections.abc import Callable
from typing import Any

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


def _trial_text_snippets(trial: dict[str, Any]) -> list[str]:
    snippets: list[str] = [str(trial.get("key_concern", "")), str(trial.get("rationale", ""))]
    for item in trial.get("critical_missing_info", []) or []:
        if isinstance(item, dict):
            snippets.extend(str(value) for value in item.values())
        else:
            snippets.append(str(item))
    return [snippet for snippet in snippets if snippet.strip()]


def _check_public_report_internal_leakage(
    scored_trials: list[dict[str, Any]], issue_builder: Callable[[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
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
        issue_builder(
            "PUBLIC_REPORT_INTERNAL_LEAKAGE",
            "critical",
            f"Internal processing language leaked into report-facing narrative in trial(s): {flagged[:5]}.",
        )
    ]


def _check_strong_too_few_criteria(
    scored_trials: list[dict[str, Any]], issue_builder: Callable[[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    flagged = [
        str(trial.get("trial_id", ""))
        for trial in scored_trials
        if str(trial.get("tier", "weak")) == "strong"
        and (int(trial.get("meets_count", 0)) + int(trial.get("fails_count", 0))) < 3
    ]
    if not flagged:
        return []
    return [
        issue_builder(
            "STRONG_TOO_FEW_CRITERIA",
            "high",
            f"Strong-tier trial(s) have too few assessed criteria (<3), which may overstate confidence: {flagged[:5]}.",
        )
    ]


def _check_strong_match_contradiction(
    scored_trials: list[dict[str, Any]], issue_builder: Callable[[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    flagged: list[str] = []
    for trial in scored_trials:
        if str(trial.get("tier", "weak")) != "strong":
            continue
        uncertain = int(trial.get("uncertain_count", 0))
        missing = [
            item for item in trial.get("critical_missing_info", []) or [] if str(item).strip()
        ]
        if uncertain <= 0 and not missing:
            continue
        narrative = f"{trial.get('key_concern', '')} {trial.get('rationale', '')}".lower()
        if any(phrase in narrative for phrase in _STRONG_OVERSTATEMENT_PHRASES):
            flagged.append(str(trial.get("trial_id", "")))
    if not flagged:
        return []
    return [
        issue_builder(
            "STRONG_MATCH_CONTRADICTION",
            "high",
            f"Strong match narrative overstates certainty despite unresolved uncertainty or gaps in trial(s): {flagged[:5]}.",
        )
    ]


def _check_na_inference_errors(
    eligibility_verdicts: dict[str, dict[str, Any]],
    issue_builder: Callable[[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
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
        issue_builder(
            "N_A_INFERENCE_ERROR",
            "high",
            f"Criteria marked as not applicable appear to have been inferred as MEETS/FAILS in trial(s): {flagged[:5]}.",
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


def _check_duplicate_missing_info(
    scored_trials: list[dict[str, Any]], issue_builder: Callable[[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    trials_with_duplicates: list[str] = []
    for trial in scored_trials:
        cleaned = [_missing_info_key(item) for item in trial.get("critical_missing_info", []) or []]
        cleaned = [value for value in cleaned if value]
        if cleaned and len(cleaned) != len(set(cleaned)):
            trials_with_duplicates.append(str(trial.get("trial_id", "")))
    if not trials_with_duplicates:
        return []
    return [
        issue_builder(
            "DUPLICATE_MISSING_INFO",
            "medium",
            f"Duplicate critical_missing_info entries detected in trial(s): {trials_with_duplicates[:5]}. Deduplicate before synthesis.",
        )
    ]


def _check_not_applicable_leakage(
    scored_trials: list[dict[str, Any]], issue_builder: Callable[[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
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
        issue_builder(
            "N_A_LEAKAGE_IN_GAPS",
            "high",
            f"Not-applicable items leaked into critical_missing_info in trial(s): {flagged[:5]}. Suppress these from information gaps and actions.",
        )
    ]


def _check_generic_biomarker_inference(
    scored_trials: list[dict[str, Any]], issue_builder: Callable[[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
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
        issue_builder(
            "GENERIC_BIOMARKER_INFERENCE",
            "high",
            f"Generic biomarker language without specific marker detected in trial(s): {flagged[:5]}. Use marker-specific evidence where possible.",
        )
    ]


def _check_summary_language_vs_critical_gaps(
    scored_trials: list[dict[str, Any]], issue_builder: Callable[[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
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
            [str(trial.get("key_concern", "")), str(trial.get("rationale", ""))]
        ).lower()
        if any(phrase in narrative for phrase in _NO_CONCERN_PHRASES):
            flagged.append(str(trial.get("trial_id", "")))
    if not flagged:
        return []
    return [
        issue_builder(
            "EXEC_SUMMARY_CONTRADICTS_CRITICAL_GAPS",
            "high",
            f"Reassuring language conflicts with critical missing information in trial narrative(s): {flagged[:5]}.",
        )
    ]


def _check_low_priority_gap_bloat(
    scored_trials: list[dict[str, Any]], issue_builder: Callable[[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
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
        issue_builder(
            "LOW_PRIORITY_GAP_BLOAT",
            "medium",
            f"Low-priority information gaps are excessively enumerated in trial(s): {flagged[:5]}. Consolidate low-priority gaps.",
        )
    ]


def check_additional_quality_rules(
    eligibility_verdicts: dict[str, dict[str, Any]],
    scored_trials: list[dict[str, Any]],
    issue_builder: Callable[[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    issues.extend(_check_public_report_internal_leakage(scored_trials, issue_builder))
    issues.extend(_check_strong_too_few_criteria(scored_trials, issue_builder))
    issues.extend(_check_strong_match_contradiction(scored_trials, issue_builder))
    issues.extend(_check_na_inference_errors(eligibility_verdicts, issue_builder))
    issues.extend(_check_duplicate_missing_info(scored_trials, issue_builder))
    issues.extend(_check_not_applicable_leakage(scored_trials, issue_builder))
    issues.extend(_check_generic_biomarker_inference(scored_trials, issue_builder))
    issues.extend(_check_low_priority_gap_bloat(scored_trials, issue_builder))
    issues.extend(_check_summary_language_vs_critical_gaps(scored_trials, issue_builder))
    return issues
