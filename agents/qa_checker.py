from __future__ import annotations

from typing import Any

from config import TIER_ORDER


async def run_qa_check(
    patient_profile: dict[str, Any],
    eligibility_verdicts: dict[str, dict[str, Any]],
    scored_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[str] = []
    issues.extend(_check_score_verdict_alignment(eligibility_verdicts, scored_trials))
    issues.extend(_check_age_consistency(patient_profile, eligibility_verdicts))
    qa_passed = not any("CRITICAL" in issue.upper() for issue in issues)
    return {"qa_passed": qa_passed, "qa_issues": issues}


def _check_age_consistency(
    patient_profile: dict[str, Any],
    eligibility_verdicts: dict[str, dict[str, Any]],
) -> list[str]:
    patient_age = patient_profile.get("age")
    if not isinstance(patient_age, int | float):
        return []
    age_verdicts = _extract_age_verdicts(eligibility_verdicts)
    return _detect_age_inconsistency(int(patient_age), age_verdicts)


def _detect_age_inconsistency(patient_age: int, age_verdicts: list[tuple[str, str]]) -> list[str]:
    by_trial: dict[str, set[str]] = {}
    for trial_id, verdict in age_verdicts:
        by_trial.setdefault(trial_id, set()).add(verdict)
    contradictory_trials = [
        trial_id for trial_id, verdicts in by_trial.items() if {"MEETS", "FAILS"} <= verdicts
    ]
    if contradictory_trials:
        sample = contradictory_trials[:3]
        return [
            (
                f"Age criterion inconsistency within trial(s) for patient age {patient_age}: "
                f"{sample}. Review parsed age criteria for contradictory verdicts."
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
) -> list[str]:
    issues: list[str] = []
    score_lookup = {str(trial["trial_id"]): trial for trial in scored_trials if "trial_id" in trial}
    for trial_id, verdict in eligibility_verdicts.items():
        scored = score_lookup.get(trial_id)
        if scored is None:
            continue

        hard_failures = int(verdict.get("hard_exclusion_failures", 0))
        tier = str(scored.get("tier", "weak"))
        if hard_failures >= 1 and TIER_ORDER.get(tier, 0) >= TIER_ORDER["moderate"]:
            issues.append(
                f"Inconsistency: {trial_id} has hard exclusion failures but tier={tier}. "
                "Hard exclusions should strongly suppress ranking."
            )

        uncertain = int(verdict.get("uncertain_count", 0))
        total = int(verdict.get("meets_count", 0)) + int(verdict.get("fails_count", 0)) + uncertain
        if total > 0 and uncertain == total and TIER_ORDER.get(tier, 0) >= TIER_ORDER["moderate"]:
            issues.append(f"Warning: {trial_id} has all UNCERTAIN verdicts but tier={tier}.")
    return issues
