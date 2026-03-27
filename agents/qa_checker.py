from __future__ import annotations


async def qa_check(patient_profile, eligibility_verdicts, scored_trials):
    issues = []
    issues.extend(_check_score_verdict_alignment(eligibility_verdicts, scored_trials))
    issues.extend(_check_age_consistency(patient_profile, eligibility_verdicts))

    qa_passed = len([i for i in issues if "CRITICAL" in i.upper()]) == 0
    return {
        "qa_passed": qa_passed,
        "qa_issues": issues,
    }


def _check_age_consistency(patient_profile, eligibility_verdicts):
    # issues = []
    patient_age = patient_profile.get("age")
    if not patient_age:
        return []
    age_verdicts = _extract_age_verdicts(eligibility_verdicts)
    return _detect_age_inconsistency(patient_age, age_verdicts)


def _detect_age_inconsistency(patient_age, age_verdicts):
    age_meets = [tid for tid, vrd in age_verdicts if vrd == "MEETS"]
    age_fails = [tid for tid, vrd in age_verdicts if vrd == "FAILS"]
    if age_meets and age_fails and len(age_meets) > 2 and len(age_fails) > 2:
        return [
            f"Age criterion inconsistency: patient (age {patient_age}) both meets and fails "
            f"age criteria across trials. Review: meets in {age_meets[:2]}, fails in {age_fails[:2]}."
        ]
    return []


def _extract_age_verdicts(eligibility_verdicts):
    age_verdicts = []
    for trial_id, verdict_data in eligibility_verdicts.items():
        for v in verdict_data.get("verdicts", []):
            text_lower = v.get("criterion_text", "").lower()
            if "age" in text_lower and (
                "years" in text_lower or "≥" in text_lower or ">=" in text_lower
            ):
                age_verdicts.append((trial_id, v.get("verdict", "UNCERTAIN")))
        return age_verdicts


def _check_score_verdict_alignment(eligibility_verdicts, scored_trials) -> list[str]:
    """Check that scores align with verdict data (simplified)."""
    issues = []
    score_lookup = {t["trial_id"]: t for t in scored_trials}

    for trial_id, verdict in eligibility_verdicts.items():
        scored = score_lookup.get(trial_id)
        if not scored:
            continue

        if verdict.get("hard_exclusion_failures", 0) >= 2 and scored.get("score", 0.0) >= 0.7:
            issues.append(
                f"Inconsistency: {trial_id} has {verdict.get('hard_exclusion_failures', 0)} hard exclusion failures "
                f"but score={scored.get('score', 0.0):.2f} (strong match tier). Score may be overestimated."
            )

        meets = verdict.get("meets_count", 0)
        fails = verdict.get("fails_count", 0)
        uncertain = verdict.get("uncertain_count", 0)
        total = meets + fails + uncertain
        if total > 0 and uncertain == total and scored.get("confidence", 1.0) > 0.5:
            issues.append(
                f"Warning: {trial_id} has all UNCERTAIN verdicts but confidence={scored.get('confidence', 1.0):.2f}."
            )

    return issues
