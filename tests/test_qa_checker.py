import pytest
from agents import qa_checker


def _find_issue(issues: list[dict[str, str]], code: str) -> dict[str, str] | None:
    for issue in issues:
        if issue.get("code") == code:
            return issue
    return None


@pytest.mark.asyncio
async def test_public_report_internal_leakage_fails_qa() -> None:
    result = await qa_checker.run_qa_check(
        patient_profile={},
        eligibility_verdicts={},
        scored_trials=[
            {
                "trial_id": "T1",
                "tier": "moderate",
                "key_concern": "Judge model noted confidence concerns.",
                "rationale": "Clinical fit uncertain.",
                "critical_missing_info": [],
            }
        ],
    )

    issue = _find_issue(result["qa_issues"], "PUBLIC_REPORT_INTERNAL_LEAKAGE")
    assert result["qa_passed"] is False
    assert issue is not None
    assert issue["severity"] == "critical"


@pytest.mark.asyncio
async def test_strong_match_contradiction_detected() -> None:
    result = await qa_checker.run_qa_check(
        patient_profile={},
        eligibility_verdicts={},
        scored_trials=[
            {
                "trial_id": "T2",
                "tier": "strong",
                "meets_count": 3,
                "fails_count": 0,
                "uncertain_count": 1,
                "key_concern": "Patient fully meets all criteria with no concerns.",
                "rationale": "Looks excellent overall.",
                "critical_missing_info": ["PD-L1 status pending"],
            }
        ],
    )

    issue = _find_issue(result["qa_issues"], "STRONG_MATCH_CONTRADICTION")
    assert issue is not None
    assert issue["severity"] == "high"


@pytest.mark.asyncio
async def test_low_priority_gap_bloat_detected() -> None:
    result = await qa_checker.run_qa_check(
        patient_profile={},
        eligibility_verdicts={},
        scored_trials=[
            {
                "trial_id": "T3",
                "tier": "moderate",
                "critical_missing_info": [
                    {
                        "field_id": f"low_gap_{idx}",
                        "field": f"Low gap {idx}",
                        "description": "Ancillary detail",
                        "priority": "low",
                    }
                    for idx in range(6)
                ],
            }
        ],
    )

    issue = _find_issue(result["qa_issues"], "LOW_PRIORITY_GAP_BLOAT")
    assert issue is not None
    assert issue["severity"] == "medium"
