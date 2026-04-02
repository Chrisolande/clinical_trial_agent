from __future__ import annotations

import pytest
from agents import report_generator


@pytest.mark.asyncio
async def test_report_includes_excluded_trials_in_evaluated_count() -> None:
    patient_profile = {"age": 62, "sex": "male", "primary_condition": "NSCLC"}
    scored_trials = [
        {
            "trial_id": "S1",
            "brief_title": "Strong Trial",
            "tier": "strong",
            "score": 0.9,
            "meets_count": 5,
            "fails_count": 0,
            "uncertain_count": 1,
            "overall_status": "RECRUITING",
            "phase": "PHASE3",
            "critical_missing_info": [],
        },
        {
            "trial_id": "W1",
            "brief_title": "Weak Trial",
            "tier": "weak",
            "score": 0.2,
            "meets_count": 1,
            "fails_count": 1,
            "uncertain_count": 3,
            "overall_status": "RECRUITING",
            "phase": "PHASE2",
            "critical_missing_info": ["PD-L1 status unknown"],
        },
    ]

    report = await report_generator.build_report(
        patient_profile=patient_profile,
        scored_trials=scored_trials,
        missing_info=[],
        eligibility_verdicts={"S1": {}, "W1": {}},
        trials_raw=[{"nct_id": "S1"}, {"nct_id": "W1"}],
        search_queries=["nsclc recruiting"],
        decision_history=[],
        qa_issues=[],
    )

    assert report["total_trials_searched"] == 2
    assert report["total_trials_evaluated"] == 2
    assert len(report["strong_matches"]) == 1
    assert report["excluded_trial_count"] == 1
    assert len(report["excluded_trials"]) == 1
