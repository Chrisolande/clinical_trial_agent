from __future__ import annotations

import pytest
from subagents.synthesis import nodes as synthesis_nodes


@pytest.mark.asyncio
async def test_synthesis_tier_ordering_and_exclusion() -> None:
    state = {
        "patient_profile": {"age": 60},
        "trial_scores": [
            {
                "trial_id": "W1",
                "brief_title": "Weak Trial",
                "tier": "weak",
                "score": 0.3,
                "meets_count": 1,
                "fails_count": 0,
                "uncertain_count": 10,
                "overall_status": "RECRUITING",
                "phase": "PHASE2",
            },
            {
                "trial_id": "S1",
                "brief_title": "Strong Trial",
                "tier": "strong",
                "score": 0.8,
                "meets_count": 8,
                "fails_count": 0,
                "uncertain_count": 1,
                "overall_status": "RECRUITING",
                "phase": "PHASE3",
            },
            {
                "trial_id": "D1",
                "brief_title": "DQ Trial",
                "tier": "disqualified",
                "score": 0.0,
                "meets_count": 0,
                "fails_count": 2,
                "uncertain_count": 0,
                "overall_status": "RECRUITING",
                "phase": "PHASE1",
            },
            {
                "trial_id": "M1",
                "brief_title": "Moderate Trial",
                "tier": "moderate",
                "score": 0.6,
                "meets_count": 5,
                "fails_count": 1,
                "uncertain_count": 3,
                "overall_status": "RECRUITING",
                "phase": "PHASE2",
                "key_concern": "missing stage",
            },
        ],
        "eligibility_verdicts": {"S1": {}, "M1": {}, "W1": {}, "D1": {}},
        "missing_info_recommendations": [],
        "trials_raw": [{}, {}],
        "search_queries": ["q1"],
        "decision_history": [],
        "qa_issues": [],
    }

    result = await synthesis_nodes.generate_report_node(state)
    text = result["report_text"]

    strong_pos = text.find("Strong Trial")
    moderate_pos = text.find("Moderate Trial")
    weak_pos = text.find("Weak Trial")
    dq_pos = text.find("DQ Trial")

    assert strong_pos != -1
    assert moderate_pos != -1
    assert strong_pos < moderate_pos
    assert weak_pos == -1
    assert dq_pos == -1
