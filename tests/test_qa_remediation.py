import pytest
from subagents.synthesis import qa_remediation


def test_normalize_qa_issues_accepts_dicts_and_strings() -> None:
    issues = [
        {"code": "X", "severity": "critical", "message": "m"},
        "plain message",
    ]
    normalized = qa_remediation.normalize_qa_issues(issues)
    assert normalized[0]["code"] == "X"
    assert normalized[1]["code"] == "UNSPECIFIED"


@pytest.mark.asyncio
async def test_attempt_qa_fix_recomputes_tiers_from_verdicts() -> None:
    state = {
        "qa_issues": [{"code": "HARD_EXCLUSION_RANKING_CONFLICT", "severity": "critical"}],
        "trial_scores": [{"trial_id": "T1", "tier": "moderate", "score": 0.9}],
        "eligibility_verdicts": {"T1": {"hard_exclusion_failures": 1}},
        "qa_fix_attempts": 0,
    }
    out = await qa_remediation.attempt_qa_fix(state)  # type: ignore[arg-type]
    assert out["trial_scores"][0]["tier"] == "disqualified"
    assert out["qa_fix_attempts"] == 1


@pytest.mark.asyncio
async def test_attempt_qa_fix_sanitizes_information_gaps() -> None:
    state = {
        "qa_issues": [{"code": "N_A_LEAKAGE_IN_GAPS", "severity": "high"}],
        "trial_scores": [
            {
                "trial_id": "T1",
                "critical_missing_info": ["n/a", "EGFR mutation status", "EGFR mutation status"],
            }
        ],
        "eligibility_verdicts": {},
        "qa_fix_attempts": 0,
    }
    out = await qa_remediation.attempt_qa_fix(state)  # type: ignore[arg-type]
    gaps = out["trial_scores"][0]["critical_missing_info"]
    assert "n/a" not in " ".join(str(g) for g in gaps).lower()
    assert len(gaps) == 1


@pytest.mark.asyncio
async def test_attempt_qa_fix_reranks_trials_when_no_issues() -> None:
    state = {
        "qa_issues": [],
        "trial_scores": [
            {"trial_id": "T1", "tier": "weak", "score": 0.4},
            {"trial_id": "T2", "tier": "moderate", "score": 0.2},
        ],
        "eligibility_verdicts": {},
        "qa_fix_attempts": 0,
    }
    out = await qa_remediation.attempt_qa_fix(state)  # type: ignore[arg-type]
    assert out["trial_scores"][0]["trial_id"] == "T2"
