from __future__ import annotations

from models.judge_verdict import JudgeVerdict


def test_disqualification_forces_zero_score() -> None:
    verdict = JudgeVerdict(
        match_tier="disqualified",
        match_score=0.8,
        major_criteria_assessable=True,
        inclusion_met=["x"],
        inclusion_failed=[],
        inclusion_uncertain=[],
        exclusion_triggered=["hard exclusion"],
        exclusion_uncertain=[],
        critical_missing_info=[],
        key_concern="hard exclusion triggered",
        rationale="disqualified",
    )
    assert verdict.match_score == 0.0


def test_uncertainty_caps_score_and_tier() -> None:
    verdict = JudgeVerdict(
        match_tier="strong",
        match_score=0.9,
        major_criteria_assessable=False,
        inclusion_met=["x"],
        inclusion_failed=[],
        inclusion_uncertain=["major unknown"],
        exclusion_triggered=[],
        exclusion_uncertain=[],
        critical_missing_info=["EGFR"],
        key_concern="major unknown",
        rationale="insufficient",
    )
    assert verdict.match_score <= 0.55
    assert verdict.match_tier in {"moderate", "weak", "disqualified"}


def test_high_uncertainty_coerces_strong_to_moderate() -> None:
    verdict = JudgeVerdict(
        match_tier="strong",
        match_score=0.8,
        major_criteria_assessable=True,
        inclusion_met=["criterion a"],
        inclusion_failed=[],
        inclusion_uncertain=["criterion b", "criterion c"],
        exclusion_triggered=[],
        exclusion_uncertain=["criterion d"],
        critical_missing_info=[],
        key_concern="uncertain",
        rationale="high uncertainty",
    )
    assert verdict.match_tier == "moderate"


def test_low_uncertainty_keeps_strong_tier() -> None:
    verdict = JudgeVerdict(
        match_tier="strong",
        match_score=0.8,
        major_criteria_assessable=True,
        inclusion_met=["criterion a", "criterion b"],
        inclusion_failed=[],
        inclusion_uncertain=["criterion c"],
        exclusion_triggered=[],
        exclusion_uncertain=[],
        critical_missing_info=[],
        key_concern="minor uncertainty",
        rationale="acceptable",
    )
    assert verdict.match_tier == "strong"
