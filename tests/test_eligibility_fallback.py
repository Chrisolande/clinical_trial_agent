from agents import eligibility_fallback
from models.judge_verdict import JudgeVerdict

from clinical_trial_agent.constants import ELIGIBILITY_TIMEOUT_FALLBACK_MESSAGE


def test_validate_verdict_falls_back_on_invalid_schema() -> None:
    verdict = eligibility_fallback.validate_verdict({"bad": "shape"}, "N1")
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.match_tier == "weak"


def test_make_fallback_and_age_bound_extractors() -> None:
    fb = eligibility_fallback._make_fallback("k", "r", "m")
    assert fb["key_concern"] == "k"
    assert fb["rationale"] == "r"
    assert fb["critical_missing_info"] == ["m"]

    assert eligibility_fallback._extract_age_bound("Age >= 18") == (18, None)
    assert eligibility_fallback._extract_age_bound("Age <= 65") == (None, 65)
    assert eligibility_fallback._extract_age_bound("between 21 and 70") == (21, 70)


def test_inclusion_assessors_cover_age_melanoma_biomarker_performance() -> None:
    assert eligibility_fallback._assess_inclusion_age("age >= 18", 17) == (
        "FAILS",
        "Age below required threshold",
    )
    assert eligibility_fallback._assess_inclusion_age("age <= 65", 66) == (
        "FAILS",
        "Age above allowed threshold",
    )
    assert eligibility_fallback._assess_inclusion_age("age >= 18", 30) == (
        "MEETS",
        "Age appears within stated bounds",
    )

    assert eligibility_fallback._assess_inclusion_melanoma("melanoma", "melanoma history") == (
        "MEETS",
        "Diagnosis mentions melanoma and patient condition includes melanoma",
    )
    assert eligibility_fallback._assess_inclusion_melanoma("melanoma", "lung cancer") == (
        "UNCERTAIN",
        "Melanoma diagnosis not explicit in profile",
    )

    assert eligibility_fallback._assess_inclusion_biomarker(
        "braf mutation", {"biomarkers": ["BRAF V600E mutation positive"]}
    ) == (
        "MEETS",
        "Criterion-specific biomarker evidence exists in profile",
    )
    assert eligibility_fallback._assess_inclusion_biomarker("pd-l1", {"biomarkers": []}) == (
        "UNCERTAIN",
        "Criterion-specific biomarker result missing",
    )
    assert eligibility_fallback._assess_inclusion_biomarker(
        "mutation required", {"biomarkers": ["KRAS mutation positive"]}
    ) == (
        "UNCERTAIN",
        "Criterion requires specific biomarker target; generic mutation evidence is insufficient",
    )

    assert eligibility_fallback._assess_inclusion_performance(
        "ecog performance status", {"ecog_performance_status": 1}
    ) == ("MEETS", "Performance status present in profile")
    assert eligibility_fallback._assess_inclusion_performance("karnofsky", {}) == (
        "UNCERTAIN",
        "Performance status missing",
    )


def test_exclusion_assessor_and_append_outcome() -> None:
    assert eligibility_fallback._assess_exclusion("uveal melanoma", {}, "uveal melanoma") == (
        "FAILS",
        "Profile indicates uveal melanoma exclusion",
    )
    assert eligibility_fallback._assess_exclusion("pregnancy", {"sex": "male"}, "") == (
        "NOT_APPLICABLE",
        "Reproductive criterion not applicable to male patient",
    )
    assert eligibility_fallback._assess_exclusion("active infection", {}, "") == (
        "UNCERTAIN",
        "Missing exclusion-history detail",
    )

    inclusion_met: list[str] = []
    inclusion_failed: list[str] = []
    inclusion_uncertain: list[str] = []
    exclusion_triggered: list[str] = []
    exclusion_uncertain: list[str] = []
    missing: list[str] = []

    eligibility_fallback._append_outcome(
        "MEETS",
        "reason",
        "criterion A",
        is_exclusion=False,
        inclusion_met=inclusion_met,
        inclusion_failed=inclusion_failed,
        inclusion_uncertain=inclusion_uncertain,
        exclusion_triggered=exclusion_triggered,
        exclusion_uncertain=exclusion_uncertain,
        missing=missing,
    )
    eligibility_fallback._append_outcome(
        "FAILS",
        "reason",
        "criterion B",
        is_exclusion=True,
        inclusion_met=inclusion_met,
        inclusion_failed=inclusion_failed,
        inclusion_uncertain=inclusion_uncertain,
        exclusion_triggered=exclusion_triggered,
        exclusion_uncertain=exclusion_uncertain,
        missing=missing,
    )
    eligibility_fallback._append_outcome(
        "UNCERTAIN",
        "need data",
        "criterion C",
        is_exclusion=False,
        inclusion_met=inclusion_met,
        inclusion_failed=inclusion_failed,
        inclusion_uncertain=inclusion_uncertain,
        exclusion_triggered=exclusion_triggered,
        exclusion_uncertain=exclusion_uncertain,
        missing=missing,
    )

    assert inclusion_met == ["criterion A"]
    assert exclusion_triggered == ["criterion B"]
    assert inclusion_uncertain == ["criterion C"]
    assert "need data" in missing


def test_timeout_tier_selection() -> None:
    assert eligibility_fallback._select_timeout_tier_and_score(["x"], [], 1) == (
        "disqualified",
        0.0,
    )
    assert eligibility_fallback._select_timeout_tier_and_score([], ["x"], 1) == (
        "weak",
        0.3,
    )
    assert eligibility_fallback._select_timeout_tier_and_score([], [], 0) == (
        "weak",
        0.25,
    )


def test_deterministic_timeout_verdict_and_exception_fallback() -> None:
    verdict = eligibility_fallback.deterministic_timeout_verdict(
        patient_profile={"age": 45, "sex": "male", "conditions": ["melanoma"]},
        all_criteria=[
            {"criteria_type": "inclusion", "text": "Age >= 18 years"},
            {"criteria_type": "inclusion", "text": "melanoma diagnosis"},
            {"criteria_type": "exclusion", "text": "active infection"},
        ],
        trial_id="NCT1",
    )
    assert verdict.match_tier in {"moderate", "weak", "disqualified", "strong"}
    assert verdict.key_concern

    timeout_verdict = eligibility_fallback.fallback_verdict_for_exception(
        TimeoutError(),
        "NCT2",
        {"age": 50},
        [{"criteria_type": "inclusion", "text": "Age >= 18"}],
    )
    assert timeout_verdict.key_concern == ELIGIBILITY_TIMEOUT_FALLBACK_MESSAGE

    generic_verdict = eligibility_fallback.fallback_verdict_for_exception(
        RuntimeError("x"),
        "NCT3",
        {},
        [],
    )
    assert "Eligibility judge error" in generic_verdict.key_concern
