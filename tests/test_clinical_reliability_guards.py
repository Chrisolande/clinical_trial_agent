from agents.eligibility_fallback import deterministic_timeout_verdict
from agents.report_generator_gaps import _deduplicate_gaps
from agents.report_generator_gaps import merge_information_gaps as _merge_information_gaps
from agents.report_synthesizer import _build_exec_summary_context
from models.judge_verdict import JudgeVerdict
from models.missing_info import CompletenessAssessment


def test_timeout_fallback_biomarker_requires_exact_target() -> None:
    verdict = deterministic_timeout_verdict(
        patient_profile={"sex": "female", "biomarkers": ["KRAS G12C mutation positive"]},
        all_criteria=[{"criteria_type": "inclusion", "text": "PD-L1 expression required"}],
        trial_id="NCT-TEST-1",
    )
    assert "PD-L1 expression required" in verdict.inclusion_uncertain
    assert "PD-L1 expression required" not in verdict.inclusion_met


def test_timeout_fallback_marks_male_reproductive_criteria_not_applicable() -> None:
    verdict = deterministic_timeout_verdict(
        patient_profile={"sex": "male"},
        all_criteria=[
            {"criteria_type": "exclusion", "text": "Pregnant or breastfeeding patients excluded"}
        ],
        trial_id="NCT-TEST-2",
    )
    assert verdict.exclusion_uncertain == []
    assert not any("pregnan" in gap.lower() for gap in verdict.critical_missing_info)


def test_timeout_fallback_keeps_male_contraception_criteria_assessable() -> None:
    verdict = deterministic_timeout_verdict(
        patient_profile={"sex": "male"},
        all_criteria=[
            {"criteria_type": "inclusion", "text": "Male subjects must use contraception"}
        ],
        trial_id="NCT-TEST-2B",
    )
    assert "Male subjects must use contraception" in verdict.inclusion_uncertain


def test_judge_verdict_downgrades_strong_when_evidence_floor_not_met() -> None:
    verdict = JudgeVerdict(
        match_score=0.9,
        match_tier="strong",
        major_criteria_assessable=True,
        inclusion_met=["EGFR mutation positive"],
        inclusion_failed=[],
        inclusion_uncertain=[],
        exclusion_triggered=[],
        exclusion_uncertain=[],
        critical_missing_info=[],
        key_concern="none",
        rationale="test",
    )
    assert verdict.match_tier == "moderate"
    assert verdict.match_score <= 0.74


def test_judge_verdict_downgrades_strong_when_exclusion_uncertain_present() -> None:
    verdict = JudgeVerdict(
        match_score=0.88,
        match_tier="strong",
        major_criteria_assessable=True,
        inclusion_met=["EGFR mutation positive", "Stage IV NSCLC", "ECOG 0-1"],
        inclusion_failed=[],
        inclusion_uncertain=[],
        exclusion_triggered=[],
        exclusion_uncertain=["Active infection"],
        critical_missing_info=[],
        key_concern="none",
        rationale="test",
    )
    assert verdict.match_tier == "moderate"


def test_judge_verdict_high_uncertainty_cannot_remain_strong() -> None:
    verdict = JudgeVerdict(
        match_score=0.95,
        match_tier="strong",
        major_criteria_assessable=True,
        inclusion_met=["EGFR mutation positive", "Stage IV NSCLC"],
        inclusion_failed=[],
        inclusion_uncertain=["ECOG", "Prior therapy"],
        exclusion_triggered=[],
        exclusion_uncertain=["Active infection history"],
        critical_missing_info=[],
        key_concern="none",
        rationale="test",
    )
    assert verdict.match_tier != "strong"


def test_information_gaps_deduplicate_by_field_id() -> None:
    gaps = _deduplicate_gaps(
        [
            {
                "field_id": "molecular_biomarker_egfr",
                "display_name": "EGFR mutation status",
                "field": "EGFR mutation status",
                "description": "Needed for trial A",
                "priority": "medium",
                "affected_trial_ids": ["NCT1"],
            },
            {
                "field_id": "molecular_biomarker_egfr",
                "display_name": "EGFR mutation status",
                "field": "EGFR mutation status",
                "description": "Needed for trial B and C",
                "priority": "high",
                "affected_trial_ids": ["NCT2", "NCT3"],
            },
        ]
    )
    assert len(gaps) == 1
    assert gaps[0]["field_id"] == "molecular_biomarker_egfr"
    assert gaps[0]["priority"] == "high"
    assert gaps[0]["affected_trial_ids"] == ["NCT1", "NCT2", "NCT3"]


def test_information_gaps_priority_escalates_from_low_to_high() -> None:
    gaps = _deduplicate_gaps(
        [
            {
                "field_id": "labs_tsh",
                "display_name": "TSH",
                "field": "TSH",
                "description": "Needed by one trial",
                "priority": "low",
                "affected_trial_ids": ["NCT1"],
            },
            {
                "field_id": "labs_tsh",
                "display_name": "TSH",
                "field": "TSH",
                "description": "Needed by many high-priority trials",
                "priority": "high",
                "affected_trial_ids": ["NCT2"],
            },
        ]
    )
    assert len(gaps) == 1
    assert gaps[0]["priority"] == "high"


def test_missing_info_field_id_generated_and_legacy_shape_validates() -> None:
    item = CompletenessAssessment.model_validate(
        {
            "field": "EGFR mutation status",
            "description": "Needed for biomarker trial matching",
            "affected_trial_ids": ["NCT1", "NCT1"],
            "priority": "high",
        }
    )
    assert item.field_id == "egfr_mutation_status"
    assert item.display_name == "EGFR mutation status"
    assert item.affected_trial_ids == ["NCT1"]


def test_merge_information_gaps_suppresses_not_applicable_and_dedupes() -> None:
    merged = _merge_information_gaps(
        scored_trials=[
            {
                "trial_id": "NCT1",
                "critical_missing_info": ["N/A pregnancy criterion", "EGFR mutation status"],
                "tier": "moderate",
            }
        ],
        missing_info=[
            {
                "field_id": "egfr_mutation_status",
                "display_name": "EGFR mutation status",
                "field": "EGFR mutation status",
                "description": "Needed once",
                "why_needed": "Needed once",
                "evidence_text": "Needed once",
                "priority": "high",
                "affected_trial_ids": ["NCT2"],
            },
            {
                "field_id": "not_applicable_pregnancy",
                "display_name": "Pregnancy status",
                "field": "Pregnancy status",
                "description": "not applicable for male patient",
                "why_needed": "not applicable for male patient",
                "evidence_text": "not applicable for male patient",
                "priority": "low",
                "affected_trial_ids": ["NCT3"],
            },
        ],
    )
    assert len(merged) == 1
    assert merged[0]["field_id"] == "egfr_mutation_status"


def test_exec_summary_context_includes_risk_signals() -> None:
    ctx = _build_exec_summary_context(
        {"age": 54, "sex": "male", "primary_condition": "NSCLC"},
        [
            {
                "tier": "strong",
                "brief_title": "Trial A",
                "trial_id": "NCTA",
                "score": 0.8,
                "hard_exclusion_failures": 1,
                "major_criteria_assessable": False,
                "critical_missing_info": ["EGFR mutation status"],
                "study_type": "Observational",
            }
        ],
    )
    assert ctx["hard_exclusion_count"] == 1
    assert ctx["uncertain_major_criteria_count"] == 1
    assert ctx["critical_missing_count"] == 1
    assert "observational=1" in ctx["trial_type_signal"]
