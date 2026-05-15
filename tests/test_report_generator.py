import re

import pytest
from agents import report_generator
from agents.report_formatter import build_text_report
from agents.report_generator import _merge_information_gaps
from models.report import ReportPlan


def _plan_with_cards() -> ReportPlan:
    return ReportPlan.model_validate(
        {
            "patient_summary": "64 year old female with metastatic NSCLC",
            "executive_summary": "Trial A is a stronger option than Trial B.",
            "bottom_line": "Prioritize Trial A and complete confirmatory workup for Trial B.",
            "strong_matches": [
                {
                    "nct_id": "NCTA",
                    "title": "Trial A",
                    "tier": "strong",
                    "score": 0.86,
                    "phase": "PHASE2",
                    "status": "RECRUITING",
                    "recommendation": "Proceed with referral screening.",
                    "why_it_matches": ["Major criteria met."],
                    "main_blockers": [],
                    "key_uncertainties": [],
                    "next_action": "Schedule screening call.",
                    "evidence_summary": "Major criteria are supported with no known hard exclusions.",
                },
            ],
            "moderate_matches": [
                {
                    "nct_id": "NCTB",
                    "title": "Trial B",
                    "tier": "moderate",
                    "score": 0.64,
                    "phase": "PHASE1",
                    "status": "RECRUITING",
                    "recommendation": "Candidate if biomarker confirmation supports eligibility.",
                    "why_it_matches": ["Plausible fit based on disease/stage profile."],
                    "main_blockers": ["Pending EGFR status"],
                    "key_uncertainties": ["Full biomarker panel still pending"],
                    "next_action": "Order EGFR testing now.",
                    "evidence_summary": "Potential fit but important eligibility confirmations remain.",
                }
            ],
            "information_gaps": [
                {
                    "item": "EGFR mutation status",
                    "priority": "high",
                    "reason": "Needed for biomarker-gated enrollment decisions.",
                    "affects_trials": ["NCTB"],
                    "action": "Order EGFR mutation testing.",
                    "applicable_to_patient": True,
                }
            ],
            "recommended_actions": [
                {
                    "item": "EGFR mutation status",
                    "priority": "high",
                    "reason": "Needed for biomarker-gated enrollment decisions.",
                    "affects_trials": ["NCTB"],
                    "action": "Order EGFR mutation testing.",
                    "applicable_to_patient": True,
                }
            ],
            "excluded_summary": "Weak/disqualified trials summarized separately.",
            "limitations": ["Eligibility remains limited by missing biomarker data."],
        }
    )


def _base_report_json(debug: bool = False) -> dict:
    return {
        "generated_at": "2026-04-25T00:00:00+00:00",
        "report_plan": _plan_with_cards().model_dump(),
        "patient_summary": "legacy patient",
        "executive_summary": "legacy summary",
        "total_trials_searched": 4,
        "total_trials_evaluated": 2,
        "strong_matches": [],
        "moderate_matches": [],
        "excluded_trial_count": 1,
        "excluded_trials": [
            {"trial_id": "NCTX", "brief_title": "Excluded", "tier": "weak", "score": 0.2}
        ],
        "information_gaps": [],
        "ranked_trials": [],
        "methodology_note": "This uses llm summarization; parser fallback if tool failed.",
        "qa_issues_internal": [{"code": "X", "severity": "high", "message": "internal qa issue"}],
        "qa_remediation_internal": {
            "attempts": 1,
            "actions": [{"action": "rerank_trials"}],
            "unresolved_issues": [],
        },
        "debug": debug,
    }


def test_normal_report_omits_forbidden_leakage_and_qa_sections() -> None:
    report = _base_report_json(debug=False)
    report["report_plan"]["executive_summary"] = "Parser fallback used by judge model."
    text = build_text_report(report)
    assert "QA ISSUES" not in text
    assert "QA REMEDIATION" not in text
    for forbidden in (
        "judge model",
        "structured verdict",
        "llm",
        "parser",
        "fallback",
        "tool failed",
        "qa issue",
        "qa check",
        "model returned none",
    ):
        pattern = re.compile(rf"\b{r'\s+'.join(re.escape(part) for part in forbidden.split())}\b")
        assert not pattern.search(text.lower())


def test_debug_report_includes_qa_internal_only_when_debug_true() -> None:
    normal = build_text_report(_base_report_json(debug=False))
    debug = build_text_report(_base_report_json(debug=True))
    assert "QA ISSUES" not in normal
    assert "QA REMEDIATION" not in normal
    assert "QA ISSUES" in debug
    assert "[HIGH] X: internal qa issue" in debug
    assert "QA REMEDIATION" in debug


def test_not_applicable_gaps_suppressed_and_duplicates_merged() -> None:
    merged = _merge_information_gaps(
        scored_trials=[
            {
                "trial_id": "NCT1",
                "tier": "moderate",
                "critical_missing_info": ["N/A pregnancy criterion", "EGFR mutation status"],
            }
        ],
        missing_info=[
            {
                "field_id": "egfr_mutation_status",
                "field": "EGFR mutation status",
                "description": "Needed.",
                "priority": "medium",
                "affected_trial_ids": ["NCT2"],
            },
            {
                "field_id": "egfr_mutation_status",
                "field": "EGFR mutation status",
                "description": "Needed for biomarker-gated trial decisions across sites.",
                "priority": "high",
                "affected_trial_ids": ["NCT3"],
            },
            {
                "field_id": "not_applicable_pregnancy",
                "field": "Pregnancy status",
                "description": "Not applicable for male patient",
                "priority": "high",
                "affected_trial_ids": ["NCT4"],
            },
        ],
    )
    assert len(merged) == 1
    assert merged[0]["field_id"] == "egfr_mutation_status"
    assert merged[0]["priority"] == "high"
    assert merged[0]["affected_trial_ids"] == ["NCT1", "NCT2", "NCT3"]


@pytest.mark.asyncio
async def test_build_report_payload_and_plan_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_report_plan(**_kwargs):
        return _plan_with_cards()

    monkeypatch.setattr(report_generator, "generate_report_plan", fake_generate_report_plan)
    report = await report_generator.build_report(
        patient_profile={"age": 64, "sex": "female", "primary_condition": "NSCLC"},
        scored_trials=[
            {
                "trial_id": "NCTB",
                "brief_title": "Trial B",
                "tier": "moderate",
                "score": 0.64,
                "critical_missing_info": [],
            },
            {
                "trial_id": "NCTA",
                "brief_title": "Trial A",
                "tier": "strong",
                "score": 0.86,
                "critical_missing_info": [],
            },
            {
                "trial_id": "NCTX",
                "brief_title": "Trial X",
                "tier": "weak",
                "score": 0.2,
                "critical_missing_info": [],
            },
        ],
        missing_info=[],
        eligibility_verdicts={
            "NCTA": {"verdicts": []},
            "NCTB": {"verdicts": []},
            "NCTX": {"verdicts": []},
        },
        trials_raw=[{}, {}, {}],
        search_queries=["nsclc trial"],
        decision_history=["initial synthesis"],
        qa_issues=[{"code": "N_A_LEAKAGE_IN_GAPS", "severity": "medium", "message": "sanitize"}],
        qa_remediation={
            "attempts": 1,
            "actions": [{"action": "sanitize_information_gaps"}],
            "unresolved_issues": [],
        },
    )

    assert "report_plan" in report
    assert report["report_plan"]["patient_summary"] == "64 year old female with metastatic NSCLC"
    assert "qa_issues" not in report
    assert "qa_issues_internal" in report
    assert report["debug"] is False

    text = build_text_report(report)
    for section in (
        "PATIENT SNAPSHOT",
        "BOTTOM LINE",
        "TRIAL RANKING TABLE",
        "STRONG MATCHES",
        "MODERATE MATCHES",
        "CRITICAL INFORMATION GAPS",
        "RECOMMENDED CLINICAL NEXT ACTIONS",
        "TRIALS NOT PRIORITIZED",
        "METHODOLOGY AND LIMITATIONS",
    ):
        assert section in text
    assert "64 year old female with metastatic NSCLC" in text
    assert "Trial A is a stronger option than Trial B" in text


def test_n_a_leakage_in_gaps_triggers_sanitization_and_not_printed() -> None:
    report = _base_report_json(debug=False)
    report["report_plan"]["information_gaps"] = [
        {
            "item": "Pregnancy status",
            "priority": "high",
            "reason": "Not applicable for male patient",
            "affects_trials": ["NCTX"],
            "action": "No action needed.",
            "applicable_to_patient": False,
        },
        {
            "item": "ECOG performance status",
            "priority": "high",
            "reason": "Needed to confirm inclusion threshold.",
            "affects_trials": ["NCTB"],
            "action": "Confirm ECOG at baseline visit.",
            "applicable_to_patient": True,
        },
    ]
    report["qa_issues_internal"] = [
        {"code": "N_A_LEAKAGE_IN_GAPS", "severity": "medium", "message": "N/A leakage detected"}
    ]
    text = build_text_report(report)
    assert "Pregnancy status" not in text
    assert "not applicable" not in text.lower()
    assert "ECOG performance status" in text
