import pytest
from agents.report_formatter import build_text_report
from subagents.synthesis import nodes
from subagents.synthesis import nodes as synthesis_nodes


def test_exception_label() -> None:
    assert nodes._exception_label(TimeoutError()) == "timeout"
    assert nodes._exception_label(RuntimeError("x")) == "RuntimeError"


@pytest.mark.asyncio
async def test_run_qa_check_success_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ok_run(**_kwargs):
        return {
            "qa_passed": False,
            "qa_issues": [{"code": "X", "severity": "high", "message": "x"}],
        }

    monkeypatch.setattr(nodes.qa_checker, "run_qa_check", ok_run)
    out = await nodes.run_qa_check({})
    assert out["qa_passed"] is False
    assert out["qa_issues"] == [{"code": "X", "severity": "high", "message": "x"}]

    async def bad_run(**_kwargs):
        raise RuntimeError("bad")

    monkeypatch.setattr(nodes.qa_checker, "run_qa_check", bad_run)
    out_bad = await nodes.run_qa_check({})
    assert out_bad["qa_passed"] is False
    assert out_bad["qa_issues"]
    assert out_bad["new_errors"]


def test_route_after_qa_branches() -> None:
    assert nodes.route_after_qa({"qa_passed": True}) == "generate_report"
    assert nodes.route_after_qa({"qa_passed": False, "qa_fix_attempts": 0}) == "attempt_qa_fix"
    assert nodes.route_after_qa({"qa_passed": False, "qa_fix_attempts": 2}) == "flag_re_evaluation"
    assert (
        nodes.route_after_qa(
            {"qa_passed": False, "qa_fix_attempts": 1, "synthesis_needs_re_evaluation": True}
        )
        == "flag_re_evaluation"
    )


@pytest.mark.asyncio
async def test_attempt_fix_and_flag_and_finalize() -> None:
    fixed = await nodes.attempt_qa_fix(
        {
            "qa_issues": [{"code": "ISSUE", "severity": "medium", "message": "issue"}],
            "qa_fix_attempts": 0,
            "trial_scores": [
                {"trial_id": "A", "tier": "weak", "score": 0.9},
                {"trial_id": "B", "tier": "strong", "score": 0.2},
            ],
        }
    )
    assert fixed["qa_fix_attempts"] == 1
    assert fixed["trial_scores"][0]["trial_id"] == "B"
    assert fixed["qa_remediation_actions"][0]["action"] == "rerank_trials"

    flagged = await nodes.flag_re_evaluation(
        {"qa_issues": [{"code": "X", "severity": "high", "message": "x"}]}
    )
    assert flagged["synthesis_needs_re_evaluation"] is True

    finalized = await nodes.finalize_synthesis_output(
        {
            "qa_issues": [{"code": "Q", "severity": "medium", "message": "q"}],
            "qa_passed": False,
            "report_json": {"ok": True},
            "report_text": "t",
            "synthesis_needs_re_evaluation": True,
            "synthesis_retry_retrieval": False,
            "new_decision_entries": ["d"],
            "new_errors": ["e"],
            "qa_fix_attempts": 1,
            "qa_remediation_actions": [
                {"attempt": "1", "action": "rerank_trials", "issue_code": "Q"}
            ],
            "qa_unresolved_issues": [],
        }
    )
    assert finalized["decision_history"] == ["d"]
    assert finalized["errors"] == ["e"]
    assert finalized["qa_remediation"]["attempts"] == 1


def test_build_text_report_formats_structured_and_string_qa_issues() -> None:
    base_report = {
        "generated_at": "2026-04-25T00:00:00+00:00",
        "patient_summary": "age=50",
        "executive_summary": "summary",
        "total_trials_searched": 1,
        "total_trials_evaluated": 1,
        "strong_matches": [],
        "moderate_matches": [],
        "excluded_trial_count": 0,
        "excluded_trials": [],
        "information_gaps": [],
        "methodology_note": "method",
        "search_queries_used": [],
        "decision_history": [],
        "ranked_trials": [],
        "qa_remediation": {"attempts": 0, "actions": [], "unresolved_issues": []},
    }
    structured = dict(base_report)
    structured["qa_issues"] = [{"code": "A", "severity": "critical", "message": "structured issue"}]
    text_structured = build_text_report(structured)
    assert "[CRITICAL] A: structured issue" in text_structured

    legacy = dict(base_report)
    legacy["qa_issues"] = ["legacy issue"]
    text_legacy = build_text_report(legacy)
    assert "legacy issue" in text_legacy


def test_build_text_report_suppresses_not_applicable_gaps() -> None:
    report = {
        "generated_at": "2026-04-25T00:00:00+00:00",
        "patient_summary": "age=50",
        "executive_summary": "summary",
        "total_trials_searched": 1,
        "total_trials_evaluated": 1,
        "strong_matches": [],
        "moderate_matches": [],
        "excluded_trial_count": 0,
        "excluded_trials": [],
        "information_gaps": [
            {
                "field_id": "not_applicable_pregnancy",
                "field": "Pregnancy status",
                "description": "Not applicable for male patient",
                "priority": "high",
            },
            {
                "field_id": "egfr_mutation_status",
                "field": "EGFR mutation status",
                "description": "Needed for selection",
                "priority": "high",
            },
        ],
        "methodology_note": "method",
        "search_queries_used": [],
        "decision_history": [],
        "ranked_trials": [],
        "qa_issues": [],
        "qa_remediation": {"attempts": 0, "actions": [], "unresolved_issues": []},
    }
    text = build_text_report(report)
    assert "EGFR mutation status" in text
    assert "Pregnancy status" not in text


@pytest.mark.asyncio
async def test_attempt_fix_recompute_from_verdicts() -> None:
    out = await nodes.attempt_qa_fix(
        {
            "qa_issues": [
                {
                    "code": "HARD_EXCLUSION_RANKING_CONFLICT",
                    "severity": "critical",
                    "message": "x",
                }
            ],
            "qa_fix_attempts": 0,
            "trial_scores": [{"trial_id": "T1", "tier": "moderate", "score": 0.91}],
            "eligibility_verdicts": {"T1": {"hard_exclusion_failures": 1}},
        }
    )
    assert out["trial_scores"][0]["tier"] == "disqualified"
    assert out["trial_scores"][0]["score"] <= 0.2


@pytest.mark.asyncio
async def test_attempt_fix_escalates_retrieval_issue() -> None:
    out = await nodes.attempt_qa_fix(
        {
            "qa_issues": [
                {
                    "code": "RETRIEVAL_FAILED_EMPTY_RESULT",
                    "severity": "critical",
                    "message": "retrieval empty",
                }
            ],
            "qa_fix_attempts": 0,
            "trial_scores": [],
        }
    )
    assert out["synthesis_needs_re_evaluation"] is True
    assert out["synthesis_retry_retrieval"] is True


def test_collect_synthesis_inputs_merges_decisions() -> None:
    out = nodes._collect_synthesis_inputs(
        {
            "decision_history": ["a"],
            "new_decision_entries": ["b"],
            "patient_profile": {},
            "trial_scores": [],
            "missing_info_recommendations": [],
            "eligibility_verdicts": {},
            "trials_raw": [],
            "search_queries": [],
            "qa_issues": [],
        }
    )
    assert out["decision_history"] == ["a", "b"]


@pytest.mark.asyncio
async def test_generate_report_node_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_build_report(**_kwargs):
        return {"summary": "ok"}

    monkeypatch.setattr(nodes.report_generator, "build_report", fake_build_report)
    monkeypatch.setattr(nodes.report_generator, "build_text_report", lambda _r: "text")
    ok = await nodes.generate_report_node({})
    assert ok["report_json"]["summary"] == "ok"

    async def bad_build_report(**_kwargs):
        raise ValueError("nope")

    monkeypatch.setattr(nodes.report_generator, "build_report", bad_build_report)
    bad = await nodes.generate_report_node({})
    assert "error" in bad["report_json"]
    assert "failed" in bad["report_text"].lower()


@pytest.mark.asyncio
async def test_synthesis_tier_ordering_and_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_build_report(**kwargs):
        scored = kwargs.get("scored_trials", [])
        filtered = [t for t in scored if t.get("tier") in {"strong", "moderate"}]
        ordered = sorted(
            filtered, key=lambda t: {"strong": 2, "moderate": 1}[t["tier"]], reverse=True
        )
        return {"ordered_titles": [t["brief_title"] for t in ordered]}

    monkeypatch.setattr(synthesis_nodes.report_generator, "build_report", fake_build_report)
    monkeypatch.setattr(
        synthesis_nodes.report_generator,
        "build_text_report",
        lambda report: "\n".join(report.get("ordered_titles", [])),
    )

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
