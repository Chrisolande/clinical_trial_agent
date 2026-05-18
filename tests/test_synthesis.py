import pytest
from agents import report_synthesizer
from models.report import ReportPlan
from subagents.synthesis import nodes

from clinical_trial_agent.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_report_plan_validates_structured_output() -> None:
    plan = ReportPlan.model_validate(
        {
            "patient_summary": "64 year old female patient with NSCLC",
            "executive_summary": "Trial A is stronger than Trial B due to fewer unresolved exclusions.",
            "bottom_line": "Prioritize Trial A and confirm biomarker status for Trial B.",
            "strong_matches": [
                {
                    "nct_id": "NCT0001",
                    "title": "Trial A",
                    "tier": "strong",
                    "score": 0.81,
                    "phase": "PHASE2",
                    "status": "RECRUITING",
                    "recommendation": "Proceed to screening.",
                    "why_it_matches": ["Major inclusion criteria are met."],
                    "main_blockers": [],
                    "key_uncertainties": [],
                    "next_action": "Schedule screening visit.",
                    "evidence_summary": "No hard exclusion identified in provided data.",
                }
            ],
            "moderate_matches": [],
            "information_gaps": [
                {
                    "item": "EGFR mutation status",
                    "priority": "high",
                    "reason": "Needed to resolve biomarker eligibility uncertainty.",
                    "affects_trials": ["NCT0002"],
                    "action": "Order EGFR testing.",
                    "applicable_to_patient": True,
                }
            ],
            "recommended_actions": [
                {
                    "item": "EGFR mutation status",
                    "priority": "high",
                    "reason": "Needed to resolve biomarker eligibility uncertainty.",
                    "affects_trials": ["NCT0002"],
                    "action": "Order EGFR testing.",
                    "applicable_to_patient": True,
                }
            ],
            "excluded_summary": "Weak/disqualified trials summarized separately.",
            "limitations": [],
        }
    )
    assert plan.strong_matches[0].tier == "strong"
    assert plan.information_gaps[0].priority == "high"


@pytest.mark.asyncio
async def test_generate_report_plan_uses_reportplan_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    structured_model: dict[str, object] = {}

    class DummyPrompt:
        def __or__(self, other):
            return other

    class DummyChain:
        async def ainvoke(self, context, config=None):
            captured["context"] = context
            captured["config"] = config
            return ReportPlan.model_validate(
                {
                    "patient_summary": "summary",
                    "executive_summary": "comparative summary",
                    "bottom_line": "summary bottom line",
                    "strong_matches": [],
                    "moderate_matches": [],
                    "information_gaps": [],
                    "recommended_actions": [],
                    "excluded_summary": "none",
                    "limitations": [],
                }
            )

    class LLMStub:
        def with_structured_output(self, model):
            structured_model["model"] = model
            return DummyChain()

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(report_synthesizer, "get_llm", lambda **_kwargs: LLMStub())
    monkeypatch.setattr(
        report_synthesizer.ChatPromptTemplate, "from_template", lambda _template: DummyPrompt()
    )

    out = await report_synthesizer.generate_report_plan(
        patient_profile={"age": 60, "sex": "male", "primary_condition": "NSCLC"},
        scored_trials=[],
        eligibility_verdicts={},
        missing_info=[],
        qa_issues=[],
    )

    assert isinstance(out, ReportPlan)
    assert structured_model["model"] is ReportPlan
    assert captured["context"]


def _report_plan() -> ReportPlan:
    return ReportPlan.model_validate(
        {
            "patient_summary": "summary",
            "executive_summary": "comparative summary",
            "bottom_line": "summary bottom line",
            "strong_matches": [],
            "moderate_matches": [],
            "information_gaps": [],
            "recommended_actions": [],
            "excluded_summary": "none",
            "limitations": [],
        }
    )


class _DummyPrompt:
    def __or__(self, other):
        return other


class _CapturingChain:
    def __init__(self, captured: dict[str, object]) -> None:
        self._captured = captured

    async def ainvoke(self, context, config=None):
        self._captured["context"] = context
        self._captured["config"] = config
        return _report_plan()


class _CapturingLLM:
    def __init__(self, captured: dict[str, object]) -> None:
        self._captured = captured

    def with_structured_output(self, model):
        self._captured["model"] = model
        return _CapturingChain(self._captured)


async def _run_synthesis_with_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_get_llm(**kwargs):
        captured["get_llm_kwargs"] = kwargs
        return _CapturingLLM(captured)

    monkeypatch.setattr(report_synthesizer, "get_llm", fake_get_llm)
    monkeypatch.setattr(
        report_synthesizer.ChatPromptTemplate, "from_template", lambda _template: _DummyPrompt()
    )

    await report_synthesizer.generate_report_plan(
        patient_profile={
            "name": "Jane Patient",
            "age": 64,
            "sex": "female",
            "primary_condition": "NSCLC",
            "zip_code": "12345",
        },
        scored_trials=[],
        eligibility_verdicts={},
        missing_info=[],
        qa_issues=[],
    )
    return captured


@pytest.mark.asyncio
async def test_synthesis_privacy_mode_blocked_rejects_external_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_PRIVACY_MODE", "blocked")
    monkeypatch.setattr(
        report_synthesizer,
        "get_llm",
        lambda **_kwargs: pytest.fail("blocked mode must not construct an LLM"),
    )

    with pytest.raises(RuntimeError, match="LLM_PRIVACY_MODE=blocked"):
        await report_synthesizer.generate_report_plan(
            patient_profile={"age": 64, "primary_condition": "NSCLC"},
            scored_trials=[],
            eligibility_verdicts={},
            missing_info=[],
            qa_issues=[],
        )


@pytest.mark.asyncio
async def test_synthesis_privacy_mode_deidentified_sends_deidentified_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_PRIVACY_MODE", "deidentified")

    captured = await _run_synthesis_with_capture(monkeypatch)

    context = captured["context"]
    assert isinstance(context, dict)
    patient_profile = str(context["patient_profile"])
    assert "age_range: 60-69" in patient_profile
    assert "Jane Patient" not in patient_profile
    assert "zip_code" not in patient_profile
    assert captured["get_llm_kwargs"] == {
        "contains_phi": False,
        "node_name": "report_synthesis",
    }


@pytest.mark.asyncio
async def test_synthesis_privacy_mode_full_consent_allows_external_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_PRIVACY_MODE", "full_consent")
    monkeypatch.setenv("CLINICAL_DATA_EXTERNAL_LLM_CONSENT", "true")

    captured = await _run_synthesis_with_capture(monkeypatch)

    context = captured["context"]
    assert isinstance(context, dict)
    patient_profile = str(context["patient_profile"])
    assert "primary_condition: NSCLC" in patient_profile
    assert "Jane Patient" not in patient_profile


@pytest.mark.asyncio
async def test_synthesis_privacy_mode_local_only_rejects_external_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_PRIVACY_MODE", "local_only")

    with pytest.raises(RuntimeError, match="LLM_PRIVACY_MODE=local_only"):
        await _run_synthesis_with_capture(monkeypatch)


def test_exception_label() -> None:
    assert nodes._exception_label(TimeoutError()) == "timeout"
    assert nodes._exception_label(RuntimeError("x")) == "RuntimeError"


@pytest.mark.asyncio
async def test_run_qa_check_success_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
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
