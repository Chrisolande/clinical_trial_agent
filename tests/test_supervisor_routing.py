from typing import Any

import pytest
from agents.supervisor import SupervisorOrchestrator


def test_route_after_eligibility_respects_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from clinical_trial_agent.langgraph_app import route_after_eligibility

    monkeypatch.setattr(
        "clinical_trial_agent.langgraph_app.get_settings",
        lambda: type("S", (), {"one_pass_mode": False, "max_retry_attempts": 2})(),
    )
    assert route_after_eligibility({"eligibility_result": {"retrieval_needs_broadening": True}}) == "retry_retrieval"
    assert route_after_eligibility({"retry_count": 2, "eligibility_result": {"retrieval_needs_broadening": True}}) == "run_synthesis"


@pytest.mark.asyncio
async def test_supervisor_retries_after_synthesis_re_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = SupervisorOrchestrator()
    eligibility_calls: list[int] = []
    synthesis_calls: list[int] = []

    monkeypatch.setattr(
        "agents.supervisor.get_settings",
        lambda: type("S", (), {"one_pass_mode": False, "max_retry_attempts": 1, "max_trials_for_eligibility": 5, "criteria_text_max_chars": 8000})(),
    )

    async def fake_retrieval(**_kwargs: Any) -> dict[str, Any]:
        return {"trials_raw": [], "trials_deduplicated": [{"nct_id": "N1"}], "search_queries": []}

    async def fake_eligibility(**kwargs: Any) -> dict[str, Any]:
        eligibility_calls.append(int(kwargs.get("attempt", 0)))
        return {
            "trial_scores": [{"trial_id": "T1", "tier": "moderate", "score": 0.6}],
            "eligibility_verdicts": {},
            "missing_info_recommendations": [],
            "decision_history": [],
            "trials_with_criteria": [],
            "retrieval_needs_broadening": False,
        }

    async def fake_synthesis(**_kwargs: Any) -> dict[str, Any]:
        call_index = len(synthesis_calls)
        synthesis_calls.append(call_index)
        if call_index == 0:
            return {
                "report_json": {"ok": False},
                "report_text": "needs re-eval",
                "synthesis_needs_re_evaluation": True,
                "synthesis_retry_retrieval": False,
            }
        return {
            "report_json": {"ok": True},
            "report_text": "done",
            "synthesis_needs_re_evaluation": False,
            "synthesis_retry_retrieval": False,
        }

    class DummyMemory:
        async def list_feedback(self, *_: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(orchestrator, "run_retrieval", fake_retrieval)
    monkeypatch.setattr(orchestrator, "run_eligibility", fake_eligibility)
    monkeypatch.setattr(orchestrator, "run_synthesis", fake_synthesis)

    result = await orchestrator._run_tools_pipeline({"age": 40}, thread_id="qa-thread", memory=DummyMemory())
    assert result["report_text"] == "done"
    assert synthesis_calls == [0, 1]
    assert eligibility_calls == [0, 1]


def test_route_after_synthesis_respects_re_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    from clinical_trial_agent.langgraph_app import route_after_synthesis

    monkeypatch.setattr(
        "clinical_trial_agent.langgraph_app.get_settings",
        lambda: type("S", (), {"one_pass_mode": False, "max_retry_attempts": 1})(),
    )
    assert route_after_synthesis({"retry_count": 0, "synthesis_needs_re_evaluation": True, "synthesis_retry_retrieval": True}) == "retry_retrieval"
    assert route_after_synthesis({"retry_count": 0, "synthesis_needs_re_evaluation": True, "synthesis_retry_retrieval": False}) == "retry_eligibility"
    assert route_after_synthesis({"retry_count": 1, "synthesis_needs_re_evaluation": True, "synthesis_retry_retrieval": False}) == "end"
