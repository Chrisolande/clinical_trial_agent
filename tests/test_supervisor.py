from __future__ import annotations

from typing import Any

import pytest
from agents.supervisor import SupervisorOrchestrator


@pytest.mark.asyncio
async def test_supervisor_calls_subagents_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    class DummyReactAgent:
        async def ainvoke(
            self, agent_input: dict[str, Any], config: dict[str, Any]
        ) -> dict[str, Any]:
            assert "messages" in agent_input
            assert "configurable" in config
            return {"report_text": "fallback text only"}

    monkeypatch.setattr(
        SupervisorOrchestrator,
        "_get_llm",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        "agents.supervisor.create_agent",
        lambda **_: DummyReactAgent(),
    )

    orchestrator = SupervisorOrchestrator()

    async def fake_run_retrieval(
        patient_profile: dict[str, Any],
        normalized_terms: dict[str, Any] | None = None,
        retry_count: int = 0,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        _ = (patient_profile, normalized_terms, retry_count, thread_id)
        order.append("retrieval")
        return {"trials_raw": [], "trials_deduplicated": [], "search_queries": []}

    async def fake_run_eligibility(
        patient_profile: dict[str, Any],
        trials_deduplicated: list[dict[str, Any]],
        eligibility_verdicts: dict[str, dict[str, Any]] | None = None,
        thread_id: str | None = None,
        attempt: int = 0,
    ) -> dict[str, Any]:
        _ = (
            patient_profile,
            trials_deduplicated,
            eligibility_verdicts,
            thread_id,
            attempt,
        )
        order.append("eligibility")
        return {
            "trial_scores": [{"trial_id": "T1", "tier": "moderate", "score": 0.6}],
            "eligibility_verdicts": {},
            "missing_info_recommendations": [],
            "decision_history": [],
            "trials_with_criteria": [],
        }

    async def fake_run_synthesis(
        patient_profile: dict[str, Any],
        trial_scores: list[dict[str, Any]],
        eligibility_verdicts: dict[str, dict[str, Any]] | None,
        missing_info_recommendations: list[dict[str, Any]] | None,
        trials_raw: list[dict[str, Any]],
        search_queries: list[str],
        decision_history: list[str],
        trials_with_criteria: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        _ = (
            patient_profile,
            trial_scores,
            eligibility_verdicts,
            missing_info_recommendations,
            trials_raw,
            search_queries,
            decision_history,
            trials_with_criteria,
            thread_id,
        )
        order.append("synthesis")
        return {"report_json": {"ok": True}, "report_text": "done"}

    class DummyMemory:
        async def init(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def lookup(self, patient_profile: dict[str, Any]) -> dict[str, Any] | None:
            _ = patient_profile
            return None

        async def store(self, patient_profile: dict[str, Any], result: dict[str, Any]) -> None:
            _ = (patient_profile, result)
            return None

    monkeypatch.setattr("agents.supervisor.EpisodicMemory", DummyMemory)
    monkeypatch.setattr(orchestrator, "run_retrieval", fake_run_retrieval)
    monkeypatch.setattr(orchestrator, "run_eligibility", fake_run_eligibility)
    monkeypatch.setattr(orchestrator, "run_synthesis", fake_run_synthesis)

    result = await orchestrator.ainvoke({"age": 42}, thread_id="test-thread")
    assert result.get("report_text") == "done"
    assert order == ["retrieval", "eligibility", "synthesis"]
