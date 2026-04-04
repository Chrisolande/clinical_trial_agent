from typing import Any
from unittest.mock import AsyncMock

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
        async def __aenter__(self) -> "DummyMemory":
            return self

        async def write_pipeline_audit(
            self,
            patient_profile: dict[str, Any],
            run_id: str,
            outcome_tier_counts: dict[str, int],
            model_version: str,
            consent_flag: bool,
        ) -> None:
            _ = (patient_profile, run_id, outcome_tier_counts, model_version, consent_flag)
            return None

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            _ = (exc_type, exc, tb)
            return None

        async def lookup(self, patient_profile: dict[str, Any]) -> dict[str, Any] | None:
            _ = patient_profile
            return None

        async def store(self, patient_profile: dict[str, Any], result: dict[str, Any]) -> None:
            _ = (patient_profile, result)
            return None

        async def list_feedback(self, profile_hash: str) -> list[dict[str, Any]]:
            _ = profile_hash
            return []

    monkeypatch.setattr("agents.supervisor.EpisodicMemory", DummyMemory)
    monkeypatch.setattr(orchestrator, "run_retrieval", fake_run_retrieval)
    monkeypatch.setattr(orchestrator, "run_eligibility", fake_run_eligibility)
    monkeypatch.setattr(orchestrator, "run_synthesis", fake_run_synthesis)

    result = await orchestrator.ainvoke({"age": 42}, thread_id="test-thread")
    assert result.get("report_text") == "done"
    assert order == ["retrieval", "eligibility", "synthesis"]


@pytest.mark.asyncio
async def test_supervisor_uses_single_memory_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import get_settings

    monkeypatch.setenv("SUPERVISOR_USE_REACT", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(SupervisorOrchestrator, "_get_llm", staticmethod(lambda: object()))

    class DummyReactAgent:
        async def ainvoke(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {"report_json": {"ok": True}, "report_text": "done"}

    monkeypatch.setattr("agents.supervisor.create_agent", lambda **_: DummyReactAgent())
    orchestrator = SupervisorOrchestrator()

    class DummyMemory:
        aenter = AsyncMock()

        async def write_pipeline_audit(
            self,
            patient_profile: dict[str, Any],
            run_id: str,
            outcome_tier_counts: dict[str, int],
            model_version: str,
            consent_flag: bool,
        ) -> None:
            _ = (patient_profile, run_id, outcome_tier_counts, model_version, consent_flag)
            return None

        async def __aenter__(self) -> "DummyMemory":
            await DummyMemory.aenter()
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            _ = (exc_type, exc, tb)
            return None

        async def lookup(self, patient_profile: dict[str, Any]) -> dict[str, Any] | None:
            _ = patient_profile
            return None

        async def store(self, patient_profile: dict[str, Any], result: dict[str, Any]) -> None:
            _ = (patient_profile, result)
            return None

        async def list_feedback(self, profile_hash: str) -> list[dict[str, Any]]:
            _ = profile_hash
            return []

    monkeypatch.setattr("agents.supervisor.EpisodicMemory", DummyMemory)
    result = await orchestrator.ainvoke({"age": 42}, thread_id="thread-one")
    get_settings.cache_clear()
    assert result["report_text"] == "done"
    assert DummyMemory.aenter.call_count == 1


def test_feedback_adjustment_changes_ranking() -> None:
    from agents.supervisor import _apply_feedback_adjustments

    scored = [
        {"trial_id": "NCT1", "tier": "moderate", "score": 0.60, "rank": 1},
        {"trial_id": "NCT2", "tier": "moderate", "score": 0.59, "rank": 2},
    ]
    feedback = [{"nct_id": "NCT2", "verdict": "confirmed"}]

    adjusted = _apply_feedback_adjustments(scored, feedback)
    assert adjusted[0]["trial_id"] == "NCT2"
