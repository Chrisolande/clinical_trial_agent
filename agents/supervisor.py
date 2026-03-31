from __future__ import annotations

import contextlib
from typing import Any

from config import settings
from langgraph.prebuilt import create_react_agent
from loguru import logger
from memory import EpisodicMemory, get_checkpointer
from prompts.supervisor import build_supervisor_prompt
from subagents.eligibility.graph import compile_eligibility_graph
from subagents.retrieval.graph import compiled_retrieval_graph
from subagents.synthesis.graph import compiled_synthesis_graph


class SupervisorOrchestrator:
    def __init__(self) -> None:
        self._react_agent = create_react_agent(
            model=self._get_llm(),
            tools=[self.run_retrieval, self.run_eligibility, self.run_synthesis],
            name="clinical_supervisor",
            prompt=build_supervisor_prompt(),
        )

    @staticmethod
    def _get_llm() -> Any:
        from config import get_llm

        return get_llm()

    async def run_retrieval(
        self,
        patient_profile: dict[str, Any],
        normalized_terms: dict[str, Any] | None = None,
        retry_count: int = 0,
    ) -> dict[str, Any]:
        retrieval_input = {
            "normalized_terms": normalized_terms or {},
            "patient_profile": patient_profile,
            "retry_count": retry_count,
            "trials_raw": [],
        }
        return await compiled_retrieval_graph.ainvoke(retrieval_input)

    async def run_eligibility(
        self,
        patient_profile: dict[str, Any],
        trials_deduplicated: list[dict[str, Any]],
        eligibility_verdicts: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        trimmed_trials = []
        for trial in trials_deduplicated:
            criteria_text = str(trial.get("eligibility_criteria_raw", ""))
            trimmed_trials.append(
                {
                    **trial,
                    "eligibility_criteria_raw": criteria_text[: settings.criteria_text_max_chars],
                }
            )
        eligibility_graph = compile_eligibility_graph(use_postgres_checkpointer=True)
        eligibility_input = {
            "patient_profile": patient_profile,
            "trials_deduplicated": trimmed_trials,
            "eligibility_verdicts": eligibility_verdicts,
        }
        return await eligibility_graph.ainvoke(eligibility_input)

    async def run_synthesis(
        self,
        patient_profile: dict[str, Any],
        trial_scores: list[dict[str, Any]],
        eligibility_verdicts: dict[str, dict[str, Any]] | None,
        missing_info_recommendations: list[dict[str, Any]] | None,
        trials_raw: list[dict[str, Any]],
        search_queries: list[str],
        decision_history: list[str],
        trials_with_criteria: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        synthesis_input = {
            "patient_profile": patient_profile,
            "trial_scores": trial_scores,
            "eligibility_verdicts": eligibility_verdicts,
            "missing_info_recommendations": missing_info_recommendations,
            "trials_raw": trials_raw,
            "search_queries": search_queries,
            "decision_history": decision_history,
            "trials_with_criteria": trials_with_criteria,
        }
        return await compiled_synthesis_graph.ainvoke(synthesis_input)

    async def ainvoke(
        self,
        patient_profile: dict[str, Any],
        *,
        thread_id: str,
        recursion_limit: int = 25,
    ) -> dict[str, Any]:
        memory = EpisodicMemory()
        await memory.init()
        try:
            cached = await memory.lookup(patient_profile)
            if cached:
                logger.info("Supervisor served result from episodic memory")
                return cached
        finally:
            await memory.close()

        agent_input = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Match this patient to trials by calling tools in order: "
                        "run_retrieval -> run_eligibility -> run_synthesis.\n"
                        f"Patient profile: {patient_profile}"
                    ),
                }
            ]
        }
        config = {"configurable": {"thread_id": thread_id, "recursion_limit": recursion_limit}}

        result = await self._react_agent.ainvoke(agent_input, config=config)
        normalized = self._extract_final_result(result)
        if "report_json" not in normalized and "report_text" in normalized:
            normalized = await self._run_tools_pipeline(patient_profile)

        memory = EpisodicMemory()
        await memory.init()
        try:
            await memory.store(patient_profile, normalized)
        finally:
            await memory.close()

        return normalized

    async def _run_tools_pipeline(self, patient_profile: dict[str, Any]) -> dict[str, Any]:
        retrieval = await self.run_retrieval(patient_profile=patient_profile)
        eligibility = await self.run_eligibility(
            patient_profile=patient_profile,
            trials_deduplicated=list(retrieval.get("trials_deduplicated", [])),
            eligibility_verdicts=None,
        )
        synthesis = await self.run_synthesis(
            patient_profile=patient_profile,
            trial_scores=list(eligibility.get("trial_scores", [])),
            eligibility_verdicts=eligibility.get("eligibility_verdicts"),
            missing_info_recommendations=eligibility.get("missing_info_recommendations"),
            trials_raw=list(retrieval.get("trials_raw", [])),
            search_queries=list(retrieval.get("search_queries", [])),
            decision_history=list(eligibility.get("decision_history", [])),
            trials_with_criteria=eligibility.get("trials_with_criteria"),
        )
        return synthesis

    @staticmethod
    def _extract_final_result(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            if isinstance(result.get("report_json"), dict):
                return result
            messages = result.get("messages")
            if isinstance(messages, list) and messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if isinstance(content, str):
                    return {"report_text": content}
        return {"report_text": str(result)}


@contextlib.asynccontextmanager
async def compile_supervisor_graph() -> Any:
    checkpointer = get_checkpointer(settings.database_uri)
    orchestrator = SupervisorOrchestrator()
    if checkpointer is not None and hasattr(checkpointer, "__aenter__"):
        async with checkpointer:
            yield orchestrator
    else:
        yield orchestrator
