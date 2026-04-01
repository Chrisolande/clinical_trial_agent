from __future__ import annotations

import asyncio
import contextlib
import inspect
from dataclasses import dataclass
from typing import Any, cast

from config import settings
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from loguru import logger
from memory import EpisodicMemory, get_checkpointer
from prompts.supervisor import build_supervisor_prompt
from subagents.eligibility.graph import compile_eligibility_graph
from subagents.retrieval.graph import compiled_retrieval_graph
from subagents.synthesis.graph import compiled_synthesis_graph

PatientSummary = dict[str, Any]
TrialSummary = dict[str, Any]
ScoredTrialSummary = dict[str, Any]


@dataclass
class SupervisorRunState:
    retrieval_result: dict[str, Any]
    scored_trials: list[ScoredTrialSummary]
    final_result: dict[str, Any]


class SupervisorOrchestrator:
    def __init__(self, checkpointer: Any | None = None) -> None:
        self._checkpointer = checkpointer
        self._react_agent = create_agent(
            model=self._get_llm(),
            tools=[self.run_retrieval, self.run_eligibility, self.run_synthesis],
            name="clinical_supervisor",
            system_prompt=build_supervisor_prompt(),
            checkpointer=checkpointer,
        )

    @staticmethod
    def _get_llm() -> Any:
        from config import get_llm

        return get_llm()

    async def run_retrieval(
        self,
        patient_profile: PatientSummary,
        normalized_terms: dict[str, Any] | None = None,
        retry_count: int = 0,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run retrieval to fetch relevant trials for the given patient.

        Args:
            patient_profile: A dictionary containing the patient's summary.
            normalized_terms: A dictionary containing the normalized search terms.
            retry_count: An integer indicating the number of times the retrieval should be retried.
            thread_id: A string indicating the thread the retrieval is running on.

        Returns:
            A dictionary containing the results of the retrieval.
        """
        retrieval_input = {
            "normalized_terms": normalized_terms or {},
            "patient_profile": patient_profile,
            "retry_count": retry_count,
            "trials_raw": [],
        }
        config: RunnableConfig | None = None
        if thread_id:
            config = {"configurable": {"thread_id": f"{thread_id}:retrieval:{retry_count}"}}
        result = await compiled_retrieval_graph.ainvoke(retrieval_input, config=config)
        return cast("dict[str, Any]", result)

    async def run_eligibility(
        self,
        patient_profile: PatientSummary,
        trials_deduplicated: list[TrialSummary],
        eligibility_verdicts: dict[str, dict[str, Any]] | None = None,
        thread_id: str | None = None,
        attempt: int = 0,
    ) -> dict[str, Any]:
        """
        Run eligibility to determine whether the given trials are eligible for the given patient.

        Args:
            patient_profile: A dictionary containing the patient's summary.
            trials_deduplicated: A list of dictionaries containing the deduplicated trials.
            eligibility_verdicts: A dictionary containing the eligibility verdicts.
            thread_id: A string indicating the thread the eligibility is running on.
            attempt: An integer indicating the number of times the eligibility should be retried.

        Returns:
            A dictionary containing the results of the eligibility.
        """
        trimmed_trials = []
        for trial in trials_deduplicated:
            criteria_text = str(trial.get("eligibility_criteria_raw", ""))
            trimmed_trials.append(
                {
                    **trial,
                    "eligibility_criteria_raw": criteria_text[: settings.criteria_text_max_chars],
                }
            )
        eligibility_graph = compile_eligibility_graph(use_postgres_checkpointer=False)
        eligibility_input = {
            "patient_profile": patient_profile,
            "trials_deduplicated": trimmed_trials,
            "eligibility_verdicts": eligibility_verdicts,
        }
        config: RunnableConfig | None = None
        if thread_id:
            config = {"configurable": {"thread_id": f"{thread_id}:eligibility:{attempt}"}}
        result = await eligibility_graph.ainvoke(eligibility_input, config=config)
        return cast("dict[str, Any]", result)

    async def run_synthesis(
        self,
        patient_profile: PatientSummary,
        trial_scores: list[ScoredTrialSummary],
        eligibility_verdicts: dict[str, dict[str, Any]] | None,
        missing_info_recommendations: list[dict[str, Any]] | None,
        trials_raw: list[dict[str, Any]],
        search_queries: list[str],
        decision_history: list[str],
        trials_with_criteria: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run synthesis synthesis to generate a report based on the given trial scores and eligibility verdicts.

        Args:
            patient_profile: A dictionary containing the patient's summary.
            trial_scores: A list of dictionaries containing the trial scores.
            eligibility_verdicts: A dictionary containing the eligibility verdicts.
            missing_info_recommendations: A list of dictionaries containing the missing information recommendations.
            trials_raw: A list of dictionaries containing the raw trials.
            search_queries: A list of strings containing the search queries used.
            decision_history: A list of strings containing the decision history.
            trials_with_criteria: A list of dictionaries containing the trials with criteria.
            thread_id: A string indicating the thread the synthesis synthesis is running on.

        Returns:
            A dictionary containing the results of the synthesis.
        """
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
        config: RunnableConfig | None = None
        if thread_id:
            config = {"configurable": {"thread_id": f"{thread_id}:synthesis"}}
        return await compiled_synthesis_graph.ainvoke(synthesis_input, config=config)

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

        patient_summary = self._project_patient_summary(patient_profile)
        agent_input = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Match this patient to trials by calling tools in order: "
                        "run_retrieval -> run_eligibility -> run_synthesis.\n"
                        f"Patient summary: {patient_summary}"
                    ),
                }
            ]
        }
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit,
        }
        if settings.supervisor_use_react:
            try:
                result = await asyncio.wait_for(
                    self._react_agent.ainvoke(agent_input, config=config),
                    timeout=settings.supervisor_agent_timeout_seconds,
                )
                normalized = self._extract_final_result(result)
                if "report_json" not in normalized and "report_text" in normalized:
                    normalized = await self._run_tools_pipeline(
                        patient_summary, thread_id=thread_id
                    )
            except TimeoutError:
                logger.warning(
                    "Supervisor ReAct agent timed out after {}s; using deterministic pipeline",
                    settings.supervisor_agent_timeout_seconds,
                )
                normalized = await self._run_tools_pipeline(patient_summary, thread_id=thread_id)
        else:
            normalized = await self._run_tools_pipeline(patient_summary, thread_id=thread_id)

        memory = EpisodicMemory()
        await memory.init()
        try:
            await memory.store(patient_profile, normalized)
        finally:
            await memory.close()

        return normalized

    async def _run_tools_pipeline(
        self, patient_profile: PatientSummary, *, thread_id: str
    ) -> dict[str, Any]:
        run_state = SupervisorRunState(retrieval_result={}, scored_trials=[], final_result={})
        retry_count = 0
        eligibility: dict[str, Any] = {}

        while retry_count < settings.max_retry_attempts:
            raw_retrieval = await self.run_retrieval(
                patient_profile=patient_profile,
                retry_count=retry_count,
                thread_id=thread_id,
            )
            run_state.retrieval_result = self._compress_retrieval_output(raw_retrieval)
            trials_for_eligibility = run_state.retrieval_result.get("trials_deduplicated", [])
            limited_trials = list(trials_for_eligibility)[: settings.max_trials_for_eligibility]

            eligibility = await self.run_eligibility(
                patient_profile=patient_profile,
                trials_deduplicated=limited_trials,
                eligibility_verdicts=None,
                thread_id=thread_id,
                attempt=retry_count,
            )
            run_state.scored_trials = self._compress_scored_trials(
                list(eligibility.get("trial_scores", []))
            )
            if not bool(eligibility.get("retrieval_needs_broadening", False)):
                break
            retry_count += 1

        synthesis_input_trials: list[ScoredTrialSummary] = [
            t
            for t in run_state.scored_trials
            if float(t.get("score", 0.0)) >= settings.min_match_score
        ]
        synthesis = await self.run_synthesis(
            patient_profile=patient_profile,
            trial_scores=synthesis_input_trials,
            eligibility_verdicts=eligibility.get("eligibility_verdicts"),
            missing_info_recommendations=eligibility.get("missing_info_recommendations"),
            trials_raw=list(run_state.retrieval_result.get("trials_raw", [])),
            search_queries=list(run_state.retrieval_result.get("search_queries", [])),
            decision_history=self._compress_decision_history(
                list(eligibility.get("decision_history", []))
            ),
            trials_with_criteria=eligibility.get("trials_with_criteria"),
            thread_id=thread_id,
        )
        run_state.final_result = synthesis
        return run_state.final_result

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

    @staticmethod
    def _project_patient_summary(patient_profile: dict[str, Any]) -> PatientSummary:
        keys = (
            "age",
            "sex",
            "primary_condition",
            "conditions",
            "medical_history",
            "medications",
            "biomarkers",
            "lab_values",
            "prior_treatments",
            "contraindications",
            "ecog_performance_status",
            "smoking_status",
            "bmi",
        )
        return {k: patient_profile[k] for k in keys if patient_profile.get(k)}

    @staticmethod
    def _project_trial_summary(trial: dict[str, Any]) -> TrialSummary:
        criteria_text = str(trial.get("eligibility_criteria_raw", ""))[
            : settings.criteria_text_max_chars
        ]
        return {
            "nct_id": str(trial.get("nct_id", "")),
            "brief_title": str(trial.get("brief_title", "")),
            "overall_status": str(trial.get("overall_status", "")),
            "phase": str(trial.get("phase", "")),
            "conditions": list(trial.get("conditions", [])),
            "interventions": list(trial.get("interventions", [])),
            "eligibility_criteria_raw": criteria_text,
            "locations": list(trial.get("locations", []))[:5],
            "primary_completion_date": str(trial.get("primary_completion_date", "")),
        }

    def _compress_retrieval_output(self, retrieval: dict[str, Any]) -> dict[str, Any]:
        deduped = [
            self._project_trial_summary(t)
            for t in list(retrieval.get("trials_deduplicated", []))
            if isinstance(t, dict)
        ]
        raw = [
            self._project_trial_summary(t)
            for t in list(retrieval.get("trials_raw", []))
            if isinstance(t, dict)
        ]
        return {
            "trials_deduplicated": deduped,
            "trials_raw": raw,
            "search_queries": list(retrieval.get("search_queries", [])),
        }

    @staticmethod
    def _compress_scored_trials(
        trials: list[dict[str, Any]],
    ) -> list[ScoredTrialSummary]:
        keep = (
            "trial_id",
            "brief_title",
            "overall_status",
            "phase",
            "score",
            "confidence",
            "tier",
            "meets_count",
            "fails_count",
            "uncertain_count",
            "hard_exclusion_failures",
            "key_inclusion_passed",
            "key_exclusion_failed",
            "key_uncertain",
            "locations_summary",
        )
        compressed: list[ScoredTrialSummary] = []
        for trial in trials:
            slim: ScoredTrialSummary = {}
            for key in keep:
                value = trial.get(key)
                if value is not None:
                    slim[key] = value
            compressed.append(slim)
        return compressed

    @staticmethod
    def _compress_decision_history(decisions: list[str]) -> list[str]:
        return [d[:240] for d in decisions[-20:]]


@contextlib.asynccontextmanager
async def compile_supervisor_graph() -> Any:
    checkpointer = get_checkpointer(settings.database_uri)
    if checkpointer is not None and hasattr(checkpointer, "__aenter__"):
        async with checkpointer as active_checkpointer:
            setup_result = active_checkpointer.setup()
            if inspect.isawaitable(setup_result):
                await setup_result
            yield SupervisorOrchestrator(checkpointer=active_checkpointer)
    else:
        if checkpointer is not None and hasattr(checkpointer, "setup"):
            setup_result = checkpointer.setup()
            if inspect.isawaitable(setup_result):
                await setup_result
        yield SupervisorOrchestrator(checkpointer=checkpointer)
