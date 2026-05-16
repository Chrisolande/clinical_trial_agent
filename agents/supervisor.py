import contextlib
import inspect
import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from loguru import logger
from prompts.supervisor import build_supervisor_prompt
from subagents.eligibility.graph import compiled_eligibility_graph
from subagents.retrieval.graph import compiled_retrieval_graph
from subagents.synthesis.graph import compiled_synthesis_graph

from agents.supervisor_helpers import (
    apply_feedback_adjustments,
    extract_final_result,
    project_patient_summary,
)
from agents.supervisor_helpers import (
    compute_tier_counts as _compute_tier_counts,
)
from agents.supervisor_pipeline import run_tools_pipeline
from clinical_trial_agent.config import get_settings
from clinical_trial_agent.memory import EpisodicMemory, get_checkpointer
from clinical_trial_agent.normalizers import (
    _normalize_eligibility_result,
    _normalize_retrieval_result,
    _normalize_supervisor_output,
    _require_dict,
)

PatientSummary = dict[str, Any]
TrialSummary = dict[str, Any]
ScoredTrialSummary = dict[str, Any]


def _apply_feedback_adjustments(
    scored_trials: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return apply_feedback_adjustments(scored_trials, feedback_rows)


class SupervisorOrchestrator:
    def __init__(self, checkpointer: Any | None = None) -> None:
        self._checkpointer = checkpointer
        self._react_agent = None
        if get_settings().supervisor_use_react:
            self._react_agent = create_agent(
                model=self._get_llm(),
                tools=[self.run_retrieval, self.run_eligibility, self.run_synthesis],
                name="clinical_supervisor",
                system_prompt=build_supervisor_prompt(),
                checkpointer=checkpointer,
            )

    @staticmethod
    def _get_llm() -> Any:
        from clinical_trial_agent.config import get_llm

        return get_llm()

    @staticmethod
    def _get_settings() -> Any:
        return get_settings()

    async def run_retrieval(
        self,
        patient_profile: PatientSummary,
        normalized_terms: dict[str, Any] | None = None,
        retry_count: int = 0,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Run retrieval subgraph and return normalized retrieval output."""
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
        return _normalize_retrieval_result(_require_dict(result, source="retrieval graph"))

    async def run_eligibility(
        self,
        patient_profile: PatientSummary,
        trials_deduplicated: list[TrialSummary],
        eligibility_verdicts: dict[str, dict[str, Any]] | None = None,
        thread_id: str | None = None,
        attempt: int = 0,
    ) -> dict[str, Any]:
        """Run eligibility subgraph over candidate trials and return scored verdicts."""
        settings = get_settings()
        trimmed_trials = [
            {
                **trial,
                "eligibility_criteria_raw": str(trial.get("eligibility_criteria_raw", ""))[
                    : settings.criteria_text_max_chars
                ],
            }
            for trial in trials_deduplicated
        ]
        eligibility_input = {
            "patient_profile": patient_profile,
            "trials_deduplicated": trimmed_trials,
            "eligibility_verdicts": eligibility_verdicts,
        }
        config: RunnableConfig | None = None
        if thread_id:
            config = {"configurable": {"thread_id": f"{thread_id}:eligibility:{attempt}"}}
        result = await compiled_eligibility_graph.ainvoke(eligibility_input, config=config)
        return _normalize_eligibility_result(_require_dict(result, source="eligibility graph"))

    async def run_synthesis(
        self,
        patient_profile: PatientSummary,
        trial_scores: list[ScoredTrialSummary],
        eligibility_verdicts: dict[str, dict[str, Any]] | None,
        missing_info_recommendations: list[dict[str, Any]] | None,
        trials_raw: list[dict[str, Any]],
        search_queries: list[str],
        decision_history: list[str],
        retrieval_errors: list[str] | None = None,
        trials_with_criteria: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Run synthesis subgraph to generate final report artifacts."""
        synthesis_input = {
            "patient_profile": patient_profile,
            "trial_scores": trial_scores,
            "eligibility_verdicts": eligibility_verdicts,
            "missing_info_recommendations": missing_info_recommendations,
            "trials_raw": trials_raw,
            "search_queries": search_queries,
            "decision_history": decision_history,
            "retrieval_errors": list(retrieval_errors or []),
            "trials_with_criteria": trials_with_criteria,
        }
        config: RunnableConfig | None = None
        if thread_id:
            config = {"configurable": {"thread_id": f"{thread_id}:synthesis"}}
        result = await compiled_synthesis_graph.ainvoke(synthesis_input, config=config)
        return _require_dict(result, source="synthesis graph")

    async def ainvoke(
        self,
        patient_profile: dict[str, Any],
        *,
        thread_id: str,
        recursion_limit: int = 25,
    ) -> dict[str, Any]:
        if recursion_limit < 1:
            raise ValueError("recursion_limit must be >= 1")

        memory_ctx = EpisodicMemory()
        async with memory_ctx:
            memory = memory_ctx
            cached = await memory.lookup(patient_profile)
            if isinstance(cached, dict) and (
                isinstance(cached.get("report_json"), dict)
                or bool(str(cached.get("report_text", "")).strip())
            ):
                logger.info("Supervisor served result from episodic memory")
                return _normalize_supervisor_output(cached)

            patient_summary = project_patient_summary(patient_profile)

            if self._react_agent is None:
                result = await run_tools_pipeline(
                    self, patient_summary, thread_id=thread_id, memory=memory
                )
            else:
                try:
                    react_result = await self._react_agent.ainvoke(
                        {
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
                        },
                        config={"configurable": {"thread_id": thread_id}},
                    )
                    extracted = extract_final_result(react_result)
                    if isinstance(extracted.get("report_json"), dict):
                        result = extracted
                    else:
                        result = await run_tools_pipeline(
                            self, patient_summary, thread_id=thread_id, memory=memory
                        )
                except Exception as exc:
                    logger.exception(
                        "Supervisor react phase failed (thread_id={}, node=supervisor.react): {}",
                        thread_id,
                        exc,
                    )
                    result = await run_tools_pipeline(
                        self, patient_summary, thread_id=thread_id, memory=memory
                    )

            result = _normalize_supervisor_output(result)
            await memory.store(patient_profile, result)
            tier_counts = _compute_tier_counts(
                list(result.get("report_json", {}).get("ranked_trials", []))
            )
            await memory.write_pipeline_audit(
                patient_profile=patient_profile,
                run_id=thread_id,
                outcome_tier_counts=tier_counts,
                model_version=get_settings().deepseek_model,
                consent_flag=os.getenv("CLINICAL_DATA_EXTERNAL_LLM_CONSENT", "false").lower()
                == "true",
            )
            return result

    async def _run_tools_pipeline(
        self, patient_profile: PatientSummary, *, thread_id: str, memory: EpisodicMemory
    ) -> dict[str, Any]:
        return await run_tools_pipeline(self, patient_profile, thread_id=thread_id, memory=memory)


@contextlib.asynccontextmanager
async def compile_supervisor_graph() -> Any:
    checkpointer_cm = get_checkpointer(get_settings().database_uri)
    if checkpointer_cm is None:
        yield SupervisorOrchestrator(checkpointer=None)
        return

    async with checkpointer_cm as checkpointer:
        setup_result = checkpointer.setup()
        if inspect.isawaitable(setup_result):
            await setup_result
        yield SupervisorOrchestrator(checkpointer=checkpointer)
