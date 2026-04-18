import contextlib
import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from loguru import logger
from subagents.eligibility.graph import compiled_eligibility_graph
from subagents.retrieval.graph import compiled_retrieval_graph
from subagents.synthesis.graph import compiled_synthesis_graph

from config import TIER_ORDER, get_settings
from memory import EpisodicMemory, get_checkpointer

PatientSummary = dict[str, Any]
TrialSummary = dict[str, Any]
ScoredTrialSummary = dict[str, Any]


def _require_dict(value: Any, *, source: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise TypeError(f"{source} must return dict, got {type(value)!r}")


def _apply_feedback_adjustments(
    scored_trials: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    boosts: dict[str, int] = {}
    for row in feedback_rows:
        nct = str(row.get("nct_id", "")).strip()
        verdict = str(row.get("verdict", "")).strip().lower()
        if not nct:
            continue
        if verdict == "confirmed":
            boosts[nct] = boosts.get(nct, 0) + 1
        elif verdict == "rejected":
            boosts[nct] = boosts.get(nct, 0) - 1

    adjusted = [dict(trial) for trial in scored_trials]
    for trial in adjusted:
        trial_id = str(trial.get("trial_id", "")).strip()
        delta = boosts.get(trial_id, 0)
        if delta == 0:
            continue
        trial["score"] = max(0.0, min(1.0, float(trial.get("score", 0.0)) + 0.05 * delta))

    adjusted.sort(
        key=lambda x: (
            TIER_ORDER.get(str(x.get("tier", "weak")), 0),
            float(x.get("score", 0.0)),
        ),
        reverse=True,
    )
    for idx, trial in enumerate(adjusted, 1):
        trial["rank"] = idx
    return adjusted


def _profile_hash_for_feedback(patient_profile: dict[str, Any]) -> str:
    canonical = json.dumps(patient_profile, sort_keys=True, default=str)
    salt = os.getenv("PROFILE_HASH_SALT", "")
    return hashlib.sha256(f"{salt}::{canonical}".encode()).hexdigest()


def _compute_tier_counts(scored_trials: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"strong": 0, "moderate": 0, "weak": 0, "disqualified": 0}
    for trial in scored_trials:
        tier = str(trial.get("tier", "weak"))
        counts[tier] = counts.get(tier, 0) + 1
    return counts


@dataclass
class SupervisorRunState:
    retrieval_result: dict[str, Any]
    scored_trials: list[ScoredTrialSummary]
    final_result: dict[str, Any]


class SupervisorOrchestrator:
    def __init__(self, checkpointer: Any | None = None) -> None:
        self._checkpointer = checkpointer
        self._react_agent = None
        if get_settings().supervisor_use_react:
            self._react_agent = create_agent(
                model=self._get_llm(),
                tools=[self.run_retrieval, self.run_eligibility, self.run_synthesis],
                name="clinical_supervisor",
                system_prompt="Use tools in order: run_retrieval -> run_eligibility -> run_synthesis.",
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
        return _require_dict(result, source="retrieval graph")

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
        return _require_dict(result, source="eligibility graph")

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
        """Run synthesis subgraph to generate final report artifacts."""
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
            if isinstance(cached, dict):
                logger.info("Supervisor served result from episodic memory")
                return cached

            patient_summary = self._project_patient_summary(patient_profile)

            if self._react_agent is None:
                result = await self._run_tools_pipeline(
                    patient_summary, thread_id=thread_id, memory=memory
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
                    extracted = self._extract_final_result(react_result)
                    if isinstance(extracted.get("report_json"), dict):
                        result = extracted
                    else:
                        result = await self._run_tools_pipeline(
                            patient_summary, thread_id=thread_id, memory=memory
                        )
                except Exception as exc:
                    logger.exception(
                        "Supervisor react phase failed (thread_id={}, node=supervisor.react): {}",
                        thread_id,
                        exc,
                    )
                    result = await self._run_tools_pipeline(
                        patient_summary, thread_id=thread_id, memory=memory
                    )

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
        run_state = SupervisorRunState(retrieval_result={}, scored_trials=[], final_result={})
        settings = get_settings()
        max_attempts = 1 if settings.one_pass_mode else settings.max_retry_attempts
        eligibility: dict[str, Any] = {}

        for retry_count in range(max_attempts):
            raw_retrieval = await self.run_retrieval(
                patient_profile=patient_profile,
                retry_count=retry_count,
                thread_id=thread_id,
            )
            run_state.retrieval_result = self._compress_retrieval_output(raw_retrieval)

            limited_trials = list(run_state.retrieval_result.get("trials_deduplicated", []))[
                : settings.max_trials_for_eligibility
            ]

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

            if settings.one_pass_mode or not bool(
                eligibility.get("retrieval_needs_broadening", False)
            ):
                break

        feedback_rows: list[dict[str, Any]] = []
        if hasattr(memory, "list_feedback"):
            feedback_rows = await memory.list_feedback(_profile_hash_for_feedback(patient_profile))
        adjusted_scores = _apply_feedback_adjustments(run_state.scored_trials, feedback_rows)

        synthesis = await self.run_synthesis(
            patient_profile=patient_profile,
            trial_scores=adjusted_scores,
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
        run_state.final_result = self._unwrap_synthesis_result(synthesis)
        return run_state.final_result

    @staticmethod
    def _unwrap_synthesis_result(result: Any) -> dict[str, Any]:
        """Promote report_json to the top level regardless of how synthesis returns it.

        Synthesis may return:
        - dict with report_json key directly (ideal path)
        - dict with report_text key containing a JSON string (LLM serialised it)
        - something else (degrade gracefully)
        """
        if not isinstance(result, dict):
            return {"report_text": str(result)}

        if isinstance(result.get("report_json"), dict):
            return result

        report_text = result.get("report_text", "")
        if isinstance(report_text, str) and report_text.strip().startswith("{"):
            try:
                parsed = json.loads(report_text)
                if isinstance(parsed, dict) and "report_json" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        return result

    @staticmethod
    def _unwrap_report_json(content: str) -> dict[str, Any] | None:
        """Unwrap nested report_json emitted as a JSON-encoded message content."""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            nested = parsed.get("report_json")
            if isinstance(nested, dict):
                merged = dict(parsed)
                merged["report_json"] = nested
                return merged
            if "report_text" in parsed:
                return parsed
        return None

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
                    unwrapped = SupervisorOrchestrator._unwrap_report_json(content)
                    if isinstance(unwrapped, dict):
                        return unwrapped
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
        settings = get_settings()
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
            "tier",
            "major_criteria_assessable",
            "key_concern",
            "critical_missing_info",
            "rationale",
            "meets_count",
            "fails_count",
            "uncertain_count",
            "hard_exclusion_failures",
            "key_inclusion_passed",
            "key_exclusion_failed",
            "key_uncertain",
            "locations_summary",
        )
        return [{key: trial[key] for key in keep if trial.get(key) is not None} for trial in trials]

    @staticmethod
    def _compress_decision_history(decisions: list[str]) -> list[str]:
        return [d[:240] for d in decisions[-20:]]


@contextlib.asynccontextmanager
async def compile_supervisor_graph() -> Any:
    checkpointer = get_checkpointer(get_settings().database_uri)
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
