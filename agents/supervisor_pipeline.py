import inspect
from dataclasses import dataclass
from typing import Any

from tools.telemetry import trace_span

from clinical_trial_agent.normalizers import _normalize_supervisor_output

from .supervisor_helpers import (
    apply_feedback_adjustments,
    compress_decision_history,
    compress_retrieval_output,
    compress_scored_trials,
    unwrap_synthesis_result,
)

PatientSummary = dict[str, Any]
ScoredTrialSummary = dict[str, Any]


@dataclass
class SupervisorRunState:
    retrieval_result: dict[str, Any]
    scored_trials: list[ScoredTrialSummary]
    final_result: dict[str, Any]


async def run_tools_pipeline(
    orchestrator: Any,
    patient_profile: PatientSummary,
    *,
    thread_id: str,
    memory: Any,
) -> dict[str, Any]:
    run_state = SupervisorRunState(retrieval_result={}, scored_trials=[], final_result={})
    settings = orchestrator._get_settings()
    max_retries = max(0, int(settings.max_retry_attempts))
    max_attempts = 1 if settings.one_pass_mode else max_retries + 1
    eligibility: dict[str, Any] = {}
    synthesis: dict[str, Any] = {}

    for retry_count in range(max_attempts):
        eligibility = await run_retrieval_eligibility_attempt(
            orchestrator,
            patient_profile,
            thread_id=thread_id,
            attempt=retry_count,
            run_state=run_state,
            eligibility_verdicts=None,
            retrieval_span="supervisor.run_retrieval",
            eligibility_span="supervisor.run_eligibility",
        )
        if settings.one_pass_mode or not bool(eligibility.get("retrieval_needs_broadening", False)):
            break

    feedback_rows: list[dict[str, Any]] = []
    if hasattr(memory, "list_feedback"):
        feedback_rows = await memory.list_feedback(patient_profile)
    adjusted_scores = apply_feedback_adjustments(run_state.scored_trials, feedback_rows)

    for reeval_attempt in range(max_attempts):
        with trace_span("supervisor.run_synthesis", run_id=thread_id, slo_ms=6000):
            synthesis = await orchestrator.run_synthesis(
                **build_synthesis_kwargs(
                    orchestrator,
                    patient_profile=patient_profile,
                    adjusted_scores=adjusted_scores,
                    eligibility=eligibility,
                    run_state=run_state,
                    thread_id=thread_id,
                ),
            )
        run_state.final_result = unwrap_synthesis_result(synthesis)
        needs_re_eval = bool(run_state.final_result.get("synthesis_needs_re_evaluation", False))
        needs_retrieval_retry = bool(run_state.final_result.get("synthesis_retry_retrieval", False))

        if settings.one_pass_mode or not needs_re_eval or reeval_attempt >= max_retries:
            break

        eligibility = await run_retrieval_eligibility_attempt(
            orchestrator,
            patient_profile,
            thread_id=thread_id,
            attempt=reeval_attempt + 1,
            run_state=run_state,
            eligibility_verdicts=eligibility.get("eligibility_verdicts"),
            retrieval_span="supervisor.remediation_retrieval_retry",
            eligibility_span="supervisor.remediation_eligibility_retry",
            run_retrieval_first=needs_retrieval_retry,
        )
        adjusted_scores = apply_feedback_adjustments(run_state.scored_trials, feedback_rows)

    return _normalize_supervisor_output(run_state.final_result)


async def run_retrieval_eligibility_attempt(
    orchestrator: Any,
    patient_profile: PatientSummary,
    *,
    thread_id: str,
    attempt: int,
    run_state: SupervisorRunState,
    eligibility_verdicts: dict[str, dict[str, Any]] | None,
    retrieval_span: str,
    eligibility_span: str,
    run_retrieval_first: bool = True,
) -> dict[str, Any]:
    settings = orchestrator._get_settings()
    if run_retrieval_first:
        with trace_span(retrieval_span, run_id=thread_id, attempt=attempt, slo_ms=6000):
            raw_retrieval = await orchestrator.run_retrieval(
                patient_profile=patient_profile,
                retry_count=attempt,
                thread_id=thread_id,
            )
        run_state.retrieval_result = compress_retrieval_output(raw_retrieval)

    limited_trials = list(run_state.retrieval_result.get("trials_deduplicated", []))[
        : settings.max_trials_for_eligibility
    ]
    with trace_span(
        eligibility_span,
        run_id=thread_id,
        attempt=attempt,
        trial_count=len(limited_trials),
        slo_ms=12000,
    ):
        eligibility = await orchestrator.run_eligibility(
            patient_profile=patient_profile,
            trials_deduplicated=limited_trials,
            eligibility_verdicts=eligibility_verdicts,
            thread_id=thread_id,
            attempt=attempt,
        )
    run_state.scored_trials = compress_scored_trials(list(eligibility.get("trial_scores", [])))
    return eligibility


def build_synthesis_kwargs(
    orchestrator: Any,
    *,
    patient_profile: PatientSummary,
    adjusted_scores: list[dict[str, Any]],
    eligibility: dict[str, Any],
    run_state: SupervisorRunState,
    thread_id: str,
) -> dict[str, Any]:
    synthesis_kwargs: dict[str, Any] = {
        "patient_profile": patient_profile,
        "trial_scores": adjusted_scores,
        "eligibility_verdicts": eligibility.get("eligibility_verdicts"),
        "missing_info_recommendations": eligibility.get("missing_info_recommendations"),
        "trials_raw": list(run_state.retrieval_result.get("trials_raw", [])),
        "search_queries": list(run_state.retrieval_result.get("search_queries", [])),
        "decision_history": compress_decision_history(list(eligibility.get("decision_history", []))),
        "trials_with_criteria": eligibility.get("trials_with_criteria"),
        "thread_id": thread_id,
    }
    if "retrieval_errors" in inspect.signature(orchestrator.run_synthesis).parameters:
        synthesis_kwargs["retrieval_errors"] = list(run_state.retrieval_result.get("retrieval_errors", []))
    return synthesis_kwargs
