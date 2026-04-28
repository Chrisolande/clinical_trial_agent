from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from subagents.eligibility.graph import _build_eligibility_graph
from subagents.retrieval.graph import _build_retrieval_graph
from subagents.synthesis.graph import _build_synthesis_graph
from tools.telemetry import trace_span

from clinical_trial_agent.config import get_settings
from clinical_trial_agent.normalizers import (
    _normalize_eligibility_result,
    _normalize_retrieval_result,
    _normalize_supervisor_output,
    _require_dict,
)


class EndToEndInput(TypedDict):
    patient_profile: dict[str, Any]
    thread_id: str | None


class EndToEndState(TypedDict):
    patient_profile: dict[str, Any]
    thread_id: str | None
    retry_count: int
    retrieval_result: dict[str, Any]
    eligibility_result: dict[str, Any]
    synthesis_result: dict[str, Any]
    report_json: dict[str, Any] | None
    report_text: str | None
    synthesis_needs_re_evaluation: bool
    synthesis_retry_retrieval: bool


class EndToEndOutput(TypedDict):
    report_json: dict[str, Any] | None
    report_text: str | None


def _normalize_output_contract(value: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_supervisor_output(value)
    report_json = normalized.get("report_json")
    report_text = normalized.get("report_text")
    return {
        **normalized,
        "report_json": report_json if isinstance(report_json, dict) else None,
        "report_text": report_text if isinstance(report_text, str) else None,
    }


def _thread_id(state: EndToEndState, stage: str) -> str:
    base = state.get("thread_id") or "langgraph-dev-thread"
    retry = int(state.get("retry_count", 0))
    if stage == "retrieval":
        return f"{base}:retrieval:{retry}"
    if stage == "eligibility":
        return f"{base}:eligibility:{retry}"
    return f"{base}:synthesis"


COMPILED_RETRIEVAL = _build_retrieval_graph().compile()
COMPILED_ELIGIBILITY = _build_eligibility_graph().compile()
COMPILED_SYNTHESIS = _build_synthesis_graph().compile()


async def run_retrieval_node(state: EndToEndState) -> dict[str, Any]:
    retrieval_input = {
        "normalized_terms": {},
        "patient_profile": state.get("patient_profile") or {},
        "retry_count": int(state.get("retry_count", 0)),
        "trials_raw": [],
    }
    config: RunnableConfig = {"configurable": {"thread_id": _thread_id(state, "retrieval")}}
    with trace_span(
        "langgraph.run_retrieval",
        run_id=state.get("thread_id") or "langgraph-dev-thread",
        attempt=int(state.get("retry_count", 0)),
        slo_ms=6000,
    ):
        result = await COMPILED_RETRIEVAL.ainvoke(retrieval_input, config=config)
    return {
        "retrieval_result": _normalize_retrieval_result(
            _require_dict(result, source="retrieval subgraph")
        )
    }


async def run_eligibility_node(state: EndToEndState) -> dict[str, Any]:
    retrieval_result = state.get("retrieval_result") or {}
    trials_for_eligibility = list(retrieval_result.get("trials_deduplicated", []))
    limited_trials = trials_for_eligibility[: get_settings().max_trials_for_eligibility]

    eligibility_input = {
        "patient_profile": state.get("patient_profile") or {},
        "trials_deduplicated": limited_trials,
        "eligibility_verdicts": None,
    }
    config: RunnableConfig = {"configurable": {"thread_id": _thread_id(state, "eligibility")}}
    with trace_span(
        "langgraph.run_eligibility",
        run_id=state.get("thread_id") or "langgraph-dev-thread",
        attempt=int(state.get("retry_count", 0)),
        trial_count=len(limited_trials),
        slo_ms=12000,
    ):
        result = await COMPILED_ELIGIBILITY.ainvoke(eligibility_input, config=config)
    return {
        "eligibility_result": _normalize_eligibility_result(
            _require_dict(result, source="eligibility subgraph")
        )
    }


def route_after_eligibility(
    state: EndToEndState,
) -> Literal["retry_retrieval", "run_synthesis"]:
    if get_settings().one_pass_mode:
        return "run_synthesis"
    eligibility_result = state.get("eligibility_result") or {}
    should_retry = bool(eligibility_result.get("retrieval_needs_broadening", False))
    retry_count = int(state.get("retry_count", 0))
    # retry_count is the attempt index (0=initial attempt, 1=first retry, ...).
    max_retries = max(0, int(get_settings().max_retry_attempts))
    if should_retry and retry_count < max_retries:
        return "retry_retrieval"
    return "run_synthesis"


async def retry_retrieval_node(state: EndToEndState) -> dict[str, Any]:
    return {"retry_count": int(state.get("retry_count", 0)) + 1}


async def retry_eligibility_node(state: EndToEndState) -> dict[str, Any]:
    return {"retry_count": int(state.get("retry_count", 0)) + 1}


async def run_synthesis_node(state: EndToEndState) -> dict[str, Any]:
    retrieval_result = state.get("retrieval_result") or {}
    eligibility_result = state.get("eligibility_result") or {}

    synthesis_input = {
        "patient_profile": state.get("patient_profile") or {},
        "trial_scores": list(eligibility_result.get("trial_scores", [])),
        "eligibility_verdicts": eligibility_result.get("eligibility_verdicts"),
        "missing_info_recommendations": eligibility_result.get("missing_info_recommendations"),
        "trials_raw": list(retrieval_result.get("trials_raw", [])),
        "search_queries": list(retrieval_result.get("search_queries", [])),
        "decision_history": list(eligibility_result.get("decision_history", [])),
        "retrieval_errors": list(
            retrieval_result.get("retrieval_errors", retrieval_result.get("errors", []))
        ),
        "trials_with_criteria": eligibility_result.get("trials_with_criteria"),
    }
    config: RunnableConfig = {"configurable": {"thread_id": _thread_id(state, "synthesis")}}
    with trace_span(
        "langgraph.run_synthesis",
        run_id=state.get("thread_id") or "langgraph-dev-thread",
        slo_ms=6000,
    ):
        result = _require_dict(
            await COMPILED_SYNTHESIS.ainvoke(synthesis_input, config=config),
            source="synthesis subgraph",
        )
    normalized = _normalize_output_contract(result)
    needs_re_eval = bool(normalized.get("synthesis_needs_re_evaluation", False))
    needs_retrieval_retry = bool(normalized.get("synthesis_retry_retrieval", False))
    return {
        "synthesis_result": normalized,
        "report_json": normalized.get("report_json"),
        "report_text": normalized.get("report_text"),
        "synthesis_needs_re_evaluation": needs_re_eval,
        "synthesis_retry_retrieval": needs_retrieval_retry,
    }


def route_after_synthesis(
    state: EndToEndState,
) -> Literal["retry_retrieval", "retry_eligibility", "end"]:
    if get_settings().one_pass_mode:
        return "end"
    retry_count = int(state.get("retry_count", 0))
    max_retries = max(0, int(get_settings().max_retry_attempts))
    if retry_count >= max_retries:
        return "end"
    if not bool(state.get("synthesis_needs_re_evaluation", False)):
        return "end"
    if bool(state.get("synthesis_retry_retrieval", False)):
        return "retry_retrieval"
    return "retry_eligibility"


def _build_end_to_end_graph() -> Any:
    graph = StateGraph(
        EndToEndState,
        input_schema=EndToEndInput,
        output_schema=EndToEndOutput,
    )

    graph.add_node("run_retrieval", run_retrieval_node)
    graph.add_node("run_eligibility", run_eligibility_node)
    graph.add_node("retry_retrieval", retry_retrieval_node)
    graph.add_node("retry_eligibility", retry_eligibility_node)
    graph.add_node("run_synthesis", run_synthesis_node)

    graph.add_edge(START, "run_retrieval")
    graph.add_edge("run_retrieval", "run_eligibility")
    graph.add_conditional_edges(
        "run_eligibility",
        route_after_eligibility,
        {
            "retry_retrieval": "retry_retrieval",
            "run_synthesis": "run_synthesis",
        },
    )
    graph.add_edge("retry_retrieval", "run_retrieval")
    graph.add_edge("retry_eligibility", "run_eligibility")
    graph.add_conditional_edges(
        "run_synthesis",
        route_after_synthesis,
        {
            "retry_retrieval": "retry_retrieval",
            "retry_eligibility": "retry_eligibility",
            "end": END,
        },
    )

    return graph.compile()


end_to_end_graph = _build_end_to_end_graph()
