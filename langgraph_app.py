from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

from config import get_settings
from langgraph.graph import END, START, StateGraph
from subagents.eligibility.graph import _build_eligibility_graph
from subagents.retrieval.graph import _build_retrieval_graph
from subagents.synthesis.graph import _build_synthesis_graph


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


class EndToEndOutput(TypedDict):
    report_json: dict[str, Any] | None
    report_text: str | None


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
    config = {"configurable": {"thread_id": _thread_id(state, "retrieval")}}
    result = await COMPILED_RETRIEVAL.ainvoke(retrieval_input, config=cast("Any", config))
    return {"retrieval_result": cast("dict[str, Any]", result)}


async def run_eligibility_node(state: EndToEndState) -> dict[str, Any]:
    retrieval_result = state.get("retrieval_result") or {}
    trials_for_eligibility = list(retrieval_result.get("trials_deduplicated", []))
    limited_trials = trials_for_eligibility[: get_settings().max_trials_for_eligibility]

    eligibility_input = {
        "patient_profile": state.get("patient_profile") or {},
        "trials_deduplicated": limited_trials,
        "eligibility_verdicts": None,
    }
    config = {"configurable": {"thread_id": _thread_id(state, "eligibility")}}
    result = await COMPILED_ELIGIBILITY.ainvoke(eligibility_input, config=cast("Any", config))
    return {"eligibility_result": cast("dict[str, Any]", result)}


def route_after_eligibility(
    state: EndToEndState,
) -> Literal["retry_retrieval", "run_synthesis"]:
    if get_settings().one_pass_mode:
        return "run_synthesis"
    eligibility_result = state.get("eligibility_result") or {}
    should_retry = bool(eligibility_result.get("retrieval_needs_broadening", False))
    retry_count = int(state.get("retry_count", 0))
    if should_retry and retry_count < get_settings().max_retry_attempts:
        return "retry_retrieval"
    return "run_synthesis"


async def retry_retrieval_node(state: EndToEndState) -> dict[str, Any]:
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
        "trials_with_criteria": eligibility_result.get("trials_with_criteria"),
    }
    config = {"configurable": {"thread_id": _thread_id(state, "synthesis")}}
    result = await COMPILED_SYNTHESIS.ainvoke(synthesis_input, config=cast("Any", config))
    casted = cast("dict[str, Any]", result)
    return {
        "synthesis_result": casted,
        "report_json": casted.get("report_json"),
        "report_text": casted.get("report_text"),
    }


def _build_end_to_end_graph() -> Any:
    graph = StateGraph(
        EndToEndState,
        input_schema=EndToEndInput,
        output_schema=EndToEndOutput,
    )

    graph.add_node("run_retrieval", run_retrieval_node)
    graph.add_node("run_eligibility", run_eligibility_node)
    graph.add_node("retry_retrieval", retry_retrieval_node)
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
    graph.add_edge("run_synthesis", END)

    return graph.compile()


end_to_end_graph = _build_end_to_end_graph()
