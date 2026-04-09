import asyncio
from typing import Any

from langgraph.graph import END, START, StateGraph

from config import bootstrap_environment

from .nodes import (
    aggregate_results,
    assess_viability_signal,
    dispatch_trial_workers,
    evaluate_trial_worker,
    fan_out_trials,
    identify_missing_info,
    set_llm_semaphore,
)
from .state import EligibilityInput, EligibilityOutput, EligibilityState

bootstrap_environment()

_LLM_SEMAPHORE = asyncio.Semaphore(5)
set_llm_semaphore(_LLM_SEMAPHORE)


def _build_eligibility_graph() -> StateGraph:
    graph = StateGraph(
        EligibilityState,
        input_schema=EligibilityInput,
        output_schema=EligibilityOutput,
    )

    graph.add_node("fan_out_trials", fan_out_trials)
    graph.add_node("evaluate_trial_worker", evaluate_trial_worker)

    # Add Aggregation and terminal nodes
    graph.add_node("aggregate_results", aggregate_results)
    graph.add_node("identify_missing_info", identify_missing_info)
    graph.add_node("assess_viability", assess_viability_signal)

    graph.add_edge(START, "fan_out_trials")
    graph.add_conditional_edges(
        "fan_out_trials",
        dispatch_trial_workers,
        ["evaluate_trial_worker"],
    )
    graph.add_edge("evaluate_trial_worker", "aggregate_results")
    graph.add_edge("aggregate_results", "identify_missing_info")
    graph.add_edge("identify_missing_info", "assess_viability")
    graph.add_edge("assess_viability", END)
    return graph


def compile_eligibility_graph(*, use_postgres_checkpointer: bool = True) -> Any:
    graph = _build_eligibility_graph()
    if not use_postgres_checkpointer:
        return graph.compile()
    # Disabled for now: this graph is compiled repeatedly inside the supervisor.
    # Reusing the supervisor-level checkpointer avoids duplicate setup/teardown paths.
    return graph.compile()


compiled_eligibility_graph = _build_eligibility_graph().compile()
