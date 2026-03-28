from __future__ import annotations

import asyncio

from config import bootstrap_environment, settings
from langgraph.graph import END, START, StateGraph
from memory import get_checkpointer

from .nodes import (
    aggregate_results,
    assess_viability_signal,
    dispatch_trial_workers,
    evaluate_trial_worker,
    fan_out_trials,
    identify_missing_info,
)
from .state import (
    EligibilityInput,
    EligibilityOutput,
    EligibilityState,
)

bootstrap_environment()


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


def compile_eligibility_graph(*, use_postgres_checkpointer: bool = True):
    graph = _build_eligibility_graph()
    if not use_postgres_checkpointer:
        return graph.compile()

    checkpointer = get_checkpointer(settings.database_uri)
    if checkpointer is None:
        return graph.compile()
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    input_data = {
        "patient_profile": {"age": 45, "primary_condition": "Condition X"},
        "trials_deduplicated": [
            {
                "nct_id": "T1",
                "eligibility_criteria_raw": "Inclusion: age >= 18. Exclusion: pregnancy.",
            },
            {
                "nct_id": "T2",
                "eligibility_criteria_raw": "Inclusion: confirmed diagnosis of Condition X.",
            },
        ],
        "eligibility_verdicts": None,
    }

    app = compile_eligibility_graph(use_postgres_checkpointer=True)
    result = asyncio.run(app.ainvoke(input_data))
    print(result)
