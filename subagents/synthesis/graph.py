from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from .nodes import (
    attempt_qa_fix,
    finalize_synthesis_output,
    flag_re_evaluation,
    generate_report_node,
    route_after_qa,
    run_qa_check,
)
from .state import SynthesisInput, SynthesisOutput, SynthesisState


def _build_synthesis_graph() -> StateGraph:
    graph = StateGraph(SynthesisState, input_schema=SynthesisInput, output_schema=SynthesisOutput)

    graph.add_node("run_qa_check", run_qa_check)
    graph.add_node("attempt_qa_fix", attempt_qa_fix)
    graph.add_node("flag_re_evaluation", flag_re_evaluation)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("finalize_output", finalize_synthesis_output)

    graph.add_edge(START, "run_qa_check")

    graph.add_conditional_edges(
        "run_qa_check",
        route_after_qa,
        {
            "generate_report": "generate_report",
            "attempt_qa_fix": "attempt_qa_fix",
            "flag_re_evaluation": "flag_re_evaluation",
        },
    )

    graph.add_edge("attempt_qa_fix", "run_qa_check")
    graph.add_edge("flag_re_evaluation", "generate_report")
    graph.add_edge("generate_report", "finalize_output")
    graph.add_edge("finalize_output", END)

    return cast("StateGraph[Any]", graph)


compiled_synthesis_graph = _build_synthesis_graph().compile()
