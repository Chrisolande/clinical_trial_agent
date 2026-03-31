from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from .nodes import (
    assess_and_finalize,
    broaden_and_retry,
    execute_searches,
    initialize_retrieval,
    should_retry_search,
)
from .state import RetrievalInput, RetrievalOutput, RetrievalState


def _build_retrieval_graph():
    graph = StateGraph(RetrievalState, input_schema=RetrievalInput, output_schema=RetrievalOutput)

    graph.add_node("initialize_retrieval", initialize_retrieval)
    graph.add_node("execute_searches", execute_searches)
    graph.add_node("broaden_and_retry", broaden_and_retry)
    graph.add_node("assess_and_finalize", assess_and_finalize)

    graph.add_edge(START, "initialize_retrieval")
    graph.add_edge("initialize_retrieval", "execute_searches")
    graph.add_conditional_edges(
        "execute_searches",
        should_retry_search,
        {"retry_search": "broaden_and_retry", "finalize": "assess_and_finalize"},
    )

    graph.add_edge("broaden_and_retry", "execute_searches")
    graph.add_edge("assess_and_finalize", END)

    return cast("StateGraph[Any]", graph)


compiled_retrieval_graph = _build_retrieval_graph().compile()
