"""
Orchestration layer: wires Planner -> Retriever -> Synthesizer -> Evaluator
-> (loop back to Retriever, or) -> Reporter using LangGraph's StateGraph.

This mirrors the 5-agent architecture (Planner, Searcher, Reflector, Reporter,
Evaluator) from the "AI Research Agent" project, adapted here into a runnable
RAG-over-local-documents demo.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END

from core.state import ResearchState
from core.vector_store import VectorStore
from agents import planner, retriever, synthesizer, evaluator, reporter


def build_graph(store: VectorStore):
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner.run)
    graph.add_node("retriever", lambda s: retriever.run(s, store))
    graph.add_node("synthesizer", synthesizer.run)
    graph.add_node("evaluator", evaluator.run)
    graph.add_node("reporter", reporter.run)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", "evaluator")

    # Reflection loop: if the evaluator finds the draft insufficient (and we
    # haven't hit the revision cap), go back to retrieval with the critique
    # in context; otherwise move on to the final report.
    graph.add_conditional_edges(
        "evaluator",
        lambda s: "reporter" if s["is_sufficient"] else "retriever",
        {"reporter": "reporter", "retriever": "retriever"},
    )
    graph.add_edge("reporter", END)

    return graph.compile()


def run_research(query: str, store: VectorStore) -> ResearchState:
    app = build_graph(store)
    initial_state: ResearchState = {"query": query, "revision_count": 0}
    return app.invoke(initial_state)
