"""
Shared state object passed between agents in the LangGraph workflow.

Using a Pydantic model (rather than a raw dict) gives us type-safe state
transitions and catches malformed agent outputs before they propagate --
the same pattern used in the "AI Research Agent" project on the resume.
"""
from __future__ import annotations
from typing import List, Optional, TypedDict


class SourceChunk(TypedDict):
    source: str
    text: str
    score: float


class ResearchState(TypedDict, total=False):
    # Input
    query: str

    # Planner output
    sub_questions: List[str]

    # Retriever output
    retrieved: List[SourceChunk]

    # Synthesizer output
    draft_answer: str

    # Evaluator output
    is_sufficient: bool
    critique: Optional[str]
    revision_count: int

    # Reporter output
    final_report: str
    citations: List[str]
