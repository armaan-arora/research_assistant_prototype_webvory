from core.llm_client import complete
from core.state import ResearchState

SYSTEM = """You are the Reporter agent. Turn the draft answer and its sources into
a polished, well-structured research report with a short executive summary,
key findings as bullet points, and a "Sources" section listing each citation."""


def run(state: ResearchState) -> ResearchState:
    sources = state.get("retrieved", [])
    citations = [f"[{i+1}] {c['source']}" for i, c in enumerate(sources)]
    user = (
        f"Research question: {state['query']}\n\n"
        f"Draft answer:\n{state.get('draft_answer', '')}\n\n"
        f"Sources:\n" + "\n".join(citations)
    )
    state["final_report"] = complete(SYSTEM, user)
    state["citations"] = citations
    return state
