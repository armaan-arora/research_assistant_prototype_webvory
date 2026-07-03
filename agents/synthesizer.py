from core.llm_client import complete
from core.state import ResearchState

SYSTEM = """You are the Synthesizer agent. Write a well-cited draft answer to the
research question using ONLY the numbered context snippets provided. Cite each
claim with its bracket number, e.g. [2]. If the context is insufficient, say so
explicitly instead of guessing."""


def run(state: ResearchState) -> ResearchState:
    retrieved = state.get("retrieved", [])
    context_block = "\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(retrieved)
    )
    user = (
        f"Research question: {state['query']}\n\n"
        f"Context:\n{context_block if context_block else '(no context retrieved)'}"
    )
    state["draft_answer"] = complete(SYSTEM, user)
    return state
