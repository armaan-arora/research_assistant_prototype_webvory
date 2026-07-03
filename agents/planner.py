from core.llm_client import complete
from core.state import ResearchState

SYSTEM = """You are the Planner agent in a multi-agent research system.
Break the user's research question into 3 focused sub-questions that together
cover the topic comprehensively (current state, benefits, risks/limitations).
Return ONLY a bullet list, one sub-question per line."""


def run(state: ResearchState) -> ResearchState:
    raw = complete(SYSTEM, f"Research question: {state['query']}")
    sub_questions = [
        line.strip("- ").strip() for line in raw.splitlines() if line.strip()
    ]
    state["sub_questions"] = sub_questions[:3] or [state["query"]]
    return state
