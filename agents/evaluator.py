from core.llm_client import complete
from core.state import ResearchState

SYSTEM = """You are the Evaluator (reflection) agent. Read the draft answer and
critique whether it is well-grounded and sufficiently supported by citations.
Reply with exactly 'SUFFICIENT' if it is good enough to finalize, otherwise
reply 'INSUFFICIENT: <one sentence reason>'."""

MAX_REVISIONS = 2


def run(state: ResearchState) -> ResearchState:
    verdict = complete(SYSTEM, f"Draft answer:\n{state.get('draft_answer', '')}")
    state["revision_count"] = state.get("revision_count", 0)
    if verdict.strip().upper().startswith("SUFFICIENT"):
        state["is_sufficient"] = True
        state["critique"] = None
    else:
        state["is_sufficient"] = state["revision_count"] >= MAX_REVISIONS
        state["critique"] = verdict
        state["revision_count"] += 1
    return state
