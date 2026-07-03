from core.state import ResearchState
from core.vector_store import VectorStore


def run(state: ResearchState, store: VectorStore, k: int = 4) -> ResearchState:
    retrieved = []
    seen = set()
    for sub_q in state.get("sub_questions", [state["query"]]):
        for hit in store.query(sub_q, k=k):
            key = (hit["source"], hit["text"][:60])
            if key in seen:
                continue
            seen.add(key)
            retrieved.append(hit)
    # keep the strongest matches overall
    retrieved.sort(key=lambda h: h["score"], reverse=True)
    state["retrieved"] = retrieved[: k * 2]
    return state
