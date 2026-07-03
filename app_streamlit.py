"""
Streamlit demo UI.

Run:
    streamlit run app_streamlit.py

Lets a reviewer type a research question, see the planner's sub-questions,
the retrieved sources, and the final synthesized + reflected report - i.e.
a visual walkthrough of the whole agent graph for the demo video.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.vector_store import VectorStore
from core.llm_client import is_live
from graph import run_research

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎", layout="wide")

st.title("🔎 AI Research Assistant")
st.caption(
    "Multi-agent RAG pipeline: Planner -> Retriever -> Synthesizer -> "
    "Evaluator (reflection loop) -> Reporter, built with LangGraph."
)

mode = "🟢 LIVE (OpenAI API)" if is_live() else "🟡 OFFLINE (template fallback, no API key)"
st.info(f"Mode: {mode}")


@st.cache_resource
def get_store():
    store = VectorStore()
    n = store.load_directory("data")
    return store, n


store, n_chunks = get_store()
st.caption(f"Knowledge base: {n_chunks} chunks indexed from ./data")

query = st.text_input(
    "Research question",
    value="What are the tradeoffs between different vector databases for RAG systems?",
)

if st.button("Run research", type="primary") and query:
    with st.spinner("Running multi-agent pipeline..."):
        result = run_research(query, store)

    st.subheader("1. Planner - sub-questions")
    for q in result.get("sub_questions", []):
        st.markdown(f"- {q}")

    st.subheader("2. Retriever - top sources")
    for i, c in enumerate(result.get("retrieved", []), start=1):
        with st.expander(f"[{i}] {c['source']} (score: {c['score']:.3f})"):
            st.write(c["text"])

    st.subheader("3-4. Synthesizer + Evaluator (reflection loop)")
    st.caption(f"Revisions triggered: {result.get('revision_count', 0)}")
    st.write(result.get("draft_answer", ""))

    st.subheader("5. Reporter - final report")
    st.markdown(result.get("final_report", ""))
