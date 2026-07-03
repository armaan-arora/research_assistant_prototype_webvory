"""
FastAPI service exposing the multi-agent research pipeline as a REST API,
so it can be integrated into a larger business workflow (e.g. called from
an internal tool, Slack bot, or a Streamlit frontend).

Run:
    uvicorn api:app --reload --port 8000

Then:
    POST /research   {"query": "..."}
    GET  /health
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from core.vector_store import VectorStore
from core.llm_client import is_live
from graph import run_research

app = FastAPI(
    title="AI Research Assistant API",
    description="Multi-agent RAG research pipeline (Planner -> Retriever -> "
                 "Synthesizer -> Evaluator -> Reporter) built on LangGraph.",
    version="1.0.0",
)

# Build the vector index once at startup rather than per-request.
_store = VectorStore()
_store.load_directory("data")


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    query: str
    sub_questions: list[str]
    final_report: str
    citations: list[str]
    revisions: int


@app.get("/health")
def health():
    return {"status": "ok", "mode": "live" if is_live() else "offline"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    result = run_research(req.query, _store)
    return ResearchResponse(
        query=req.query,
        sub_questions=result.get("sub_questions", []),
        final_report=result.get("final_report", ""),
        citations=result.get("citations", []),
        revisions=result.get("revision_count", 0),
    )
