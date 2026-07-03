# AI Research Assistant — Multi-Agent RAG Prototype

A working prototype for the **AI Research Assistant** business workflow: given a
research question, a graph of specialized agents plans sub-questions, retrieves
grounded context from a document knowledge base, drafts a cited answer,
self-critiques it, and produces a polished final report.

Built with **LangGraph + LangChain-core + OpenAI API**, matching the stack used
in the accompanying resume's "AI Research Agent" and "Intelligent Study
Assistant" projects.

## Architecture

![architecture](architecture_diagram.png)

| Agent | Responsibility |
|---|---|
| **Planner** | Breaks the research question into 3 focused sub-questions |
| **Retriever** | Runs similarity search against the vector store for each sub-question |
| **Synthesizer** | Drafts a citation-grounded answer from retrieved chunks only |
| **Evaluator** | Reflects on the draft; loops back to Retriever if under-supported (max 2 revisions) |
| **Reporter** | Formats the final answer into a structured report with a sources list |

State is passed between agents as a typed `ResearchState` object (`core/state.py`),
not a raw dict — malformed agent outputs fail fast instead of propagating silently.

## Two run modes, same code path

| | Offline (default) | Live |
|---|---|---|
| Trigger | no `OPENAI_API_KEY` set | `OPENAI_API_KEY` set in `.env` |
| LLM calls | deterministic template fallback in `core/llm_client.py` | real OpenAI Chat Completions |
| Cost | $0 | standard OpenAI token pricing |
| Use for | grading / demo without API keys | production-quality answers |

This lets a reviewer clone the repo and run the entire pipeline immediately with
zero setup cost, while the same code is production-ready once a key is added.

## Vector store

The default backend is a TF-IDF cosine-similarity index (`core/vector_store.py`)
over `./data/*.txt` — no external service, no embeddings API cost, deterministic
results for grading. It implements the same `index()` / `query()` interface a
real Pinecone, ChromaDB, or FAISS backend would — swapping the backend is a
~20-line change isolated to that one file; no agent code changes.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add OPENAI_API_KEY to run in live mode
```

## Run it

**CLI:**
```bash
python main.py "What are the tradeoffs between different vector databases for RAG systems?"
```

**API:**
```bash
uvicorn api:app --reload --port 8000
curl -X POST http://localhost:8000/research -H "Content-Type: application/json" \
     -d '{"query": "How does hybrid retrieval improve RAG relevance?"}'
```

**UI (for the demo video):**
```bash
streamlit run app_streamlit.py
```

## Project structure

```
research_assistant/
├── agents/
│   ├── planner.py         # sub-question decomposition
│   ├── retriever.py       # vector store queries per sub-question
│   ├── synthesizer.py     # cited draft answer
│   ├── evaluator.py       # reflection / self-critique loop
│   └── reporter.py        # final structured report
├── core/
│   ├── state.py           # typed ResearchState schema
│   ├── llm_client.py      # OpenAI wrapper + offline fallback
│   └── vector_store.py    # pluggable TF-IDF vector store
├── data/                  # sample knowledge base (.txt files)
├── graph.py                # LangGraph StateGraph wiring + reflection loop
├── main.py                  # CLI entrypoint
├── api.py                   # FastAPI service
├── app_streamlit.py          # Streamlit demo UI
└── requirements.txt
```

## Notes / honest limitations

- The offline LLM fallback is a template/extractive stand-in for grading without
  API cost — it demonstrates the **control flow** (planning, retrieval, reflection
  loop, reporting) correctly, but the prose quality is naturally far below a real
  model. Set `OPENAI_API_KEY` for representative output quality.
- The TF-IDF vector store is a lightweight, dependency-free stand-in for a real
  embedding-based vector database; see `recommendation_report.docx` for the
  production architecture (Pinecone/ChromaDB) and why.
- No authentication/rate-limiting is implemented; see the report's "Risks &
  Limitations" section for what a production deployment would add.
