"""
CLI demo: builds a local knowledge base from ./data/*.txt, then runs the
Planner -> Retriever -> Synthesizer -> Evaluator -> Reporter graph against a
research question.

Usage:
    python main.py "How do RAG systems reduce hallucination?"

Runs fully offline (no API key needed) using the template-based LLM fallback,
or in "live" mode against real OpenAI models if OPENAI_API_KEY is set in the
environment / .env file.
"""
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from core.vector_store import VectorStore
from core.llm_client import is_live
from graph import run_research


def main():
    query = " ".join(sys.argv[1:]) or (
        "What are the tradeoffs between different vector databases for RAG systems?"
    )

    print(f"Mode: {'LIVE (OpenAI API)' if is_live() else 'OFFLINE (template fallback)'}")
    print(f"Research question: {query}\n")

    store = VectorStore()
    n_chunks = store.load_directory("data")
    print(f"Indexed {n_chunks} chunks from ./data\n")

    t0 = time.time()
    result = run_research(query, store)
    elapsed = time.time() - t0

    print("=" * 70)
    print("SUB-QUESTIONS")
    print("=" * 70)
    for q in result.get("sub_questions", []):
        print(f"  - {q}")

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(result.get("final_report", "(no report generated)"))

    print("\n" + "=" * 70)
    print(f"Revisions triggered by reflection loop: {result.get('revision_count', 0)}")
    print(f"Chunks retrieved: {len(result.get('retrieved', []))}")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
