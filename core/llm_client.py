"""
Thin LLM wrapper.

Design goal: the SAME code path should work in two modes, controlled purely
by environment variables -- no code changes needed to go from a free local
demo to a production run against a real model.

  1. LIVE MODE   - OPENAI_API_KEY is set  -> calls OpenAI's Chat Completions API.
  2. OFFLINE MODE - no key set            -> uses a lightweight extractive/
                    template-based "mock LLM" so graders/reviewers can run
                    the whole pipeline with zero cost and zero external calls.

This mirrors how you'd swap Gemini/OpenAI in the internship project referenced
in the resume: the agent logic never talks to a vendor SDK directly, only to
this wrapper.
"""
from __future__ import annotations
import os
import re
import textwrap
from typing import List

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        _client = None


def is_live() -> bool:
    return _client is not None


def complete(system: str, user: str, temperature: float = 0.2) -> str:
    """Single entry point every agent uses to talk to 'the model'."""
    if _client is not None:
        resp = _client.chat.completions.create(
            model=MODEL_NAME,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

    # ---- OFFLINE fallback -------------------------------------------------
    return _offline_complete(system, user)


def _offline_complete(system: str, user: str) -> str:
    """
    Deterministic stand-in for an LLM call. It does simple extractive /
    template reasoning over the prompt so the multi-agent control flow
    (planning -> retrieval -> synthesis -> reflection -> report) can be
    demonstrated end-to-end without an API key.

    NOTE: this is a demo aid, not a claim that this is "as good as" a real
    LLM -- see README's "Offline mode" section.
    """
    if "Break the user's research question" in system:
        return _offline_plan(user)
    if "Write a well-cited draft answer" in system:
        return _offline_synthesize(user)
    if "critique" in system.lower():
        return _offline_critique(user)
    if "polished, well-structured research report" in system:
        return _offline_report(user)
    return "[offline-mode] No matching template for this prompt."


def _offline_plan(user: str) -> str:
    m = re.search(r"Research question:\s*(.+)", user)
    q = m.group(1).strip() if m else user.strip()
    topic = q.rstrip("?")
    angles = [
        f"Current state / how it works: {topic}",
        f"Key benefits or advantages: {topic}",
        f"Main risks, limitations, or tradeoffs: {topic}",
    ]
    return "\n".join(f"- {a}" for a in angles)


def _offline_synthesize(user: str) -> str:
    chunks = re.findall(r"\[(\d+)\]\s(.+)", user)
    if not chunks:
        return "No supporting context was retrieved for this sub-question."
    bullets = [f"- {textwrap.shorten(text, 160)} [{idx}]" for idx, text in chunks[:4]]
    return "Based on the retrieved context:\n" + "\n".join(bullets)


def _offline_critique(user: str) -> str:
    # Very rough heuristic: if the draft cites at least 2 sources per section, call it sufficient.
    citation_count = len(re.findall(r"\[\d+\]", user))
    if citation_count >= 3:
        return "SUFFICIENT"
    return "INSUFFICIENT: add more supporting citations before finalizing."


def _offline_report(user: str) -> str:
    return (
        "# Research Report (offline-mode draft)\n\n"
        "This report was generated in offline/template mode. Set OPENAI_API_KEY "
        "to produce a fully model-written narrative report.\n\n" + user
    )
