"""
VectorStore abstraction.

The resume's projects use Pinecone (managed, cloud) and ChromaDB/FAISS
(local/self-hosted). To keep this prototype runnable offline, in one
command, with zero infra, the DEFAULT backend here is a TF-IDF cosine
similarity index (scikit-learn) -- conceptually the same "embed + nearest
neighbor" retrieval pattern, just without needing an embeddings API key or
a running vector DB service.

Swapping to a real vector DB only requires implementing the same 3 methods
below against Pinecone/Chroma/FAISS's client -- the rest of the pipeline
(agents/retriever.py) is unaware of which backend is in use. This is the
same "swap the storage layer, keep the interface" pattern used for the
Pinecone -> in the internship RAG chatbot and FAISS -> in the multi-document
study assistant project.
"""
from __future__ import annotations
import glob
import os
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    source: str
    text: str


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Simple sliding-window chunking (stand-in for semantic chunking)."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks or [text]


class VectorStore:
    """Minimal interface every backend implements:
       - index(chunks)
       - query(text, k) -> List[Chunk with score]
    Real deployments would implement PineconeVectorStore / ChromaVectorStore
    / FAISSVectorStore against this same interface (see README).
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self._chunks: List[Chunk] = []

    def load_directory(self, directory: str) -> int:
        chunks: List[Chunk] = []
        for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            for piece in _chunk_text(raw):
                chunks.append(Chunk(source=os.path.basename(path), text=piece))
        self.index(chunks)
        return len(chunks)

    def index(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        texts = [c.text for c in chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    def query(self, text: str, k: int = 4):
        if self._matrix is None or not self._chunks:
            return []
        qvec = self._vectorizer.transform([text])
        sims = cosine_similarity(qvec, self._matrix)[0]
        ranked = sorted(zip(self._chunks, sims), key=lambda t: t[1], reverse=True)
        return [
            {"source": c.source, "text": c.text, "score": float(s)}
            for c, s in ranked[:k]
            if s > 0
        ]
