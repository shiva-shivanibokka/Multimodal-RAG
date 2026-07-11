# backend/app/index/store.py
"""Per-session text index: FAISS (inner-product) dense search + BM25 sparse search.

Only chunks with non-empty ``text`` are indexed (i.e. "text" and "table"
chunk kinds — empty-text "figure" chunks get a CLIP image index in Phase 3,
not here).

Result shape (consumed by Task 2.3's hybrid fusion):
    [{"chunk": <chunk dict>, "score": float}, ...]   sorted by score descending

``embed_texts``/``embed_query`` (Task 2.1) already return L2-normalized
vectors, so a plain ``faiss.IndexFlatIP`` over them computes cosine
similarity directly — no re-normalization, no L2 distance here.
"""
import re

import faiss
from rank_bm25 import BM25Okapi

from app.index.embedders import embed_query, embed_texts

# ponytail: lowercase + \w+ split is good enough for BM25 term matching;
# swap for a real tokenizer only if retrieval quality demands it.
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Index:
    """Holds one session's dense (FAISS IP) + sparse (BM25) text index."""

    def __init__(self):
        self._chunks: list[dict] = []
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[dict]) -> None:
        indexable = [c for c in chunks if c.get("text")]
        if not indexable:
            return
        self._chunks = indexable

        vecs = embed_texts([c["text"] for c in indexable])
        self._faiss_index = faiss.IndexFlatIP(vecs.shape[1])
        self._faiss_index.add(vecs)

        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in indexable])

    def dense(self, query: str, k: int) -> list[dict]:
        """Top-k by cosine similarity (FAISS inner product on normalized vectors)."""
        if self._faiss_index is None:
            return []
        q = embed_query(query).reshape(1, -1)
        k = min(k, len(self._chunks))
        scores, idxs = self._faiss_index.search(q, k)
        return [
            {"chunk": self._chunks[i], "score": float(s)}
            for s, i in zip(scores[0], idxs[0])
            if i != -1
        ]

    def bm25(self, query: str, k: int) -> list[dict]:
        """Top-k by BM25 score, descending."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{"chunk": self._chunks[i], "score": float(scores[i])} for i in ranked]
