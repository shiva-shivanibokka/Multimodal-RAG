# backend/app/index/store.py
"""Per-session text index: FAISS (inner-product) dense search + BM25 sparse search,
plus a separate CLIP image index over ``figure`` chunks for cross-modal retrieval,
plus a THIRD index — bge text embeddings of OCR'd figure captions — for the
caption-baseline A/B path (Task 3.3).

Only chunks with non-empty ``text`` are indexed by ``dense``/``bm25`` (i.e.
"text" and "table" chunk kinds). Empty-text "figure" chunks instead go into
two separate indexes:
  - a FAISS index of CLIP image embeddings (Task 3.2) — see ``cross_modal``,
    true cross-modal text->image retrieval on visual semantics.
  - a FAISS index of bge text embeddings of each figure's OCR'd caption text
    (Task 3.3) — see ``caption_baseline``, "fake multimodal" retrieval that
    only works when the figure's page image contains legible text. A figure
    whose OCR caption comes back empty simply isn't in this index.

Result shape (consumed by Task 2.3's hybrid fusion):
    [{"chunk": <chunk dict>, "score": float}, ...]   sorted by score descending

``embed_texts``/``embed_query`` (Task 2.1) already return L2-normalized
vectors, so a plain ``faiss.IndexFlatIP`` over them computes cosine
similarity directly — no re-normalization, no L2 distance here. Same applies
to ``embed_images``/``embed_query_clip`` (Task 3.1) in their own CLIP-space
index, and to the caption index below (bge space, like the main text index,
but kept as a SEPARATE FAISS index so figure captions never pollute
``dense()`` results over text/table chunks).
"""
import re

import faiss
from rank_bm25 import BM25Okapi

from app.index.embedders import embed_images, embed_query, embed_query_clip, embed_texts
from app.ingest.ocr import ocr_page

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
        self._image_chunks: list[dict] = []
        self._image_index: faiss.IndexFlatIP | None = None
        self._caption_chunks: list[dict] = []
        self._caption_index: faiss.IndexFlatIP | None = None

    def add(self, chunks: list[dict], pages=None) -> None:
        indexable = [c for c in chunks if c.get("text")]
        if indexable:
            self._chunks = indexable

            vecs = embed_texts([c["text"] for c in indexable])
            self._faiss_index = faiss.IndexFlatIP(vecs.shape[1])
            self._faiss_index.add(vecs)

            self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in indexable])

        if pages is not None:
            # ``pages`` is a list of page dicts, each carrying its own
            # ``index`` (see ingest/chunk.py) — key by that rather than
            # list position, so it's correct whether or not pages happen
            # to already be in index order.
            pages_by_index = {p["index"]: p for p in pages}
            figures = [c for c in chunks if c.get("kind") == "figure"]
            if figures:
                self._image_chunks = figures
                images = [pages_by_index[c["page"]]["image_png"] for c in figures]
                img_vecs = embed_images(images)
                self._image_index = faiss.IndexFlatIP(img_vecs.shape[1])
                self._image_index.add(img_vecs)

                # ponytail: cache OCR per page index within this call — several
                # figures can share a page, and OCR is the expensive step.
                ocr_cache: dict[int, str] = {}
                for c in figures:
                    page_idx = c["page"]
                    if page_idx not in ocr_cache:
                        words = ocr_page(pages_by_index[page_idx]["image_png"])
                        ocr_cache[page_idx] = " ".join(w["text"] for w in words)
                    c["caption_text"] = ocr_cache[page_idx]

                captioned = [c for c in figures if c["caption_text"]]
                if captioned:
                    self._caption_chunks = captioned
                    cap_vecs = embed_texts([c["caption_text"] for c in captioned])
                    self._caption_index = faiss.IndexFlatIP(cap_vecs.shape[1])
                    self._caption_index.add(cap_vecs)

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

    def cross_modal(self, query: str, k: int) -> list[dict]:
        """Text -> image retrieval: embed ``query`` with CLIP and search the
        image (figure chunk) index. Top-k by cosine similarity, descending."""
        if self._image_index is None:
            return []
        q = embed_query_clip(query).reshape(1, -1)
        k = min(k, len(self._image_chunks))
        scores, idxs = self._image_index.search(q, k)
        return [
            {"chunk": self._image_chunks[i], "score": float(s)}
            for s, i in zip(scores[0], idxs[0])
            if i != -1
        ]

    def caption_baseline(self, query: str, k: int) -> list[dict]:
        """"Caption-then-embed" A/B baseline: embed ``query`` with bge (same
        space as ``dense``) and search the figure-caption index — figures
        retrieved via their OCR'd text, not their pixels. A figure whose OCR
        caption was empty (e.g. a pure-visual image with no text) is not in
        this index and can never be returned here, unlike ``cross_modal``."""
        if self._caption_index is None:
            return []
        q = embed_query(query).reshape(1, -1)
        k = min(k, len(self._caption_chunks))
        scores, idxs = self._caption_index.search(q, k)
        return [
            {"chunk": self._caption_chunks[i], "score": float(s)}
            for s, i in zip(scores[0], idxs[0])
            if i != -1
        ]
