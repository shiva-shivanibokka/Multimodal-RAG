# backend/app/index/store.py
"""Per-session text index: FAISS (inner-product) dense search + BM25 sparse search,
plus a separate CLIP image index over every PAGE image for cross-modal
retrieval, plus a THIRD index — bge text embeddings of each page's OCR'd
text — for the caption-baseline A/B path (Task 3.3, extended to page level
by Task 5.2b).

Only chunks with non-empty ``text`` are indexed by ``dense``/``bm25`` (i.e.
"text" and "table" chunk kinds). Separately, when ``pages`` is supplied to
``add``, EVERY page's rendered image is indexed (ColPali-style whole-page
visual retrieval, Task 5.2b) — not just pages that happen to contain a
``figure`` chunk. This means a page with zero figure chunks (e.g. a dense
text page, or a page whose only "visual" content is layout) is still
retrievable by pixels. Two separate indexes are built per page:
  - a FAISS index of CLIP image embeddings (Task 3.2) — see ``cross_modal``,
    true cross-modal text->image retrieval on visual semantics.
  - a FAISS index of bge text embeddings of each page's OCR'd text (Task
    3.3) — see ``caption_baseline``, "fake multimodal" retrieval that only
    works when the page image contains legible text. A page whose OCR comes
    back empty simply isn't in this index.

Both indexes return synthetic PAGE results shaped like a chunk dict so
downstream code (hybrid fusion, answer.py) works unchanged:
    {"id": "page-<index>", "kind": "page", "text": "", "page": <index>,
     "bbox": [0, 0, width, height], "table_df_json": None,
     "caption_text": <OCR text, "" if none>}

Result shape (consumed by Task 2.3's hybrid fusion):
    [{"chunk": <chunk dict>, "score": float}, ...]   sorted by score descending

``embed_texts``/``embed_query`` (Task 2.1) already return L2-normalized
vectors, so a plain ``faiss.IndexFlatIP`` over them computes cosine
similarity directly — no re-normalization, no L2 distance here. Same applies
to ``embed_images``/``embed_query_clip`` (Task 3.1) in their own CLIP-space
index, and to the caption index below (bge space, like the main text index,
but kept as a SEPARATE FAISS index so page captions never pollute
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
        self._bm25_max_score: float = 0.0  # Task 7: see add()
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

            tokenized = [_tokenize(c["text"]) for c in indexable]
            self._bm25 = BM25Okapi(tokenized)
            # Task 7: raw BM25 scores aren't 0-1 scaled like cosine, so the
            # hybrid grounding gate (answer.py) can't compare them to
            # settings.retrieval_min_score directly. Estimate a corpus-scale
            # upper bound by self-scoring each document against its own
            # tokens (a query that matches a document exactly) and take the
            # max -- then bm25_normalized_top1() divides by this at query
            # time. O(n^2) in chunk count; fine for a single session's chunks.
            self._bm25_max_score = max(
                (self._bm25.get_scores(toks)[i] for i, toks in enumerate(tokenized)),
                default=0.0,
            )

        if pages:
            # One synthetic page-result dict per PAGE (not per figure chunk)
            # so every page image is retrievable, including pages with zero
            # figure chunks. OCR each page exactly once here (single loop),
            # then split into the image index (all pages) and the caption
            # index (only pages whose OCR came back non-empty).
            page_results = []
            images = []
            for p in pages:
                # Task 4: pages already OCR'd upstream (/ingest fills
                # needs_ocr pages' text_blocks; born-digital pages already
                # have their native text_blocks) carry the text we need
                # right here -- both OCR word dicts and native text-block
                # dicts expose {"text", "bbox"}, so joining "text" works
                # either way. Only fall back to a fresh ocr_page call when
                # text_blocks is missing/empty (direct Index.add callers/
                # tests that build page dicts without going through
                # /ingest), so a page isn't OCR'd a 3rd time here.
                text_blocks = p.get("text_blocks")
                if text_blocks:
                    caption_text = " ".join(b["text"] for b in text_blocks if b.get("text"))
                else:
                    words = ocr_page(p["image_png"])
                    caption_text = " ".join(w["text"] for w in words)
                page_results.append(
                    {
                        "id": f"page-{p['index']}",
                        "kind": "page",
                        "text": "",
                        "page": p["index"],
                        "bbox": [0, 0, p["width"], p["height"]],
                        "table_df_json": None,
                        "caption_text": caption_text,
                    }
                )
                images.append(p["image_png"])

            self._image_chunks = page_results
            img_vecs = embed_images(images)
            self._image_index = faiss.IndexFlatIP(img_vecs.shape[1])
            self._image_index.add(img_vecs)

            captioned = [pr for pr in page_results if pr["caption_text"]]
            if captioned:
                self._caption_chunks = captioned
                cap_vecs = embed_texts([pr["caption_text"] for pr in captioned])
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

    def bm25_normalized_top1(self, query: str) -> float:
        """Top-1 BM25 score normalized (roughly) to [0, 1] by the corpus's
        own max self-score (see ``add``) -- lets the hybrid grounding gate
        (Task 7, app/generate/answer.py) compare a lexical signal against
        the same 0-1 threshold used for dense cosine. Returns 0.0 if there's
        no BM25 index or no usable normalizer (avoids ZeroDivisionError)."""
        if self._bm25 is None or not self._bm25_max_score:
            return 0.0
        top = self.bm25(query, 1)
        if not top:
            return 0.0
        return top[0]["score"] / self._bm25_max_score

    def cross_modal(self, query: str, k: int) -> list[dict]:
        """Text -> image retrieval: embed ``query`` with CLIP and search the
        page-image index (every page, not just pages with a figure chunk).
        Top-k by cosine similarity, descending."""
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
        space as ``dense``) and search the page-caption index — pages
        retrieved via their OCR'd text, not their pixels. A page whose OCR
        came back empty (e.g. a pure-visual page with no text) is not in
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
