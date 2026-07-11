# backend/app/retrieve/hybrid.py
"""Hybrid retrieval: RRF fusion of dense + BM25, optional cross-encoder rerank.
Plus the two image/figure modes (Task 3.4): cross_modal and caption_baseline.

Result shape (all modes):
    [{"chunk": <chunk dict>, "score": float, ...}, ...]   descending, len <= k
When rerank is applied (text modes only), each dict also carries
``fusion_score`` (see ``rerank.rerank``); without rerank, ``score`` is the
fusion (or dense) score.
"""
from app.index.rerank import rerank as _rerank

# Reciprocal Rank Fusion constant. Standard default from the RRF paper
# (Cormack et al. 2009) — de-emphasizes noisy top-1 flukes from either
# retriever while still rewarding high rank. Not tuned for this corpus.
RRF_C = 60

# Pull a wider candidate pool from each retriever before fusing/reranking,
# so a chunk that's e.g. rank 8 in dense but rank 1 in bm25 still surfaces.
_MIN_CANDIDATE_POOL = 20

# Image/figure modes: the bge cross-encoder reranker scores (query, passage
# TEXT) pairs. cross_modal figure chunks carry no text at all (``text==""``,
# see store.py); caption_baseline figure chunks carry OCR caption text, not
# prose the cross-encoder was trained to judge against a question. Neither
# is a text passage, so these two modes skip reranking entirely and return
# the index's own top-k directly, regardless of ``use_rerank``.
_FIGURE_MODES = {"cross_modal", "caption_baseline"}


def _rrf_fuse(dense_results: list[dict], bm25_results: list[dict]) -> list[dict]:
    """Reciprocal Rank Fusion: rrf_score = sum over retrievers of 1/(RRF_C + rank),
    rank is 1-based. Dedupes by chunk id, keeping one chunk dict per id."""
    scores: dict = {}
    chunks: dict = {}
    for results in (dense_results, bm25_results):
        for rank, r in enumerate(results, start=1):
            cid = r["chunk"]["id"]
            chunks[cid] = r["chunk"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_C + rank)

    fused = [{"chunk": chunks[cid], "score": score} for cid, score in scores.items()]
    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused


def retrieve(
    index, query: str, mode: str = "hybrid", k: int = 5, use_rerank: bool = True
) -> list[dict]:
    """Retrieve top-k chunks for ``query``.

    mode="dense": dense-only search.
    mode="hybrid": RRF-fuse dense + BM25 candidate pools.
    mode="cross_modal": CLIP text->image search over figure chunks
        (``Index.cross_modal``) — no rerank, see ``_FIGURE_MODES``.
    mode="caption_baseline": bge search over OCR'd figure captions
        (``Index.caption_baseline``) — no rerank, see ``_FIGURE_MODES``.
    Any other/unknown mode falls back to "hybrid".

    For text modes (dense/hybrid), if use_rerank, the candidate pool is
    re-scored/re-sorted by the bge cross-encoder (see app.index.rerank) and
    truncated to k; otherwise the fusion (or dense) top-k is returned
    directly.
    """
    if mode in _FIGURE_MODES:
        method = index.cross_modal if mode == "cross_modal" else index.caption_baseline
        return method(query, k)

    pool = max(k * 4, _MIN_CANDIDATE_POOL)

    if mode == "dense":
        candidates = index.dense(query, pool)
    else:
        dense_results = index.dense(query, pool)
        bm25_results = index.bm25(query, pool)
        candidates = _rrf_fuse(dense_results, bm25_results)

    if use_rerank:
        return _rerank(query, candidates, k)
    return candidates[:k]
