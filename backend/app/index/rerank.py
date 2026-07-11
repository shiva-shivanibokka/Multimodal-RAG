# backend/app/index/rerank.py
"""Cross-encoder reranking (bge-reranker-base), CPU only.

``rerank`` takes fusion/dense candidates (``[{"chunk", "score"}, ...]``) and
re-scores each ``(query, chunk["text"])`` pair with a
``sentence_transformers.CrossEncoder``. The cross-encoder sees query and
passage together (unlike bi-encoder dense search), so it's slower but more
accurate — used as a final top-k reorder over a small candidate pool, not for
first-stage retrieval over the whole index.
"""
from sentence_transformers import CrossEncoder

from app.config import settings

_model = None  # ponytail: module-level lazy singleton, loaded once on first use


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(settings.reranker_model, device="cpu")
    return _model


def rerank(query: str, results: list[dict], k: int) -> list[dict]:
    """Re-score ``results`` with the cross-encoder and return the top-k.

    Returns ``[{"chunk": ..., "score": <rerank score>, "fusion_score": <prior
    score>}, ...]`` sorted by rerank score descending, length <= k.
    """
    if not results:
        return []
    model = _get_model()
    pairs = [(query, r["chunk"]["text"]) for r in results]
    rerank_scores = model.predict(pairs)

    reranked = [
        {"chunk": r["chunk"], "score": float(score), "fusion_score": r["score"]}
        for r, score in zip(results, rerank_scores)
    ]
    reranked.sort(key=lambda r: r["score"], reverse=True)
    return reranked[:k]
