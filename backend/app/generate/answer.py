# backend/app/generate/answer.py
"""Retrieve from the session index, gate on grounding, generate, cite.

Grounding gate: we threshold on the top DENSE cosine score
(``Index.dense`` -- a plain FAISS inner product over bge-small's
L2-normalized vectors, i.e. a stable 0-1 cosine similarity). We deliberately
do NOT threshold on the ``retrieve()`` result score: in "hybrid" mode that's
an RRF fusion score (sum of ``1/(60+rank)`` terms, tiny and not on a 0-1
scale) and, when reranking is on, a bge cross-encoder logit (unbounded) --
neither is comparable to ``settings.retrieval_min_score``.
"""
from app.config import settings
from app.generate.providers import generate
from app.retrieve.hybrid import retrieve
from app.schemas import AnswerRequest, AnswerResponse, Citation
from app.session import get_index

_SNIPPET_LEN = 150
_SUPPORTED_MODES = {"hybrid", "dense"}  # ponytail: Phase 3 adds cross_modal/caption_baseline
_NOT_IN_DOCUMENTS = "NOT_IN_DOCUMENTS"
_SYSTEM_PROMPT = (
    "Answer ONLY using the provided context. If the context does not "
    f"contain the answer, reply exactly: {_NOT_IN_DOCUMENTS}."
)


def _refuse() -> AnswerResponse:
    return AnswerResponse(answer="", refused=True)


def answer_question(req: AnswerRequest) -> AnswerResponse:
    if not req.session_id:
        return _refuse()

    index = get_index(req.session_id)
    if index is None:
        return _refuse()

    top = index.dense(req.question, 1)
    if not top or top[0]["score"] < settings.retrieval_min_score:
        return _refuse()

    mode = req.retrieval_mode if req.retrieval_mode in _SUPPORTED_MODES else "hybrid"
    results = retrieve(index, req.question, mode=mode, k=5, use_rerank=True)

    context = "\n".join(f"[page {r['chunk']['page']}] {r['chunk']['text']}" for r in results)
    user = f"Context:\n{context}\n\nQuestion: {req.question}"
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    text = generate(req.provider, req.model, req.api_key, messages)

    if text.strip() == _NOT_IN_DOCUMENTS:
        return _refuse()

    citations = [
        Citation(
            page=r["chunk"]["page"],
            bbox=r["chunk"]["bbox"],
            snippet=r["chunk"]["text"][:_SNIPPET_LEN],
        )
        for r in results
    ]
    return AnswerResponse(answer=text, refused=False, citations=citations)
