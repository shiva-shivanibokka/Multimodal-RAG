# backend/app/generate/answer.py
"""Retrieve from the session index, gate on grounding, generate, cite.

Mode-aware grounding gate (Task 3.4): each retrieval mode has its own
primary index with its own cosine-similarity scale, so the gate reads the
top-1 score from THAT mode's index, not always ``Index.dense``:
  - hybrid/dense -> ``Index.dense(question, 1)``       (bge cosine, bounded 0-1)
  - cross_modal  -> ``Index.cross_modal(question, 1)``  (CLIP cosine, bounded 0-1)
  - caption_baseline -> ``Index.caption_baseline(question, 1)`` (bge cosine, bounded 0-1)
All three are FAISS inner-product over L2-normalized vectors, so all are
directly comparable to ``settings.retrieval_min_score``. We deliberately do
NOT threshold on the ``retrieve()`` result score: in "hybrid" mode that's an
RRF fusion score (tiny, not 0-1 scaled) and, when reranking is on, a bge
cross-encoder logit (unbounded) -- neither is comparable to the threshold.
Refuse (no LLM call) if the mode's top result is empty or below threshold.

Figure images to the VLM (Task 3.4): any retrieved chunk with
``kind == "figure"`` has no passage text (see store.py), so its whole page
image is looked up from the session's stored pages (same
``pages_by_index`` keying used in store.py), base64-encoded, and passed via
``generate(..., images=[...])`` so the VLM actually sees it. Text/table
chunks are concatenated into the context string as before. A single result
set can contain both kinds.
"""
import base64

from app.config import settings
from app.generate.providers import generate
from app.retrieve.hybrid import retrieve
from app.schemas import AnswerRequest, AnswerResponse, Citation
from app.session import get_index, get_session

_SNIPPET_LEN = 150
_SUPPORTED_MODES = {"hybrid", "dense", "cross_modal", "caption_baseline"}
_NOT_IN_DOCUMENTS = "NOT_IN_DOCUMENTS"
_SYSTEM_PROMPT = (
    "Answer ONLY using the provided context. If the context does not "
    f"contain the answer, reply exactly: {_NOT_IN_DOCUMENTS}."
)


def _refuse() -> AnswerResponse:
    return AnswerResponse(answer="", refused=True)


def _grounding_top(index, mode: str, question: str) -> list[dict]:
    """Top-1 result from the mode's own primary index -- used only for the
    grounding gate, not the full ``retrieve()`` (see module docstring)."""
    if mode == "cross_modal":
        return index.cross_modal(question, 1)
    if mode == "caption_baseline":
        return index.caption_baseline(question, 1)
    return index.dense(question, 1)


def answer_question(req: AnswerRequest) -> AnswerResponse:
    if not req.session_id:
        return _refuse()

    index = get_index(req.session_id)
    if index is None:
        return _refuse()

    mode = req.retrieval_mode if req.retrieval_mode in _SUPPORTED_MODES else "hybrid"

    top = _grounding_top(index, mode, req.question)
    if not top or top[0]["score"] < settings.retrieval_min_score:
        return _refuse()

    results = retrieve(index, req.question, mode=mode, k=5, use_rerank=True)

    session = get_session(req.session_id)
    pages_by_index = {p["index"]: p for p in (session["pages"] if session else [])}

    context_lines = []
    images = []
    citations = []
    for r in results:
        chunk = r["chunk"]
        if chunk.get("kind") == "figure":
            page = pages_by_index.get(chunk["page"])
            if page is not None:
                images.append(base64.b64encode(page["image_png"]).decode("ascii"))
            snippet = chunk.get("caption_text") or "[figure]"
            citations.append(
                Citation(page=chunk["page"], bbox=chunk["bbox"], snippet=snippet[:_SNIPPET_LEN])
            )
        else:
            context_lines.append(f"[page {chunk['page']}] {chunk['text']}")
            citations.append(
                Citation(page=chunk["page"], bbox=chunk["bbox"], snippet=chunk["text"][:_SNIPPET_LEN])
            )

    context = "\n".join(context_lines)
    user = f"Context:\n{context}\n\nQuestion: {req.question}"
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    text = generate(req.provider, req.model, req.api_key, messages, images=images)

    if text.strip() == _NOT_IN_DOCUMENTS:
        return _refuse()

    return AnswerResponse(answer=text, refused=False, citations=citations)
