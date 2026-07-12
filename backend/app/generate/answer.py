# backend/app/generate/answer.py
"""Retrieve from the session index, gate on grounding, generate, cite.

Mode-aware grounding gate (Task 3.4): each retrieval mode has its own
primary index with its own cosine-similarity scale, so the gate reads the
top-1 score from THAT mode's index, not always ``Index.dense``:
  - dense        -> ``Index.dense(question, 1)``        (bge cosine, bounded 0-1)
  - cross_modal  -> ``Index.cross_modal(question, 1)``  (CLIP cosine, bounded 0-1)
  - caption_baseline -> ``Index.caption_baseline(question, 1)`` (bge cosine, bounded 0-1)
  - hybrid       -> ``max(Index.dense(question, 1), Index.bm25_normalized_top1(question))``
    (Task 7: a short exact-keyword query can score low on dense cosine while
    still being a strong lexical hit -- BM25's raw score is normalized to
    roughly [0, 1] by the corpus's own max self-score, see store.py, so it's
    comparable to the same threshold.)
All are directly comparable to ``settings.retrieval_min_score``. We
deliberately do NOT threshold on the ``retrieve()`` result score: in
"hybrid" mode that's an RRF fusion score (tiny, not 0-1 scaled) and, when
reranking is on, a bge cross-encoder logit (unbounded) -- neither is
comparable to the threshold. Refuse (no LLM call) if the mode's gate score
is below threshold.

Figure/page images to the VLM (Task 3.4, extended to whole pages by Task
5.2b): any retrieved chunk with ``kind == "figure"`` or ``kind == "page"``
has no passage text (see store.py), so its whole page image is looked up
from the session's stored pages (same ``pages_by_index`` keying used in
store.py), base64-encoded, and passed via ``generate(..., images=[...])``
so the VLM actually sees it. Text/table chunks are concatenated into the
context string as before. A single result set can contain any mix of kinds.

Deterministic table answers (Task 4.3): right after ``results`` is computed
(and the grounding gate above has already passed), ``try_table_answer`` gets
first look. If the top retrieved chunk is a table and the question is a
numeric aggregate over one of its columns, it returns a computed, exact
``AnswerResponse`` immediately -- ``generate()`` and NLI are both skipped,
because a value read straight out of the source DataFrame needs no LLM
drafting and no faithfulness check. Otherwise (no aggregate keyword, no
confident column match, or nothing numeric survives coercion) it returns
None and the normal LLM path below runs unchanged.

Faithfulness firewall (Task 4.2): when ``req.verified`` (default True),
every generated answer is split into claims and each is checked against the
retrieved evidence via ``verify_claims`` (Task 4.1's NLI gate). If NOT A
SINGLE claim is supported, the answer is a hallucination end-to-end and we
override ``refused=True`` -- this is the calibrated-refusal firewall. We
deliberately KEEP ``answer=text`` (the model's raw draft) instead of
blanking it: the frontend needs the actual text to red-flag it ("the model
produced this, but none of it is grounded"), and discarding it would just
push the user to distrust an empty response instead of an informative one.
Top-level ``citations`` become the union of citations from only the
SUPPORTED claims (deduped by (page, bbox)) -- unsupported claims must not
lend their (bogus) provenance to the response. When ``req.verified`` is
False, NLI never runs (it's slow on CPU) and behavior is unchanged from
Task 3.4: claims=[], citations come from the retrieved chunks.
"""
import base64

from app.config import settings
from app.generate.providers import generate
from app.generate.table_answer import try_table_answer
from app.retrieve.hybrid import retrieve
from app.schemas import AnswerRequest, AnswerResponse, Citation
from app.session import get_index, get_session
from app.verify.nli import verify_claims

_SNIPPET_LEN = 150
_SUPPORTED_MODES = {"hybrid", "dense", "cross_modal", "caption_baseline"}
_NOT_IN_DOCUMENTS = "NOT_IN_DOCUMENTS"
_SYSTEM_PROMPT = (
    "Answer ONLY using the provided context. If the context does not "
    f"contain the answer, reply exactly: {_NOT_IN_DOCUMENTS}."
)


def _refuse() -> AnswerResponse:
    return AnswerResponse(answer="", refused=True)


def _dedup_citations(citations) -> list[Citation]:
    seen = set()
    deduped = []
    for c in citations:
        key = (c.page, tuple(c.bbox))
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _grounding_score(index, mode: str, question: str) -> float:
    """Top-1 score used only for the grounding gate, not the full
    ``retrieve()`` (see module docstring).

    Task 7: "hybrid" mode gates on ``max(dense_top1, normalized_bm25_top1)``
    instead of dense cosine alone -- a short exact-keyword query (a SKU, a
    proper noun) can sit below ``retrieval_min_score`` on dense cosine while
    still being a strong, unambiguous lexical hit. ``bm25_normalized_top1``
    (see store.py) scales raw BM25 to roughly [0, 1] by the corpus's own max
    self-score so it's comparable to the same threshold. "dense" mode is
    left untouched (dense-only by design)."""
    if mode == "cross_modal":
        top = index.cross_modal(question, 1)
        return top[0]["score"] if top else 0.0
    if mode == "caption_baseline":
        top = index.caption_baseline(question, 1)
        return top[0]["score"] if top else 0.0
    dense_top = index.dense(question, 1)
    dense_score = dense_top[0]["score"] if dense_top else 0.0
    if mode == "hybrid":
        return max(dense_score, index.bm25_normalized_top1(question))
    return dense_score


def answer_question(req: AnswerRequest) -> AnswerResponse:
    if not req.session_id:
        return _refuse()

    index = get_index(req.session_id)
    if index is None:
        return _refuse()

    mode = req.retrieval_mode if req.retrieval_mode in _SUPPORTED_MODES else "hybrid"

    if _grounding_score(index, mode, req.question) < settings.retrieval_min_score:
        return _refuse()

    results = retrieve(index, req.question, mode=mode, k=5, use_rerank=True)

    table_answer = try_table_answer(req.question, results)
    if table_answer is not None:
        return table_answer

    session = get_session(req.session_id)
    pages_by_index = {p["index"]: p for p in (session["pages"] if session else [])}

    context_lines = []
    images = []
    citations = []
    for r in results:
        chunk = r["chunk"]
        if chunk.get("kind") in ("figure", "page"):
            page = pages_by_index.get(chunk["page"])
            if page is not None:
                images.append(base64.b64encode(page["image_png"]).decode("ascii"))
            snippet = chunk.get("caption_text") or f"[{chunk.get('kind')}]"
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

    if not req.verified:
        return AnswerResponse(answer=text, refused=False, citations=citations)

    claims = verify_claims(text, results)
    refused = not any(c.supported for c in claims)
    supported_citations = _dedup_citations(
        c for claim in claims if claim.supported for c in claim.citations
    )
    return AnswerResponse(answer=text, refused=refused, claims=claims, citations=supported_citations)
