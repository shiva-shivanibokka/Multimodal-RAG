# backend/app/main.py
import hmac
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from app.config import settings
from app.generate.providers import ProviderError
from app.schemas import AnswerRequest, AnswerResponse

logger = logging.getLogger(__name__)


def _warm_models():
    """Task 1: load every lazy model singleton once, serially, before the
    app starts serving -- otherwise the first N concurrent requests each
    race to double-init the same model (wasted memory/CPU, see
    app/index/embedders.py etc.'s lazy-singleton comments). Leaves the
    singletons themselves unchanged; this just forces the first call to
    happen here instead of on first request."""
    from app.index.embedders import _get_model, _get_clip_model
    from app.index.rerank import _get_model as _get_reranker
    from app.verify.nli import _get_model as _get_nli
    from app.ingest.ocr import _get_predictor
    from app.ingest.tables import _get_ocr

    _get_model()
    _get_clip_model()
    _get_reranker()
    _get_nli()
    _get_predictor()
    _get_ocr()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _warm_models()
    yield


app = FastAPI(title="Multimodal RAG Trust Layer", lifespan=_lifespan)

# Task 5.4: report written by backend/eval/run_eval.py's --out default.
# Module-level so tests can monkeypatch it (see test_eval_report_endpoint.py).
REPORT_PATH = Path(__file__).resolve().parent.parent / "eval" / "report.json"

def require_token(authorization: str = Header(default="")):
    if settings.backend_token and not hmac.compare_digest(authorization, f"Bearer {settings.backend_token}"):
        raise HTTPException(status_code=401, detail="unauthorized")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/answer", response_model=AnswerResponse, dependencies=[Depends(require_token)])
def answer(req: AnswerRequest):
    from app.generate.answer import answer_question
    try:
        return answer_question(req)
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None

async def _process_files(files, doc_id_start, page_index_start, chunk_id_start):
    """Shared ingestion pipeline (1.1-1.4) over one or more uploaded files,
    accumulating into ONE combined session's page/chunk space. Each file
    becomes a ``doc`` with its own ``doc_id`` (so it can be removed later);
    page indices and chunk ids continue from the given starts so they stay
    globally unique across an existing session's contents.

    Returns ``(pages, chunks, docs)`` -- pages/chunks tagged with ``doc_id``,
    docs = ``[{"doc_id", "filename", "n_pages"}]``. Raises on parse failure,
    empty docs, oversized upload, or a page count over the session cap.
    """
    from app.ingest.chunk import chunk_pages
    from app.ingest.loader import load_document
    from app.ingest.ocr import ocr_page
    from app.ingest.tables import extract_tables

    cap = settings.max_upload_bytes
    all_pages: list[dict] = []
    tables_by_page: dict[int, list[dict]] = {}
    docs: list[dict] = []
    page_i = page_index_start
    doc_id = doc_id_start
    total_bytes = 0

    for file in files:
        # ponytail: read one byte past the cap and reject on actual size (a
        # missing/lying Content-Length can't sneak a huge upload past this);
        # total_bytes bounds the whole multi-file request, not just each file.
        data = await file.read(cap + 1)
        total_bytes += len(data)
        if len(data) > cap or total_bytes > cap:
            raise HTTPException(status_code=413, detail="upload too large (max 25 MB total)")
        try:
            pages = load_document(data)
        except Exception as exc:
            # Task 13: never echo the raw parser exception (could leak internal
            # paths/library internals) -- log server-side, return generic.
            logger.exception("failed to parse uploaded document")
            raise HTTPException(status_code=400, detail="could not parse document") from exc
        if not pages:
            raise HTTPException(status_code=400, detail="document has no pages")
        # Cap TOTAL session pages before any OCR/table work -- an
        # attacker-controlled huge page count is a cheap DoS lever.
        if page_i + len(pages) > settings.max_pages:
            raise HTTPException(status_code=413, detail=f"too many pages in session (max {settings.max_pages})")

        for page in pages:
            page["index"] = page_i
            page["doc_id"] = doc_id
            if page["needs_ocr"]:
                page["text_blocks"] = ocr_page(page["image_png"])
            # Task 4: extract_tables runs its own internal DocTR pass on every
            # page (see store.py note); a scanned page is OCR'd at most twice.
            tables = extract_tables(page["image_png"])
            if tables:
                tables_by_page[page_i] = tables
            all_pages.append(page)
            page_i += 1
        docs.append({"doc_id": doc_id, "filename": file.filename or f"document-{doc_id}", "n_pages": len(pages)})
        doc_id += 1

    chunks = chunk_pages(all_pages, tables_by_page)
    page_to_doc = {p["index"]: p["doc_id"] for p in all_pages}
    for c in chunks:
        c["id"] += chunk_id_start  # keep chunk ids unique across the session (hybrid.py dedups on id)
        c["doc_id"] = page_to_doc[c["page"]]
    return all_pages, chunks, docs


def _session_summary(session_id: str, session: dict) -> dict:
    return {
        "session_id": session_id,
        "docs": session["docs"],
        "n_pages": len(session["pages"]),
        "n_chunks": len(session["chunks"]),
    }


@app.post("/ingest", dependencies=[Depends(require_token)])
async def ingest(files: list[UploadFile] = File(...)):
    """Ingest one OR MORE files into a new combined session. Each file is a
    removable ``doc``; all are searchable together."""
    from app.session import create_session

    pages, chunks, docs = await _process_files(files, doc_id_start=0, page_index_start=0, chunk_id_start=0)
    session_id = create_session(pages, chunks, docs)
    return {"session_id": session_id, "docs": docs, "n_pages": len(pages), "n_chunks": len(chunks)}


@app.post("/documents", dependencies=[Depends(require_token)])
async def add_documents_endpoint(session_id: str = Form(...), files: list[UploadFile] = File(...)):
    """Append more files to an existing session (OCRs only the new files;
    existing pages keep their cached OCR/text)."""
    from app.session import add_documents, get_session

    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session expired -- re-upload the document")
    page_start = max((p["index"] for p in session["pages"]), default=-1) + 1
    chunk_start = max((c["id"] for c in session["chunks"]), default=-1) + 1
    pages, chunks, docs = await _process_files(files, session["next_doc_id"], page_start, chunk_start)
    add_documents(session_id, pages, chunks, docs)
    return _session_summary(session_id, get_session(session_id))


@app.delete("/documents", dependencies=[Depends(require_token)])
def remove_document_endpoint(session_id: str, doc_id: int):
    """Remove one file from a session by its doc_id. The index is rebuilt
    lazily (re-embed only, no re-OCR)."""
    from app.session import remove_document

    session = remove_document(session_id, doc_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session expired -- re-upload the document")
    return _session_summary(session_id, session)


@app.get("/eval/report", dependencies=[Depends(require_token)])
def eval_report():
    """Serve the benchmark report written by backend/eval/run_eval.py
    (Task 5.4). 404 when no report has been run yet -- the frontend
    dashboard falls back to its own bundled sample data in that case."""
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="no report")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/page/{session_id}/{page_index}", dependencies=[Depends(require_token)])
def get_page(session_id: str, page_index: int):
    """Serve a session's rendered page image (PNG) — powers the frontend
    citation viewer's bbox overlay (Task 4.4)."""
    from app.session import get_session

    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    pages = session["pages"]
    if page_index < 0 or page_index >= len(pages):
        raise HTTPException(status_code=404, detail="unknown page")
    return Response(content=pages[page_index]["image_png"], media_type="image/png")
