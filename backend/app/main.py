# backend/app/main.py
import json
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from app.config import settings
from app.generate.providers import ProviderError
from app.schemas import AnswerRequest, AnswerResponse
app = FastAPI(title="Multimodal RAG Trust Layer")

# Task 5.4: report written by backend/eval/run_eval.py's --out default.
# Module-level so tests can monkeypatch it (see test_eval_report_endpoint.py).
REPORT_PATH = Path(__file__).resolve().parent.parent / "eval" / "report.json"

def require_token(authorization: str = Header(default="")):
    if settings.backend_token and authorization != f"Bearer {settings.backend_token}":
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

@app.post("/ingest", dependencies=[Depends(require_token)])
async def ingest(request: Request, file: UploadFile = File(...)):
    """Orchestrate the ingestion pipeline (1.1-1.4) into a stored session:
    load pages -> OCR-fill scanned pages -> extract tables -> chunk -> store."""
    from app.ingest.chunk import chunk_pages
    from app.ingest.loader import load_document
    from app.ingest.ocr import ocr_page
    from app.ingest.tables import extract_tables
    from app.session import create_session

    cap = settings.max_upload_bytes
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > cap:
        raise HTTPException(status_code=413, detail="file too large (max 25 MB)")

    # ponytail: don't trust Content-Length alone (missing/lying header) -- read
    # one byte past the cap and reject on the actual size, before any parsing.
    data = await file.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(status_code=413, detail="file too large (max 25 MB)")
    try:
        pages = load_document(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not parse document: {exc}") from exc
    if not pages:
        raise HTTPException(status_code=400, detail="document has no pages")

    tables_by_page: dict[int, list[dict]] = {}
    for page in pages:
        if page["needs_ocr"]:
            page["text_blocks"] = ocr_page(page["image_png"])
        tables = extract_tables(page["image_png"])
        if tables:
            tables_by_page[page["index"]] = tables

    chunks = chunk_pages(pages, tables_by_page)
    session_id = create_session(pages, chunks)
    return {"session_id": session_id, "n_pages": len(pages), "n_chunks": len(chunks)}


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
