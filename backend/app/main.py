# backend/app/main.py
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from app.config import settings
from app.schemas import AnswerRequest, AnswerResponse
app = FastAPI(title="Multimodal RAG Trust Layer")

def require_token(authorization: str = Header(default="")):
    if settings.backend_token and authorization != f"Bearer {settings.backend_token}":
        raise HTTPException(status_code=401, detail="unauthorized")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/answer", response_model=AnswerResponse, dependencies=[Depends(require_token)])
def answer(req: AnswerRequest):
    from app.generate.answer import answer_question
    return answer_question(req, context="")   # context wired in later phases

@app.post("/ingest", dependencies=[Depends(require_token)])
async def ingest(file: UploadFile = File(...)):
    """Orchestrate the ingestion pipeline (1.1-1.4) into a stored session:
    load pages -> OCR-fill scanned pages -> extract tables -> chunk -> store."""
    from app.ingest.chunk import chunk_pages
    from app.ingest.loader import load_document
    from app.ingest.ocr import ocr_page
    from app.ingest.tables import extract_tables
    from app.session import create_session

    data = await file.read()
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
