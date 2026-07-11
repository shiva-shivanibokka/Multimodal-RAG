# backend/tests/test_ingest_endpoint.py
from fastapi.testclient import TestClient

from app.main import app
from app.session import get_session
from tests.fixtures import make_scanned_pdf, make_text_pdf

client = TestClient(app)


def test_ingest_text_pdf_returns_session_and_chunks():
    pdf = make_text_pdf("Hello ingest world")
    r = client.post("/ingest", files={"file": ("doc.pdf", pdf, "application/pdf")})
    assert r.status_code == 200

    body = r.json()
    assert "session_id" in body
    assert body["n_pages"] >= 1
    assert body["n_chunks"] > 0

    session = get_session(body["session_id"])
    assert session is not None
    assert len(session["chunks"]) == body["n_chunks"]
    assert len(session["pages"]) == body["n_pages"]


def test_ingest_scanned_pdf_runs_ocr_fill_path():
    pdf = make_scanned_pdf("INVOICE TOTAL")
    r = client.post("/ingest", files={"file": ("scan.pdf", pdf, "application/pdf")})
    assert r.status_code == 200

    body = r.json()
    assert body["n_chunks"] > 0


def test_ingest_corrupt_file_returns_400_not_500():
    r = client.post("/ingest", files={"file": ("bad.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400
    assert "detail" in r.json()
