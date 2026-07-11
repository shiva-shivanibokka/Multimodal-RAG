# backend/tests/test_ingest_endpoint.py
from fastapi.testclient import TestClient

from app import config, session
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

    # Prove OCR text actually landed in a chunk, not just that chunk_pages'
    # fallback figure-chunk-for-empty-page kicked in (which would pass even
    # if the OCR-fill wiring in main.py were deleted).
    session = get_session(body["session_id"])
    assert session is not None
    expected = ("invoice", "total")
    text_chunks = [c for c in session["chunks"] if c["kind"] == "text"]
    assert any(
        word in c["text"].lower() for c in text_chunks for word in expected
    ), f"no OCR'd text chunk found: {[c['text'] for c in text_chunks]}"


def test_ingest_corrupt_file_returns_400_not_500():
    r = client.post("/ingest", files={"file": ("bad.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_ingest_oversized_upload_rejected_before_ingestion(monkeypatch):
    # Tiny cap so the test payload can stay small and fast.
    monkeypatch.setattr(config.settings, "max_upload_bytes", 10)

    called = {"v": False}

    def fake_create_session(*args, **kwargs):
        called["v"] = True
        return "should-not-be-reached"

    monkeypatch.setattr(session, "create_session", fake_create_session)

    r = client.post("/ingest", files={"file": ("big.pdf", b"x" * 1000, "application/pdf")})

    assert r.status_code == 413
    assert "too large" in r.json()["detail"]
    assert called["v"] is False
