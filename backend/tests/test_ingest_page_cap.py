# backend/tests/test_ingest_page_cap.py
"""Task 3: /ingest page-count cap -- a document over settings.max_pages is
rejected with 413 right after loading, before OCR/table extraction ever
touches a page (the actual DoS-amplification cost this guards against)."""
from fastapi.testclient import TestClient

from app import config
from app.ingest import loader as loader_module
from app.main import app

client = TestClient(app)


def _fake_page(i: int) -> dict:
    return {
        "index": i,
        "image_png": b"",
        "width": 10.0,
        "height": 10.0,
        "text_blocks": [],
        "needs_ocr": False,
    }


def test_ingest_rejects_document_over_page_cap(monkeypatch):
    fake_pages = [_fake_page(i) for i in range(101)]
    monkeypatch.setattr(loader_module, "load_document", lambda data: fake_pages)

    ocr_calls = {"n": 0}
    tables_calls = {"n": 0}
    monkeypatch.setattr(
        "app.ingest.ocr.ocr_page",
        lambda image_png: (ocr_calls.__setitem__("n", ocr_calls["n"] + 1), [])[1],
    )
    monkeypatch.setattr(
        "app.ingest.tables.extract_tables",
        lambda image_png: (tables_calls.__setitem__("n", tables_calls["n"] + 1), [])[1],
    )

    r = client.post("/ingest", files={"files": ("doc.pdf", b"x" * 10, "application/pdf")})

    assert r.status_code == 413
    assert "too many pages" in r.json()["detail"]
    assert ocr_calls["n"] == 0
    assert tables_calls["n"] == 0


def test_ingest_accepts_document_at_page_cap(monkeypatch):
    fake_pages = [_fake_page(i) for i in range(config.settings.max_pages)]
    monkeypatch.setattr(loader_module, "load_document", lambda data: fake_pages)
    monkeypatch.setattr("app.ingest.tables.extract_tables", lambda image_png: [])

    r = client.post("/ingest", files={"files": ("doc.pdf", b"x" * 10, "application/pdf")})

    assert r.status_code == 200
    assert r.json()["n_pages"] == config.settings.max_pages
