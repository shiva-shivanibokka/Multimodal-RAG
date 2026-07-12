# backend/tests/test_ocr_dedup.py
"""Task 4: a scanned page must not be OCR'd more than once by ocr_page()
across the whole /ingest -> create_session -> first get_index flow.
store.py's Index.add now reuses text_blocks (filled by /ingest) instead of
re-running ocr_page for the caption index -- this proves it end to end."""
from fastapi.testclient import TestClient

from app import session as session_module
from app.index import store as store_module
from app.ingest import ocr as ocr_module
from app.main import app
from tests.fixtures import make_scanned_pdf

client = TestClient(app)


def test_ocr_page_called_at_most_once_per_page_across_ingest_flow(monkeypatch):
    calls = {"n": 0}
    words = [
        {"text": "INVOICE", "bbox": [0, 0, 10, 10]},
        {"text": "TOTAL", "bbox": [10, 0, 20, 10]},
    ]

    def fake_ocr_page(image_png):
        calls["n"] += 1
        return words

    # main.py re-imports `ocr_page` from app.ingest.ocr on every call (local
    # import), so patching the source attribute is enough for it. store.py
    # imported the name at module load time, so it needs its own patch.
    monkeypatch.setattr(ocr_module, "ocr_page", fake_ocr_page)
    monkeypatch.setattr(store_module, "ocr_page", fake_ocr_page)
    # keep this test fast/hermetic -- table extraction isn't what's under test
    monkeypatch.setattr("app.ingest.tables.extract_tables", lambda image_png: [])

    pdf = make_scanned_pdf("INVOICE TOTAL")
    r = client.post("/ingest", files={"file": ("scan.pdf", pdf, "application/pdf")})
    assert r.status_code == 200
    session_id = r.json()["session_id"]

    assert calls["n"] == 1  # main.py's OCR-fill call for the one scanned page

    session_module.get_index(session_id)

    assert calls["n"] == 1  # store.py reused text_blocks -- no extra OCR call
