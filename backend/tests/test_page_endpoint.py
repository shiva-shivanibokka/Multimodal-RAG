# backend/tests/test_page_endpoint.py
from fastapi.testclient import TestClient

from app.main import app
from tests.fixtures import make_text_pdf

client = TestClient(app)


def test_get_page_returns_png_bytes():
    pdf = make_text_pdf("Hello page endpoint")
    r = client.post("/ingest", files={"file": ("doc.pdf", pdf, "application/pdf")})
    assert r.status_code == 200
    session_id = r.json()["session_id"]

    r = client.get(f"/page/{session_id}/0")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_get_page_unknown_session_404():
    r = client.get("/page/does-not-exist/0")
    assert r.status_code == 404


def test_get_page_unknown_page_index_404():
    pdf = make_text_pdf("Hello page endpoint")
    r = client.post("/ingest", files={"file": ("doc.pdf", pdf, "application/pdf")})
    session_id = r.json()["session_id"]

    r = client.get(f"/page/{session_id}/99")
    assert r.status_code == 404
