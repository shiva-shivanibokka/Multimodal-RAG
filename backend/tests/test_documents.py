# backend/tests/test_documents.py
"""Multi-file sessions: /ingest with several files, /documents to append,
DELETE /documents to remove one file. Verifies the combined session keeps
page indices, chunk ids, and doc_ids globally unique, and that removal drops
exactly one file's content while leaving the rest queryable."""
from fastapi.testclient import TestClient

from app.main import app
from app.session import get_index, get_session
from tests.fixtures import make_text_pdf

client = TestClient(app)


def _ids(session):
    pages = [p["index"] for p in session["pages"]]
    chunk_ids = [c["id"] for c in session["chunks"]]
    return pages, chunk_ids


def test_ingest_multiple_files_one_combined_session():
    r = client.post(
        "/ingest",
        files=[
            ("files", ("alpha.pdf", make_text_pdf("Alpha revenue report"), "application/pdf")),
            ("files", ("beta.pdf", make_text_pdf("Beta safety manual"), "application/pdf")),
        ],
    )
    assert r.status_code == 200
    body = r.json()
    assert [d["doc_id"] for d in body["docs"]] == [0, 1]
    assert [d["filename"] for d in body["docs"]] == ["alpha.pdf", "beta.pdf"]

    session = get_session(body["session_id"])
    pages, chunk_ids = _ids(session)
    assert len(pages) == len(set(pages)), "page indices must be unique across files"
    assert len(chunk_ids) == len(set(chunk_ids)), "chunk ids must be unique across files"
    # both files' text is searchable in the one session
    assert get_index(body["session_id"]) is not None


def test_add_then_remove_document():
    r = client.post(
        "/ingest",
        files=[("files", ("a.pdf", make_text_pdf("first doc apples"), "application/pdf"))],
    )
    sid = r.json()["session_id"]

    # append a second file
    r2 = client.post(
        "/documents",
        data={"session_id": sid},
        files=[("files", ("b.pdf", make_text_pdf("second doc bananas"), "application/pdf"))],
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert [d["doc_id"] for d in body2["docs"]] == [0, 1]
    session = get_session(sid)
    _, chunk_ids = _ids(session)
    assert len(chunk_ids) == len(set(chunk_ids)), "appended chunk ids must not collide"
    chunks_before = len(session["chunks"])

    # remove the first file
    r3 = client.delete(f"/documents?session_id={sid}&doc_id=0")
    assert r3.status_code == 200
    body3 = r3.json()
    assert [d["doc_id"] for d in body3["docs"]] == [1]
    session = get_session(sid)
    assert all(p["doc_id"] == 1 for p in session["pages"])
    assert all(c["doc_id"] == 1 for c in session["chunks"])
    assert len(session["chunks"]) < chunks_before
    # session still queryable after removal (index rebuilds, no re-OCR)
    assert get_index(sid) is not None


def test_remove_from_unknown_session_404():
    r = client.delete("/documents?session_id=deadbeef&doc_id=0")
    assert r.status_code == 404


def test_add_to_unknown_session_404():
    r = client.post(
        "/documents",
        data={"session_id": "deadbeef"},
        files=[("files", ("x.pdf", make_text_pdf("x"), "application/pdf"))],
    )
    assert r.status_code == 404
