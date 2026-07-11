# backend/tests/test_ocr.py
from app.ingest.loader import load_document
from app.ingest.ocr import ocr_page
from tests.fixtures import make_scanned_pdf


def test_ocr_page_finds_known_phrase_with_bbox_in_page_bounds():
    pages = load_document(make_scanned_pdf("INVOICE TOTAL"))
    page = pages[0]
    assert page["needs_ocr"] is True

    words = ocr_page(page["image_png"])
    assert len(words) > 0

    texts = [w["text"].strip(".,:;").lower() for w in words]
    # ponytail: token overlap, not exact string — OCR is never pixel-perfect.
    expected = {"invoice", "total"}
    hits = [w for w, t in zip(words, texts) if t in expected]
    assert hits, f"expected one of {expected} among OCR tokens, got {texts}"

    for w in hits:
        x0, y0, x1, y1 = w["bbox"]
        assert 0 <= x0 < x1 <= page["width"]
        assert 0 <= y0 < y1 <= page["height"]
