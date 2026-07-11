# backend/tests/test_loader.py
from app.ingest.loader import load_document
from tests.fixtures import make_text_pdf, make_scanned_pdf
import io
from PIL import Image


def test_born_digital_pdf_yields_text_blocks():
    pages = load_document(make_text_pdf("HELLO WORLD"))
    assert len(pages) == 1
    page = pages[0]
    assert page["index"] == 0
    assert page["needs_ocr"] is False
    assert len(page["text_blocks"]) >= 1
    joined = " ".join(b["text"] for b in page["text_blocks"])
    assert "HELLO WORLD" in joined
    x0, y0, x1, y1 = page["text_blocks"][0]["bbox"]
    assert 0 <= x0 < x1 <= page["width"]
    assert 0 <= y0 < y1 <= page["height"]
    assert isinstance(page["image_png"], bytes) and len(page["image_png"]) > 0


def test_scanned_pdf_flags_needs_ocr():
    pages = load_document(make_scanned_pdf("SCANNED"))
    assert len(pages) == 1
    page = pages[0]
    assert page["text_blocks"] == []
    assert page["needs_ocr"] is True
    assert isinstance(page["image_png"], bytes) and len(page["image_png"]) > 0
    # image_png must be a real, decodable PNG
    Image.open(io.BytesIO(page["image_png"])).verify()


def test_image_input_treated_as_one_page_document():
    img = Image.new("RGB", (200, 100), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pages = load_document(buf.getvalue())
    assert len(pages) == 1
    assert pages[0]["index"] == 0
