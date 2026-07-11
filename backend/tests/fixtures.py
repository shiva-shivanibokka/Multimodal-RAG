# backend/tests/fixtures.py
"""Synthetic PDF builders for hermetic ingestion tests. No network, no dataset download."""
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import io


def make_text_pdf(text: str) -> bytes:
    """A born-digital, single-page PDF with a real text layer containing `text`."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=18)
    return doc.tobytes()


def make_scanned_pdf(text: str) -> bytes:
    """An image-only, single-page PDF: `text` is rendered to a PNG and placed as a
    full-page image with NO text layer, so it exercises the needs_ocr path."""
    img = Image.new("RGB", (612, 792), "white")
    ImageDraw.Draw(img).text((72, 72), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=buf.getvalue())
    # text is intentionally NOT inserted into the PDF text layer; it's only
    # baked into the pixels (a real OCR engine would read it back in 1.2).
    return doc.tobytes()
