# backend/app/ingest/loader.py
"""PDF/image -> page images + native text layer, with per-block provenance.

Coordinate convention: all bboxes are ``[x0, y0, x1, y1]`` in PyMuPDF page
coordinate space — origin top-left, x increasing right, y increasing down,
in PDF points (1/72 inch), matching ``page.rect`` / the rendered
``image_png``. Downstream tasks (1.2 OCR, 1.4 chunking) must keep bboxes in
this same space so citations line up with the rendered page image.
"""
import fitz  # PyMuPDF

TEXT_BLOCK_TYPE = 0  # page.get_text("blocks") block_type: 0=text, 1=image


def _open(data: bytes) -> fitz.Document:
    try:
        return fitz.open(stream=data, filetype="pdf")
    except Exception:
        return fitz.open(stream=data)  # falls back to image auto-detection


def _page_to_dict(page: fitz.Page, index: int) -> dict:
    text_blocks = [
        {"text": text.strip(), "bbox": [x0, y0, x1, y1]}
        for x0, y0, x1, y1, text, *_rest in page.get_text("blocks")
        if _rest[1] == TEXT_BLOCK_TYPE and text.strip()
    ]
    return {
        "index": index,
        "image_png": page.get_pixmap().tobytes("png"),
        "width": page.rect.width,
        "height": page.rect.height,
        "text_blocks": text_blocks,
        "needs_ocr": len(text_blocks) == 0,
    }


def load_document(data: bytes) -> list[dict]:
    """Load a PDF (or a single PNG/JPEG treated as a one-page document) into
    a list of ``Page`` dicts. Scanned pages (empty native text layer) are
    still returned, with ``text_blocks: []`` and ``needs_ocr: True`` so a
    later OCR pass (Task 1.2) can fill them in."""
    doc = _open(data)
    try:
        return [_page_to_dict(doc[i], i) for i in range(doc.page_count)]
    finally:
        doc.close()
