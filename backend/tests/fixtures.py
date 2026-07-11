# backend/tests/fixtures.py
"""Synthetic PDF builders for hermetic ingestion tests. No network, no dataset download."""
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
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


def make_table_image(rows: list[list[str]], cell_w: int = 200, cell_h: int = 80, font_size: int = 36) -> bytes:
    """A bordered grid table (visible ruling lines, required by img2table's
    bordered-table detector) rendered to a standalone PNG, e.g.
    ``make_table_image([["Item", "Amount"], ["A", "10"], ["B", "20"]])``.
    Cells are large with a bundled scalable font (``ImageFont.load_default``
    with a size, portable across machines -- no OS font dependency) so OCR
    reads them reliably."""
    n_rows, n_cols = len(rows), len(rows[0])
    width, height = cell_w * n_cols + 40, cell_h * n_rows + 40
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=font_size)
    x0, y0 = 20, 20
    for r in range(n_rows + 1):
        y = y0 + r * cell_h
        draw.line([(x0, y), (x0 + n_cols * cell_w, y)], fill="black", width=2)
    for c in range(n_cols + 1):
        x = x0 + c * cell_w
        draw.line([(x, y0), (x, y0 + n_rows * cell_h)], fill="black", width=2)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            draw.text((x0 + c * cell_w + 15, y0 + r * cell_h + 20), str(val), fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_solid_image(color: str, size: tuple[int, int] = (224, 224)) -> bytes:
    """A solid-color PNG (e.g. "red", "blue") for CLIP directional tests."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
