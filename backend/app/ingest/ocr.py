# backend/app/ingest/ocr.py
"""docTR OCR fallback for scanned pages (empty native text layer).

Granularity: WORD-level. Each recognized word becomes one item; docTR also
exposes block/line grouping but words are the smallest unit with reliable
bboxes, and downstream chunking (Task 1.4) is expected to regroup words into
larger units itself.

Coordinate convention: MUST match ``app.ingest.loader`` — ``bbox`` is
``[x0, y0, x1, y1]`` in page-point space, origin top-left, y increasing down
(see loader.py docstring). docTR returns geometry as RELATIVE coordinates in
``[0, 1]`` (fraction of image width/height, origin top-left). We convert by
multiplying by the page's pixel width/height.

This conversion is exact only because ``loader.py`` renders pages with
PyMuPDF's default pixmap zoom (scale 1), under which rendered pixel
dimensions equal the PDF page's point dimensions (72 DPI) — verified: a
612x792pt page renders to a 612x792px PNG. If ``image_png`` ever comes from
a differently-scaled render, pass the true page point size explicitly via
``page_width``/``page_height``.
"""
import io

import numpy as np
from PIL import Image

_predictor = None  # ponytail: module-level lazy singleton, loaded once on first use


def _get_predictor():
    global _predictor
    if _predictor is None:
        from doctr.models import ocr_predictor

        _predictor = ocr_predictor(pretrained=True)  # CPU by default
    return _predictor


def ocr_page(
    image_png: bytes, page_width: float | None = None, page_height: float | None = None
) -> list[dict]:
    """Run OCR on a rendered page image and return word-level results.

    Args:
        image_png: PNG bytes of the page image (e.g. ``Page["image_png"]``).
        page_width, page_height: page size in PDF points. Defaults to the
            image's own pixel dimensions, which equals the point size for
            pages rendered by ``loader.py`` (see module docstring).

    Returns: ``[{"text": str, "bbox": [x0, y0, x1, y1]}, ...]`` one entry per
    recognized word, bbox in page-point space.
    """
    img = np.array(Image.open(io.BytesIO(image_png)).convert("RGB"))
    if page_width is None:
        page_height, page_width = img.shape[0], img.shape[1]

    model = _get_predictor()
    result = model([img])
    data = result.export()

    words = []
    for page in data["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    (rx0, ry0), (rx1, ry1) = word["geometry"]
                    words.append(
                        {
                            "text": word["value"],
                            "bbox": [
                                rx0 * page_width,
                                ry0 * page_height,
                                rx1 * page_width,
                                ry1 * page_height,
                            ],
                        }
                    )
    return words
