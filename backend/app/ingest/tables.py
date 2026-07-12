# backend/app/ingest/tables.py
"""img2table table extraction -> structured cells + markdown + df_json.

OCR backend: img2table's built-in docTR integration (``img2table.ocr.DocTR``),
which wraps the same docTR predictor used in Task 1.2 (``app.ingest.ocr``).
No system Tesseract binary is required or installed on this box, keeping
table extraction CPU-only and self-contained.

Coordinate convention: MUST match ``app.ingest.loader`` -- bbox is
``[x0, y0, x1, y1]`` in page-point space, origin top-left (see loader.py
docstring). img2table already returns absolute PIXEL coordinates of the
input image, and (per ``ocr.py``'s docstring) pages rendered by
``loader.py`` at zoom 1 have pixel dimensions == PDF point dimensions, so no
additional scaling is applied here -- the image passed in must be a page
image from that same pipeline (or anything else at a 1px == 1pt scale).
"""
import re

import pandas as pd
from img2table.document import Image as I2TImage
from img2table.ocr import DocTR

_TRAILING_SUFFIX_RE = re.compile(r"\.\d+$")

_ocr = None  # ponytail: module-level lazy singleton, loaded once on first use


def _get_ocr() -> DocTR:
    global _ocr
    if _ocr is None:
        _ocr = DocTR(detect_language=False)
    return _ocr


def _dedupe_columns(names: list[str]) -> list[str]:
    """Real scanned tables sometimes have two blank/identical header cells
    (e.g. two empty strings). Duplicate column labels break every
    label-based DataFrame op downstream (df[col], to_json(orient="columns"))
    -- suffix repeats (".1", ".2", ...) the same way pandas' own CSV reader
    does, so every column stays uniquely addressable by name.

    Task 8: tracks the accumulated OUTPUT set (``used``), not just how many
    times each original name was seen -- a naive "seen count" scheme can
    generate a suffixed candidate (e.g. "A.1") that collides with an
    ALREADY-PRESENT original column of that exact name (e.g. input
    ``["A", "A", "A.1"]`` naively dedupes to ``["A", "A.1", "A.1"]``, still
    a collision -- and even keying the retry counter by the literal
    colliding name, e.g. "A.1", would produce "A.1.1" instead of the next
    free slot in the SAME family). Suffix counters are keyed by the name's
    ROOT (itself, with any existing trailing ".<digits>" stripped) so a
    duplicate that happens to already look like a suffixed name (e.g. a
    literal header cell reading "A.1") still lands in the "A" family and
    keeps incrementing from wherever that family left off, e.g.
    ``["A", "A", "A.1"] -> ["A", "A.1", "A.2"]``."""
    seen: dict[str, int] = {}
    used: set[str] = set()
    deduped = []
    for name in names:
        root = _TRAILING_SUFFIX_RE.sub("", name)
        candidate = name
        while candidate in used:
            seen[root] = seen.get(root, 0) + 1
            candidate = f"{root}.{seen[root]}"
        used.add(candidate)
        deduped.append(candidate)
    return deduped


def _rows_to_dataframe(rows_values: list[list[str]]) -> pd.DataFrame:
    """First row is treated as the header (column names) when there's more
    than one row -- this is what lets Phase 4's numeric-aggregate lookups
    address a table by column name (e.g. "Amount")."""
    if len(rows_values) > 1:
        df = pd.DataFrame(rows_values[1:], columns=_dedupe_columns(rows_values[0]))
    else:
        df = pd.DataFrame(rows_values)

    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().all():  # whole column parses as numeric -> use it
            df[col] = numeric
    return df


def extract_tables(image_png: bytes) -> list[dict]:
    """Detect bordered tables in a page image and return structured cells,
    a GitHub-flavored markdown rendering, and a DataFrame JSON payload.

    Returns: ``[{"bbox": [x0,y0,x1,y1], "cells": [{"row", "col", "text",
    "bbox"}], "markdown": str, "df_json": str}, ...]``. ``df_json`` is
    ``df.to_json()`` and round-trips via ``pd.read_json``.
    """
    doc = I2TImage(image_png)
    extracted = doc.extract_tables(ocr=_get_ocr(), borderless_tables=False)

    tables = []
    for table in extracted:
        cells = []
        rows_values: list[list[str]] = []
        for row_idx, row_cells in table.content.items():
            row_values = [cell.value or "" for cell in row_cells]
            rows_values.append(row_values)
            for col_idx, cell in enumerate(row_cells):
                cells.append(
                    {
                        "row": row_idx,
                        "col": col_idx,
                        "text": cell.value or "",
                        "bbox": [cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2],
                    }
                )

        df = _rows_to_dataframe(rows_values)
        tables.append(
            {
                "bbox": [table.bbox.x1, table.bbox.y1, table.bbox.x2, table.bbox.y2],
                "cells": cells,
                "markdown": df.to_markdown(index=False),
                "df_json": df.to_json(),
            }
        )
    return tables
