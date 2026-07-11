# backend/app/ingest/chunk.py
"""Pages + tables -> retrieval-unit Chunks, each carrying (page, bbox) provenance.

Coordinate convention: bboxes are passed through unchanged from the loader
(Task 1.1) / table extractor (Task 1.3) — page-point space, origin top-left.
No transformation happens here.

Signature: ``chunk_pages(pages, tables_by_page) -> list[Chunk]`` where
``tables_by_page`` is ``{page_index: [Table, ...]}`` (the suggested shape
from the task spec) mapping a page index to the tables extracted from it.
"""

WINDOW_TOKENS = 500  # ponytail: approx word-count window, not a real tokenizer


def _token_count(text: str) -> int:
    return len(text.split())


def _union_bbox(bboxes: list[list[float]]) -> list[float]:
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [x0, y0, x1, y1]


def _window_text_blocks(blocks: list[dict], window_tokens: int = WINDOW_TOKENS):
    """Group blocks into windows of ~window_tokens words, never splitting a
    block. Yields (joined_text, bbox, blocks_in_window)."""
    current: list[dict] = []
    current_tokens = 0
    for block in blocks:
        block_tokens = _token_count(block["text"])
        if current and current_tokens + block_tokens > window_tokens:
            yield current
            current, current_tokens = [], 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        yield current


def chunk_pages(pages: list[dict], tables_by_page: dict) -> list[dict]:
    """Window each page's text_blocks into ~WINDOW_TOKENS-word chunks,
    emit one chunk per table (from ``tables_by_page.get(page['index'], [])``),
    and emit one minimal 'figure' chunk for pages with no usable native text
    (scanned pages, or any page whose text_blocks came back empty) so a
    scanned/image page is never silently dropped.

    Figure handling is intentionally minimal (Phase 3 does real region
    detection for CLIP): one chunk per qualifying page, bbox = full page,
    text="". No per-figure region detection here.

    Every chunk gets a stable, monotonically increasing ``id`` across the
    whole return value, and every chunk carries a valid ``page`` + ``bbox``.
    """
    chunks: list[dict] = []
    next_id = 0

    for page in pages:
        page_index = page["index"]
        page_bbox = [0.0, 0.0, page["width"], page["height"]]

        for window in _window_text_blocks(page["text_blocks"]):
            chunks.append({
                "id": next_id,
                "kind": "text",
                "text": " ".join(b["text"] for b in window),
                "page": page_index,
                "bbox": _union_bbox([b["bbox"] for b in window]),
                "table_df_json": None,
            })
            next_id += 1

        for table in tables_by_page.get(page_index, []):
            chunks.append({
                "id": next_id,
                "kind": "table",
                "text": table["markdown"],
                "page": page_index,
                "bbox": table["bbox"],
                "table_df_json": table["df_json"],
            })
            next_id += 1

        if not page["text_blocks"]:  # scanned / no native text -> figure chunk
            chunks.append({
                "id": next_id,
                "kind": "figure",
                "text": "",
                "page": page_index,
                "bbox": page_bbox,
                "table_df_json": None,
            })
            next_id += 1

    return chunks
