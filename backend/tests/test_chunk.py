# backend/tests/test_chunk.py
from app.ingest.chunk import chunk_pages, WINDOW_TOKENS


def _page(index, text_blocks, needs_ocr=False, width=612.0, height=792.0):
    return {
        "index": index,
        "image_png": b"fake-png-bytes",
        "width": width,
        "height": height,
        "text_blocks": text_blocks,
        "needs_ocr": needs_ocr,
    }


def _block(text, bbox):
    return {"text": text, "bbox": bbox}


def test_text_and_table_chunks_have_provenance_and_monotonic_ids():
    pages = [
        _page(0, [
            _block("Hello world", [10, 10, 100, 30]),
            _block("Second block", [10, 40, 100, 60]),
        ]),
        _page(1, [
            _block("Page two text", [10, 10, 100, 30]),
        ]),
    ]
    tables_by_page = {
        0: [{
            "bbox": [50, 100, 200, 200],
            "cells": [],
            "markdown": "| A | B |\n|---|---|\n| 1 | 2 |",
            "df_json": '{"A":{"0":1},"B":{"0":2}}',
        }],
    }

    chunks = chunk_pages(pages, tables_by_page)

    # monotonic ids 0..n-1
    assert [c["id"] for c in chunks] == list(range(len(chunks)))

    # every chunk has valid page + bbox
    valid_pages = {0, 1}
    for c in chunks:
        assert c["page"] in valid_pages
        x0, y0, x1, y1 = c["bbox"]
        assert x0 <= x1 and y0 <= y1

    # all page text is covered by some text chunk
    text_chunks = [c for c in chunks if c["kind"] == "text"]
    all_text = " ".join(c["text"] for c in text_chunks)
    assert "Hello world" in all_text
    assert "Second block" in all_text
    assert "Page two text" in all_text

    # exactly one table chunk, on page 0
    table_chunks = [c for c in chunks if c["kind"] == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0]["page"] == 0


def test_table_chunk_shape():
    pages = [_page(0, [_block("intro", [0, 0, 10, 10])])]
    table = {
        "bbox": [1, 2, 3, 4],
        "cells": [{"row": 0, "col": 0, "text": "1", "bbox": [1, 2, 2, 3]}],
        "markdown": "| A |\n|---|\n| 1 |",
        "df_json": '{"A":{"0":1}}',
    }
    chunks = chunk_pages(pages, {0: [table]})
    table_chunk = next(c for c in chunks if c["kind"] == "table")

    assert table_chunk["kind"] == "table"
    assert table_chunk["table_df_json"] == table["df_json"]
    assert table_chunk["text"] == table["markdown"]
    assert table_chunk["bbox"] == table["bbox"]


def test_long_text_windows_into_multiple_chunks():
    # one block just under the window, one block that pushes well past it
    long_block_text = " ".join(f"word{i}" for i in range(WINDOW_TOKENS - 10))
    overflow_text = " ".join(f"more{i}" for i in range(50))
    pages = [_page(0, [
        _block(long_block_text, [0, 0, 100, 20]),
        _block(overflow_text, [0, 20, 100, 40]),
    ])]

    chunks = chunk_pages(pages, {})
    text_chunks = [c for c in chunks if c["kind"] == "text"]

    assert len(text_chunks) > 1
    for c in text_chunks:
        assert len(c["text"].split()) <= WINDOW_TOKENS


def test_scanned_page_with_no_text_produces_figure_chunk():
    pages = [_page(0, [], needs_ocr=True, width=612.0, height=792.0)]
    chunks = chunk_pages(pages, {})

    assert len(chunks) == 1
    fig = chunks[0]
    assert fig["kind"] == "figure"
    assert fig["page"] == 0
    assert fig["bbox"] == [0.0, 0.0, 612.0, 792.0]
    assert fig["text"] == ""
