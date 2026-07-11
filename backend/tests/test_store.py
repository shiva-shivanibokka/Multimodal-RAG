# backend/tests/test_store.py
from app.index.store import Index
from tests.fixtures import make_solid_image

CHUNKS = [
    {
        "id": 0,
        "kind": "text",
        "text": "photosynthesis converts sunlight into chemical energy in plants",
        "page": 1,
        "bbox": [0, 0, 10, 10],
        "table_df_json": None,
    },
    {
        "id": 1,
        "kind": "text",
        "text": "quarterly revenue increased due to strong enterprise sales",
        "page": 1,
        "bbox": [0, 10, 10, 20],
        "table_df_json": None,
    },
    {
        "id": 2,
        "kind": "table",
        "text": "| Amount | 100 | 200 |",
        "page": 2,
        "bbox": [0, 0, 10, 10],
        "table_df_json": "{}",
    },
    {
        "id": 3,
        "kind": "text",
        "text": "the mitochondria is the powerhouse of the cell",
        "page": 2,
        "bbox": [0, 10, 10, 20],
        "table_df_json": None,
    },
    {
        "id": 4,
        "kind": "text",
        "text": "docker containers package software with its dependencies",
        "page": 3,
        "bbox": [0, 0, 10, 10],
        "table_df_json": None,
    },
    {
        "id": 5,
        "kind": "figure",
        "text": "",
        "page": 3,
        "bbox": [0, 0, 100, 100],
        "table_df_json": None,
    },
]


def _photosynthesis_chunk():
    return next(c for c in CHUNKS if c["id"] == 0)


def test_dense_ranks_semantic_match_top():
    idx = Index()
    idx.add(CHUNKS)

    results = idx.dense("how do plants convert sunlight", 3)

    assert len(results) == 3
    assert results[0]["chunk"]["id"] == 0
    assert results[0]["chunk"] == _photosynthesis_chunk()
    # descending scores
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    for r in results:
        assert set(r.keys()) == {"chunk", "score"}
        assert isinstance(r["score"], float)


def test_bm25_ranks_exact_term_match_top():
    idx = Index()
    idx.add(CHUNKS)

    results = idx.bm25("photosynthesis sunlight", 3)

    assert len(results) == 3
    assert results[0]["chunk"]["id"] == 0
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    for r in results:
        assert set(r.keys()) == {"chunk", "score"}


def test_figure_chunk_not_indexed():
    idx = Index()
    idx.add(CHUNKS)

    dense_ids = {r["chunk"]["id"] for r in idx.dense("anything", 10)}
    bm25_ids = {r["chunk"]["id"] for r in idx.bm25("anything", 10)}

    assert 5 not in dense_ids
    assert 5 not in bm25_ids


def test_empty_index_returns_empty_lists():
    idx = Index()

    assert idx.dense("anything", 3) == []
    assert idx.bm25("anything", 3) == []


def test_empty_index_after_adding_only_figure_chunks():
    idx = Index()
    idx.add([CHUNKS[5]])

    assert idx.dense("anything", 3) == []
    assert idx.bm25("anything", 3) == []


def _cross_modal_fixture():
    pages = [
        {"index": 0, "image_png": make_solid_image("red"), "width": 224, "height": 224},
        {"index": 1, "image_png": make_solid_image("blue"), "width": 224, "height": 224},
    ]
    chunks = [
        {
            "id": 0,
            "kind": "text",
            "text": "an ordinary text paragraph",
            "page": 0,
            "bbox": [0, 0, 10, 10],
            "table_df_json": None,
        },
        {
            "id": 1,
            "kind": "figure",
            "text": "",
            "page": 0,
            "bbox": [0, 0, 224, 224],
            "table_df_json": None,
        },
        {
            "id": 2,
            "kind": "figure",
            "text": "",
            "page": 1,
            "bbox": [0, 0, 224, 224],
            "table_df_json": None,
        },
    ]
    return chunks, pages


def test_cross_modal_retrieves_matching_figure_by_color():
    chunks, pages = _cross_modal_fixture()
    idx = Index()
    idx.add(chunks, pages=pages)

    red_results = idx.cross_modal("a red image", 2)
    assert red_results[0]["chunk"]["id"] == 1
    assert red_results[0]["chunk"]["page"] == 0

    blue_results = idx.cross_modal("a blue image", 2)
    assert blue_results[0]["chunk"]["id"] == 2
    assert blue_results[0]["chunk"]["page"] == 1

    scores = [r["score"] for r in red_results]
    assert scores == sorted(scores, reverse=True)


def test_cross_modal_excludes_text_chunks():
    chunks, pages = _cross_modal_fixture()
    idx = Index()
    idx.add(chunks, pages=pages)

    ids = {r["chunk"]["id"] for r in idx.cross_modal("a red image", 10)}
    assert 0 not in ids
    assert ids == {1, 2}


def test_cross_modal_empty_without_pages():
    idx = Index()
    idx.add(CHUNKS)  # no pages passed -> no image index built

    assert idx.cross_modal("a red image", 3) == []
    # existing text search still works
    assert idx.dense("photosynthesis", 1)[0]["chunk"]["id"] == 0
    assert idx.bm25("photosynthesis", 1)[0]["chunk"]["id"] == 0


def test_cross_modal_empty_with_no_figure_chunks():
    idx = Index()
    text_only = [c for c in CHUNKS if c["kind"] != "figure"]
    idx.add(text_only, pages=[{"index": 1, "image_png": b"", "width": 1, "height": 1}])

    assert idx.cross_modal("anything", 3) == []
