# backend/tests/test_hybrid.py
from app.index.store import Index
from app.retrieve.hybrid import retrieve

# Query: "solar panel electricity generation"
#
# id0 is retrieved strongly by BOTH retrievers: shares all four query terms
# AND is genuinely about solar power generation (dense rank 1, bm25 rank 1).
# id1 shares the same four literal terms (bm25 rank 2, near id0) but is a
# sentence about naming a cat as a joke -- semantically unrelated to solar
# power, so it drops to dense rank 3, behind id5 (a genuinely
# energy-related sentence with no exact term overlap, dense rank 2, bm25
# score 0). This lets us distinguish "boosted because both retrievers agree"
# (id0) from "boosted only by shared vocabulary" (id1).
CHUNKS = [
    {
        "id": 0,
        "kind": "text",
        "text": "solar panel electricity generation systems power homes using sunlight energy",
        "page": 1,
        "bbox": [0, 0, 10, 10],
        "table_df_json": None,
    },
    {
        "id": 1,
        "kind": "text",
        "text": "my neighbor renamed his cat solar panel electricity generation as a joke",
        "page": 1,
        "bbox": [0, 10, 10, 20],
        "table_df_json": None,
    },
    {
        "id": 2,
        "kind": "text",
        "text": "quarterly revenue increased due to strong enterprise sales",
        "page": 1,
        "bbox": [0, 20, 10, 30],
        "table_df_json": None,
    },
    {
        "id": 3,
        "kind": "text",
        "text": "the mitochondria is the powerhouse of the cell",
        "page": 2,
        "bbox": [0, 0, 10, 10],
        "table_df_json": None,
    },
    {
        "id": 4,
        "kind": "text",
        "text": "docker containers package software with its dependencies",
        "page": 2,
        "bbox": [0, 10, 10, 20],
        "table_df_json": None,
    },
    {
        "id": 5,
        "kind": "text",
        "text": "wind turbines convert kinetic energy into electricity for the power grid",
        "page": 3,
        "bbox": [0, 0, 10, 10],
        "table_df_json": None,
    },
]

QUERY = "solar panel electricity generation"


def _index():
    idx = Index()
    idx.add(CHUNKS)
    return idx


def _ids(results):
    return [r["chunk"]["id"] for r in results]


def test_hybrid_rrf_ranks_both_retriever_chunk_above_single_retriever_chunk():
    idx = _index()

    results = retrieve(idx, QUERY, mode="hybrid", k=6, use_rerank=False)
    ids = _ids(results)

    # id0 (hit strongly by both dense and bm25) beats id1 (only propped up
    # by shared vocabulary, semantically off-topic) -- proves RRF fusion
    # rewards agreement between retrievers, not just a single strong hit.
    assert ids.index(0) < ids.index(1)
    assert ids[0] == 0


def test_dense_mode_does_not_give_bm25_only_chunk_the_hybrid_boost():
    idx = _index()

    dense_results = retrieve(idx, QUERY, mode="dense", k=6, use_rerank=False)
    dense_ids = _ids(dense_results)

    # still finds the semantically-right chunk
    assert dense_ids[0] == 0
    # id5 (genuinely energy-related, no exact term overlap) outranks id1
    # (exact term overlap, semantically unrelated) on dense alone...
    assert dense_ids.index(5) < dense_ids.index(1)

    # ...but in hybrid mode id1 catches up past id5, because RRF credits it
    # for a strong bm25 rank that dense-only search ignores entirely.
    hybrid_results = retrieve(idx, QUERY, mode="hybrid", k=6, use_rerank=False)
    hybrid_ids = _ids(hybrid_results)
    assert hybrid_ids.index(1) < hybrid_ids.index(5)


def test_result_length_capped_at_k_and_scores_descending():
    idx = _index()

    results = retrieve(idx, QUERY, mode="hybrid", k=3, use_rerank=False)

    assert len(results) <= 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    for r in results:
        assert set(r.keys()) == {"chunk", "score"}
        assert isinstance(r["score"], float)


def test_result_shape_documented():
    idx = _index()

    results = retrieve(idx, QUERY, mode="dense", k=2, use_rerank=False)

    assert len(results) <= 2
    for r in results:
        assert "chunk" in r and "score" in r
        assert "id" in r["chunk"]
