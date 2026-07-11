# backend/tests/test_rerank.py
"""Downloads BAAI/bge-reranker-base (~1GB) on first run.
Set HF_HOME/TORCH_HOME to a writable cache dir before running, e.g.:
    HF_HOME=C:/mrag/.cache TORCH_HOME=C:/mrag/.cache \
        /c/mrag/.venv/Scripts/python.exe -m pytest tests/test_rerank.py -v
"""
from app.index.rerank import rerank

QUERY = "What are the health benefits of regular exercise?"

CANDIDATES = [
    {
        "chunk": {
            "id": 0,
            "text": (
                "Programming languages like Python and Rust differ in memory "
                "management: Python uses garbage collection while Rust uses "
                "compile-time ownership checks."
            ),
        },
        "score": 0.9,  # deliberately high fusion score -- irrelevant to query
    },
    {
        "chunk": {
            "id": 1,
            "text": (
                "Regular exercise strengthens the cardiovascular system, "
                "improves mood through endorphin release, and lowers the "
                "risk of chronic diseases such as diabetes and heart disease."
            ),
        },
        "score": 0.1,  # deliberately low fusion score -- most relevant chunk
    },
    {
        "chunk": {
            "id": 2,
            "text": (
                "The stock market closed higher today as investors reacted "
                "to the latest quarterly earnings reports from major banks."
            ),
        },
        "score": 0.5,
    },
]


def test_rerank_puts_most_relevant_chunk_first():
    results = rerank(QUERY, CANDIDATES, k=3)

    assert len(results) == 3
    assert results[0]["chunk"]["id"] == 1

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    for r in results:
        assert set(r.keys()) == {"chunk", "score", "fusion_score"}
        assert isinstance(r["score"], float)


def test_rerank_respects_k():
    results = rerank(QUERY, CANDIDATES, k=1)

    assert len(results) == 1
    assert results[0]["chunk"]["id"] == 1
