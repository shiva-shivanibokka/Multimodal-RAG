# backend/tests/test_run_eval.py
"""Task 5.3: pure aggregation test for the eval runner. No models, no
corpus, no network -- ``aggregate`` is the same per-mode aggregation
run_eval.py's main() calls after evaluate_mode(), factored out so it can be
unit-tested with hand-made records instead of a real 40-doc ingest (that's
exercised manually via `eval/run_eval.py --limit N`, not in this suite)."""
from eval.run_eval import aggregate

# Four synthetic per-item records, one per interesting case:
#   1. answerable, cited page wrong, retrieved within top-5 but not top-1
#   2. answerable, everything correct (top-1 hit, correct citation)
#   3. unanswerable (OOD): empty source_pages, correctly refused
#   4. answerable but nothing retrieved: miss on recall/mrr/citation,
#      incorrectly refused (should have answered)
_RECORDS = [
    {
        "retrieved_pages": [5, 2, 1],
        "cited_pages": [5],
        "refused": False,
        "source_pages": [2],
        "answerable": True,
    },
    {
        "retrieved_pages": [7, 3, 9],
        "cited_pages": [7],
        "refused": False,
        "source_pages": [7],
        "answerable": True,
    },
    {
        "retrieved_pages": [4, 5],
        "cited_pages": [4],
        "refused": True,
        "source_pages": [],
        "answerable": False,
    },
    {
        "retrieved_pages": [],
        "cited_pages": [],
        "refused": True,
        "source_pages": [10],
        "answerable": True,
    },
]


def test_aggregate_computes_exact_metrics_over_synthetic_records():
    result = aggregate(_RECORDS)
    # recall@1: 3 answerable-with-source items (1, 2, 4); only item 2 hits -> 1/3
    assert result["recall_at_1"] == 1 / 3
    # recall@5: items 1 and 2 hit (page 2 is within top-5 of item 1), item 4 misses -> 2/3
    assert result["recall_at_5"] == 2 / 3
    # mrr: item1 -> 1/2 (rank 2), item2 -> 1.0 (rank 1), item4 -> 0.0 -> mean 0.5
    assert result["mrr"] == 0.5
    # citation_accuracy: item1 wrong (0), item2 correct (1), item4 no citation (0) -> 1/3
    assert result["citation_accuracy"] == 1 / 3
    # refusal_accuracy over ALL 4 records: items 1,2,3 correct, item 4 wrong -> 3/4
    assert result["refusal_accuracy"] == 0.75


def test_aggregate_empty_records_returns_zeros():
    result = aggregate([])
    assert result == {
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "mrr": 0.0,
        "citation_accuracy": 0.0,
        "refusal_accuracy": 0.0,
    }
