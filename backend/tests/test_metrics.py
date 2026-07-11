# backend/tests/test_metrics.py
"""Task 5.2: pure eval metrics. No models, no network -- every assertion is
a hand-computed exact number."""
from eval.metrics import (
    citation_accuracy,
    faithfulness_rate,
    mrr,
    recall_at_k,
    refusal_accuracy,
)


def test_recall_at_k_exact_fraction_and_k_boundary():
    retrieved = [
        [5, 3, 1, 9],  # source hits at index 1 (< k) -> hit
        [5, 3, 1, 9],  # source page 9 is at index 3, outside k=3 -> miss
        [2, 4, 6],  # source hits at index 1 (< k) -> hit
    ]
    sources = [[3], [9], [4, 7]]
    assert recall_at_k(retrieved, sources, k=3) == 2 / 3


def test_recall_at_k_skips_unanswerable_queries():
    # A query with empty source_pages (out-of-corpus) must not count as a
    # miss and must not affect the denominator.
    retrieved = [[5, 3, 1], [1, 2, 3]]
    sources = [[3], []]
    assert recall_at_k(retrieved, sources, k=3) == 1.0


def test_recall_at_k_no_answerable_queries_returns_zero():
    assert recall_at_k([[1, 2]], [[]], k=3) == 0.0


def test_mrr_exact_mean():
    retrieved = [
        [9, 3, 1],  # relevant page 3 at rank 2 -> 1/2
        [9, 3, 1],  # page 8 never retrieved -> 0
    ]
    sources = [[3], [8]]
    assert mrr(retrieved, sources) == 0.25


def test_mrr_skips_unanswerable_queries():
    retrieved = [[9, 3, 1], [1, 2, 3]]
    sources = [[3], []]
    assert mrr(retrieved, sources) == 0.5


def test_citation_accuracy_correct_wrong_and_missing():
    cited = [
        [5],  # correct: overlaps source
        [2],  # wrong: no overlap
        [],  # missing citation -> counted as 0.0, not skipped
    ]
    sources = [[5, 6], [5, 6], [5, 6]]
    assert citation_accuracy(cited, sources) == 1 / 3


def test_faithfulness_rate_exact_fraction():
    assert faithfulness_rate([[True, False], [True]]) == 2 / 3


def test_faithfulness_rate_zero_claims_returns_zero():
    assert faithfulness_rate([[], []]) == 0.0


def test_refusal_accuracy_exact_fraction():
    predicted_refused = [False, True, True, True]
    answerable = [True, True, False, False]
    # q0: answerable & answered -> correct
    # q1: answerable & refused -> incorrect
    # q2: unanswerable & refused -> correct
    # q3: unanswerable & refused -> correct
    assert refusal_accuracy(predicted_refused, answerable) == 3 / 4


def test_refusal_accuracy_empty_returns_zero():
    assert refusal_accuracy([], []) == 0.0
