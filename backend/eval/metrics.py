"""Pure evaluation metrics for retrieval + trust-layer quality.

No models, no network, no I/O -- every function here is a plain aggregation
over lists the caller (eval/run_eval.py, Task 5.3) already collected. Each
function documents its own edge cases; see docstrings for the exact rule
used when a query has no source pages, no citation, or no claims.
"""
from __future__ import annotations


def recall_at_k(
    retrieved_pages_per_query: list[list[int]],
    source_pages_per_query: list[list[int]],
    k: int,
) -> float:
    """Fraction of *answerable* queries where any source page appears in the
    first k retrieved pages.

    Per query: 1.0 if ``set(retrieved[:k]) & set(source_pages)`` is
    non-empty, else 0.0. The mean is taken only over queries whose
    ``source_pages`` is non-empty -- a query with no source pages is
    unanswerable (out-of-corpus) and has nothing to recall, so it is
    skipped rather than counted as a miss.

    Returns 0.0 if there are no answerable queries at all.
    """
    hits = []
    for retrieved, sources in zip(retrieved_pages_per_query, source_pages_per_query):
        if not sources:
            continue
        top_k = set(retrieved[:k])
        hits.append(1.0 if top_k & set(sources) else 0.0)
    return sum(hits) / len(hits) if hits else 0.0


def mrr(
    retrieved_pages_per_query: list[list[int]],
    source_pages_per_query: list[list[int]],
) -> float:
    """Mean reciprocal rank over answerable queries.

    Per query: 1 / (1-based rank of the first retrieved page that is in
    source_pages), or 0.0 if no retrieved page matches. Queries with empty
    ``source_pages`` (unanswerable) are skipped, same rule as recall_at_k.

    Returns 0.0 if there are no answerable queries.
    """
    scores = []
    for retrieved, sources in zip(retrieved_pages_per_query, source_pages_per_query):
        if not sources:
            continue
        source_set = set(sources)
        score = 0.0
        for rank, page in enumerate(retrieved, start=1):
            if page in source_set:
                score = 1.0 / rank
                break
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def citation_accuracy(
    cited_pages_per_query: list[list[int]],
    source_pages_per_query: list[list[int]],
) -> float:
    """Fraction of answerable queries whose citations point at a real source
    page.

    Per query: 1.0 if any cited page is in source_pages, else 0.0. A query
    that produced *no* citation counts as 0.0 (not skipped) -- failing to
    cite anything is itself a failure to attribute, so it should drag the
    score down rather than be excluded from the average. The denominator is
    all answerable queries (non-empty source_pages), matching recall/MRR.

    Returns 0.0 if there are no answerable queries.
    """
    scores = []
    for cited, sources in zip(cited_pages_per_query, source_pages_per_query):
        if not sources:
            continue
        scores.append(1.0 if set(cited) & set(sources) else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def faithfulness_rate(supported_flags_per_query: list[list[bool]]) -> float:
    """Fraction of claims (flattened across all queries) that the NLI gate
    marked supported.

    total supported / total claims across every query's claim list. Returns
    0.0 if there are zero claims total (rather than NaN) so callers can
    aggregate/compare this metric without special-casing division by zero.
    """
    total = 0
    supported = 0
    for flags in supported_flags_per_query:
        total += len(flags)
        supported += sum(1 for f in flags if f)
    return supported / total if total else 0.0


def refusal_accuracy(predicted_refused: list[bool], answerable: list[bool]) -> float:
    """Accuracy of the refuse-iff-unanswerable decision, over all queries.

    A prediction is correct when ``refused == (not answerable)``: refusing
    on an unanswerable query is correct, answering on an answerable query is
    correct; refusing on an answerable one or answering on an unanswerable
    one is wrong. Unlike the other metrics, every query counts here (that's
    the whole point of this metric -- including out-of-corpus questions).

    Returns 0.0 if both lists are empty.
    """
    if not predicted_refused:
        return 0.0
    correct = sum(
        1 for refused, ans in zip(predicted_refused, answerable) if refused == (not ans)
    )
    return correct / len(predicted_refused)
