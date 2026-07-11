# backend/tests/test_table_answer.py
"""Task 4.3: deterministic numeric-aggregate answers over a retrieved table.

Unit-tests ``try_table_answer`` directly against a hand-made ``results`` list
(no session/index plumbing needed -- it only ever looks at ``results[0]``).
"""
import pandas as pd

from app.generate.table_answer import try_table_answer

AMOUNT_DF = pd.DataFrame({"Item": ["Widget", "Gadget"], "Amount": [10, 20]})
TABLE_CHUNK = {
    "id": 0,
    "kind": "table",
    "text": AMOUNT_DF.to_markdown(index=False),
    "page": 2,
    "bbox": [1.0, 2.0, 3.0, 4.0],
    "table_df_json": AMOUNT_DF.to_json(),
}

TEXT_CHUNK = {
    "id": 1,
    "kind": "text",
    "text": "Some unrelated paragraph about the CEO.",
    "page": 5,
    "bbox": [0, 0, 10, 10],
    "table_df_json": None,
}

JUNK_DF = pd.DataFrame({"Item": ["Widget", "Gadget"], "Amount": ["N/A", "??"]})
JUNK_CHUNK = {
    "id": 2,
    "kind": "table",
    "text": JUNK_DF.to_markdown(index=False),
    "page": 6,
    "bbox": [0, 0, 10, 10],
    "table_df_json": JUNK_DF.to_json(),
}

# Fix 1: "Count" is itself a numeric column, and "count" is also an aggregate
# keyword declared before "maximum" in _AGG_KEYWORDS. Position-based (not
# declaration-order) matching must pick `max` here, not `count`.
COUNT_DF = pd.DataFrame({"Item": ["Widget", "Gadget", "Sprocket"], "Count": [5, 40, 12]})
COUNT_CHUNK = {
    "id": 3,
    "kind": "table",
    "text": COUNT_DF.to_markdown(index=False),
    "page": 3,
    "bbox": [0, 0, 10, 10],
    "table_df_json": COUNT_DF.to_json(),
}

# Fix 2: "Price" and "Unit Price" both overlap the token "price" -- ambiguous.
PRICE_DF = pd.DataFrame({"Price": [10, 20], "Unit Price": [1, 2]})
PRICE_CHUNK = {
    "id": 4,
    "kind": "table",
    "text": PRICE_DF.to_markdown(index=False),
    "page": 4,
    "bbox": [0, 0, 10, 10],
    "table_df_json": PRICE_DF.to_json(),
}


def _results(chunk):
    return [{"chunk": chunk, "score": 1.0}]


def test_sum_aggregate_returns_exact_value():
    resp = try_table_answer("what is the total amount?", _results(TABLE_CHUNK))

    assert resp is not None
    assert "30" in resp.answer
    assert resp.refused is False
    assert len(resp.claims) == 1
    assert resp.claims[0].supported is True
    assert resp.claims[0].score == 1.0
    assert len(resp.citations) == 1
    assert resp.citations[0].page == 2


def test_average_aggregate_returns_mean():
    resp = try_table_answer("what is the average amount?", _results(TABLE_CHUNK))

    assert resp is not None
    assert "15" in resp.answer


def test_no_aggregate_keyword_returns_none():
    assert try_table_answer("who is the CEO?", _results(TABLE_CHUNK)) is None


def test_top_chunk_not_a_table_returns_none():
    assert try_table_answer("what is the total amount?", _results(TEXT_CHUNK)) is None


def test_non_numeric_column_after_coercion_returns_none():
    resp = try_table_answer("what is the total amount?", _results(JUNK_CHUNK))
    assert resp is None


def test_empty_results_returns_none():
    assert try_table_answer("what is the total amount?", []) is None


def test_count_aggregate_returns_row_count():
    resp = try_table_answer("how many amount entries are there?", _results(TABLE_CHUNK))
    assert resp is not None
    assert "2" in resp.answer


def test_min_aggregate_returns_minimum():
    resp = try_table_answer("what is the minimum amount?", _results(TABLE_CHUNK))
    assert resp is not None
    assert "10" in resp.answer


def test_max_aggregate_returns_maximum():
    resp = try_table_answer("what is the maximum amount?", _results(TABLE_CHUNK))
    assert resp is not None
    assert "20" in resp.answer


def test_aggregate_keyword_with_no_column_overlap_returns_none():
    assert try_table_answer("what is the total?", _results(TABLE_CHUNK)) is None


def test_maximum_count_picks_max_of_count_column_not_row_count():
    """Fix 1: "maximum count" must match `max` (earliest keyword position),
    not `count`, and compute max(Count) -- not len(df)."""
    resp = try_table_answer("what is the maximum count?", _results(COUNT_CHUNK))
    assert resp is not None
    assert "40" in resp.answer
    assert "maximum" in resp.answer.lower()


def test_ambiguous_column_falls_through_to_none():
    """Fix 2: "Price" and "Unit Price" tie on overlap with "price" -- must
    return None (fall through to LLM) instead of guessing one of them."""
    assert try_table_answer("what is the total price?", _results(PRICE_CHUNK)) is None
