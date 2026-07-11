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
