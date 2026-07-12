# backend/tests/test_tables.py
import io

import pandas as pd

from app.ingest.tables import _dedupe_columns, _rows_to_dataframe, extract_tables
from tests.fixtures import make_table_image


def test_dedupe_columns_suffixes_duplicates_and_blanks():
    # Real scanned tables can have repeated/blank header cells; duplicate
    # labels break label-based DataFrame ops downstream. Repeats get
    # pandas-style ".1"/".2" suffixes so every column stays addressable.
    assert _dedupe_columns(["Amount", "Amount"]) == ["Amount", "Amount.1"]
    assert _dedupe_columns(["", "", ""]) == ["", ".1", ".2"]
    assert _dedupe_columns(["A", "B", "A", "A"]) == ["A", "B", "A.1", "A.2"]
    # a DataFrame built with the result has unique, addressable columns
    import pandas as pd

    df = pd.DataFrame([[1, 2]], columns=_dedupe_columns(["x", "x"]))
    assert list(df.columns) == ["x", "x.1"]
    assert df["x"].iloc[0] == 1 and df["x.1"].iloc[0] == 2


def test_dedupe_columns_does_not_recollide_when_a_generated_suffix_already_exists():
    # Task 8: a real header cell literally reading "A.1" (not itself a
    # dedupe artifact) can collide with a suffix _dedupe_columns would
    # otherwise generate for an earlier "A" duplicate -- the fix must keep
    # incrementing within that same family instead of stopping at the first
    # (still-colliding) candidate.
    assert _dedupe_columns(["A", "A", "A.1"]) == ["A", "A.1", "A.2"]


def test_dedupe_columns_recollision_survives_rows_to_dataframe():
    df = _rows_to_dataframe([["A", "A", "A.1"], ["1", "2", "3"]])
    assert list(df.columns) == ["A", "A.1", "A.2"]


def test_extract_tables_recovers_values_and_supports_deterministic_math():
    png = make_table_image([["Item", "Amount"], ["A", "10"], ["B", "20"]])
    tables = extract_tables(png)

    assert len(tables) >= 1
    table = tables[0]

    x0, y0, x1, y1 = table["bbox"]
    assert 0 <= x0 < x1 and 0 <= y0 < y1

    cell_texts = {c["text"].strip() for c in table["cells"]}
    assert {"Item", "Amount", "A", "B", "10", "20"} <= cell_texts
    for cell in table["cells"]:
        assert isinstance(cell["row"], int) and isinstance(cell["col"], int)
        cx0, cy0, cx1, cy1 = cell["bbox"]
        assert cx0 < cx1 and cy0 < cy1

    assert "Amount" in table["markdown"]
    assert "10" in table["markdown"]
    assert "20" in table["markdown"]

    # pandas 3.0's read_json requires a file-like object for raw JSON strings
    # (bare strings are now treated as paths) -- StringIO is the documented fix.
    df = pd.read_json(io.StringIO(table["df_json"]))
    assert pd.to_numeric(df["Amount"]).sum() == 30
