# backend/tests/test_tables.py
import io

import pandas as pd

from app.ingest.tables import extract_tables
from tests.fixtures import make_table_image


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
