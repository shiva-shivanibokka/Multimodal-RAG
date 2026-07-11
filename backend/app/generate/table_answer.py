# backend/app/generate/table_answer.py
"""Deterministic numeric-aggregate answers over a retrieved table (Task 4.3).

When the top retrieved chunk is a ``table`` chunk and the question asks for a
numeric aggregate (sum/average/count/max/min) over one of its columns, compute
the value directly from the chunk's stored ``table_df_json`` DataFrame instead
of letting the LLM eyeball the number from a markdown rendering. This bypasses
`generate()` and NLI entirely -- a computed value needs no faithfulness check,
it *is* the source.

# ponytail: keyword + column-name-overlap matching only, no NL-to-pandas. If a
question doesn't contain one of the aggregate keywords below, or its tokens
don't overlap any column name, we return None and fall through to the normal
LLM path -- never guess. Upgrade to a real NL-to-pandas parser only if this
heuristic misfires often in practice.

pandas 3.x note: ``table_df_json`` is ``df.to_json()`` and MUST be re-read via
``pd.read_json(io.StringIO(df_json))`` -- the bare-string path raises on
pandas 3.x. Numeric coercion in ``tables.py`` is all-or-nothing per column (one
OCR misread leaves a "numeric" column as strings), so the target column is
defensively re-coerced here with ``pd.to_numeric(errors="coerce")`` before
aggregating; if nothing numeric survives, we return None rather than a bogus
number.
"""
import io
import re

import pandas as pd

from app.schemas import AnswerResponse, Citation, Claim

_SNIPPET_LEN = 150

# (keyword, pandas-agg) pairs. List order is irrelevant -- _detect_aggregate
# picks whichever matching keyword starts earliest in the question, not the
# first one declared here. All keywords match on \b word boundaries so e.g.
# "count" doesn't fire inside "discount" or "account".
_AGG_KEYWORDS: list[tuple[str, str]] = [
    ("how many", "count"),
    ("number of", "count"),
    ("sum", "sum"),
    ("total", "sum"),
    ("average", "mean"),
    ("mean", "mean"),
    ("avg", "mean"),
    ("count", "count"),
    ("maximum", "max"),
    ("highest", "max"),
    ("largest", "max"),
    ("max", "max"),
    ("minimum", "min"),
    ("lowest", "min"),
    ("smallest", "min"),
    ("min", "min"),
]

_AGG_LABEL = {"sum": "sum", "mean": "average", "count": "count", "max": "maximum", "min": "minimum"}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _detect_aggregate(question: str) -> str | None:
    """Pick the aggregate keyword that starts EARLIEST in the question.

    All keywords (single- and multi-word) are matched with \\b word
    boundaries so e.g. "account" doesn't fire "count". Declaration order in
    _AGG_KEYWORDS no longer matters for which one wins -- "maximum count"
    must pick `max` (pos of "maximum" < pos of "count"), not whichever
    keyword happens to be listed first.
    """
    q = question.lower()
    best_agg, best_pos = None, None
    for kw, agg in _AGG_KEYWORDS:
        m = re.search(rf"\b{re.escape(kw)}\b", q)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos, best_agg = m.start(), agg
    return best_agg


def _detect_column(question: str, columns) -> str | None:
    """Return the uniquely-best-overlap column, or None if ambiguous.

    If two or more columns tie for the highest (non-zero) token overlap
    (e.g. "Price" and "Unit Price" both overlap "price" for "total price"),
    picking either one silently is a confident wrong answer -- return None
    so the caller falls through to the LLM instead of guessing.
    """
    q_tokens = set(_WORD_RE.findall(question.lower()))
    best_col, best_overlap, tie_count = None, 0, 0
    for col in columns:
        col_tokens = set(_WORD_RE.findall(str(col).lower()))
        overlap = len(q_tokens & col_tokens)
        if overlap > best_overlap:
            best_col, best_overlap, tie_count = col, overlap, 1
        elif overlap == best_overlap and overlap > 0:
            tie_count += 1
    if best_overlap == 0 or tie_count > 1:
        return None
    return best_col


def _format_value(value) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    # ponytail: display-only rounding to 4dp -- the claim's underlying value
    # is exact (computed by pandas), this only trims the rendered string so
    # e.g. 33.333333... shows "33.3333" instead of an unbounded repeating
    # decimal. Not a precision loss in the computed answer, just the text.
    return f"{round(value, 4):g}"


def try_table_answer(question: str, results: list[dict]) -> AnswerResponse | None:
    if not results:
        return None
    chunk = results[0]["chunk"]
    if chunk.get("kind") != "table" or not chunk.get("table_df_json"):
        return None

    agg = _detect_aggregate(question)
    if agg is None:
        return None

    df = pd.read_json(io.StringIO(chunk["table_df_json"]))
    if df.empty:
        return None

    column = _detect_column(question, df.columns)
    if column is None:
        return None

    numeric = pd.to_numeric(df[column], errors="coerce")
    if agg == "count":
        value = int(numeric.notna().sum())
    else:
        numeric = numeric.dropna()
        if numeric.empty:
            return None
        value = numeric.agg(agg)

    value_str = str(value) if agg == "count" else _format_value(value)
    answer_text = f"The {_AGG_LABEL[agg]} of the {column} column is {value_str}."

    citation = Citation(
        page=chunk["page"],
        bbox=chunk["bbox"],
        snippet=(chunk.get("text") or "table")[:_SNIPPET_LEN],
    )
    claim = Claim(text=answer_text, supported=True, score=1.0, citations=[citation])
    return AnswerResponse(answer=answer_text, refused=False, claims=[claim], citations=[citation])
