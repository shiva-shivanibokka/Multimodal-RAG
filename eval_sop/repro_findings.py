"""Deterministic reproductions of code-level findings. NO product code is changed;
these only demonstrate behaviour so a reviewer can verify each claim in RESULTS.md.

F1  NOT_IN_DOCUMENTS sentinel is matched with exact equality
    (backend/app/generate/answer.py: `if text.strip() == _NOT_IN_DOCUMENTS`), so a
    model reply of "NOT_IN_DOCUMENTS." (trailing period) is NOT treated as a refusal.
F2  Deterministic table shortcut (backend/app/generate/table_answer.py) fires on
    "how many"/"total"/... questions whenever the top hybrid+rerank chunk is a table
    and a column name overlaps the question; it then returns a column aggregate with
    supported=True, score=1.0, bypassing the LLM and the NLI gate. We run every one
    of the 151 DocVQA questions + 38 OOD questions through production retrieval and
    record when it fires and whether its answer matches the DocVQA gold answer.
Output: results/repro_findings.json
"""
import json

import common  # noqa: F401
from common import DATA, RESULTS, install_rerank_cache, load_or_ingest

install_rerank_cache()
from stats import anls, contains_match, exact_match


def f1():
    from app.generate import answer as a
    cases = ["NOT_IN_DOCUMENTS", "NOT_IN_DOCUMENTS.", "NOT_IN_DOCUMENTS\n", "not_in_documents", "The answer is NOT_IN_DOCUMENTS."]
    return {c: {"treated_as_refusal_by_answer.py": c.strip() == a._NOT_IN_DOCUMENTS} for c in cases}


def f2():
    from app.generate.table_answer import try_table_answer
    from app.retrieve.hybrid import retrieve
    sid, index, d2p = load_or_ingest()
    gold = json.loads((DATA / "gold_all151.json").read_text(encoding="utf-8"))
    ood = json.loads((DATA / "ood_questions.json").read_text(encoding="utf-8"))
    fired = []
    for q in [(g["id"], g["question"], g["answers"]) for g in gold] + [(o["id"], o["question"], None) for o in ood]:
        res = retrieve(index, q[1], mode="hybrid", k=5, use_rerank=True)
        out = try_table_answer(q[1], res)
        if out is not None:
            row = {"id": q[0], "question": q[1], "shortcut_answer": out.answer, "claim_supported": out.claims[0].supported,
                   "claim_score": out.claims[0].score, "gold": q[2]}
            if q[2] is not None:
                row.update(em=exact_match(out.answer, q[2]), contains=contains_match(out.answer, q[2]), anls=anls(out.answer, q[2]))
            fired.append(row)
    n_ans = sum(1 for r in fired if r["gold"] is not None)
    return {"n_questions_checked": len(gold) + len(ood), "n_fired": len(fired), "n_fired_on_answerable": n_ans,
            "n_fired_on_ood": len(fired) - n_ans,
            "n_correct_contains": sum(r.get("contains", 0) for r in fired), "fired": fired}


def f2_synthetic():
    """F2b: synthetic table modelled on DocVQA doc 4751 (attendance table). A per-row
    'how many' question gets a COLUMN COUNT, marked supported=True / score=1.0."""
    import pandas as pd
    from app.generate.table_answer import try_table_answer
    df = pd.DataFrame({"Director": ["Y.C. Deveshwar", "A. Baijal", "S. Banerjee"],
                       "No. of meetings attended": [2, 2, 3]})
    chunk = {"kind": "table", "table_df_json": df.to_json(), "text": df.to_markdown(), "page": 0, "bbox": [0, 0, 1, 1]}
    q = "How many meetings has Y.C. Deveshwar attended?"
    out = try_table_answer(q, [{"chunk": chunk, "score": 1.0}])
    return {"question": q, "true_answer": "2", "shortcut_answer": out.answer if out else None,
            "claim_supported": out.claims[0].supported if out else None, "claim_score": out.claims[0].score if out else None}


def main():
    out = {"F1_sentinel_exact_match": f1(), "F2_table_shortcut": f2(), "F2b_table_shortcut_synthetic": f2_synthetic()}
    (RESULTS / "repro_findings.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out["F1_sentinel_exact_match"], indent=1))
    t = out["F2_table_shortcut"]
    print({k: v for k, v in t.items() if k != "fired"})
    print(out["F2b_table_shortcut_synthetic"])
    for r in t["fired"]:
        print(r["id"], "|", r["question"], "|", r["shortcut_answer"], "| gold:", r["gold"])


if __name__ == "__main__":
    main()
