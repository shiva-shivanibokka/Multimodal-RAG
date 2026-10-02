"""Step 2: (a) reproduce backend/eval/report.json with the ORIGINAL runner functions;
(b) retrieval ablations over the expanded gold set; (c) dump per-query gate scores
for answerable + OOD questions (consumed by 03_refusal.py).

Everything here is deterministic (no sampling). CIs are percentile bootstrap over
questions (10k resamples, seed 0).
"""
import json
import re

import common  # noqa: F401
from common import BACKEND, DATA, RESULTS, install_rerank_cache, load_or_ingest
from stats import bootstrap_ci, paired_bootstrap_diff

from app.generate.answer import _grounding_score
from app.retrieve.hybrid import retrieve
from app.session import get_session
from eval.run_eval import MODES, aggregate, evaluate_mode, _dedup_pages

K = 5  # chunks, same as production /answer and the original runner

CONFIGS = {
    # name: (mode, use_rerank)
    "dense+rerank": ("dense", True),
    "dense-norerank": ("dense", False),
    "hybrid+rerank": ("hybrid", True),
    "hybrid-norerank": ("hybrid", False),
    "bm25-only": ("bm25", False),          # simple lexical baseline (not a product mode)
    "cross_modal(CLIP)": ("cross_modal", False),
    "caption_baseline": ("caption_baseline", False),
}


def norm_text(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def run_config(index, q, mode, rerank):
    if mode == "bm25":
        res = index.bm25(q, K)
    else:
        res = retrieve(index, q, mode=mode, k=K, use_rerank=rerank)
    pages = _dedup_pages(res)
    top_score = res[0]["score"] if res else float("nan")
    return pages, top_score


def main():
    install_rerank_cache()
    sid, index, d2p = load_or_ingest()
    session = get_session(sid)
    page_text = {p["index"]: norm_text(" ".join(b["text"] for b in p["text_blocks"])) for p in session["pages"]}

    # ---------- (a) reproduction ----------
    # (first run was WITHOUT the rerank cache; skipped on re-runs if already written)
    if not (RESULTS / "reproduce_report.json").exists():
        orig_gold = json.loads((BACKEND / "eval" / "gold" / "enterprise_docs.json").read_text(encoding="utf-8"))
        repro = {m: aggregate(evaluate_mode(index, orig_gold, d2p, m)) for m in MODES}
        committed = json.loads((BACKEND / "eval" / "report.json").read_text(encoding="utf-8"))["modes"]
        diffs = {m: {k: round(repro[m][k] - committed[m][k], 6) for k in repro[m]} for m in MODES}
        (RESULTS / "reproduce_report.json").write_text(json.dumps({"reproduced": repro, "committed": committed, "diff": diffs}, indent=2))
        print("reproduction diff (reproduced - committed):", json.dumps(diffs))

    # ---------- (b) ablations on expanded gold ----------
    gold = json.loads((DATA / "gold_all151.json").read_text(encoding="utf-8"))
    ood = json.loads((DATA / "ood_questions.json").read_text(encoding="utf-8"))

    per_query = []
    for it in gold:
        src = d2p[it["source_doc"]]
        answers = [norm_text(a) for a in it["answers"]]
        long_answers = [a for a in answers if len(a) >= 4]
        rec = {"id": it["id"], "ambiguous": it["ambiguous"], "source_page": src}
        for name, (mode, rr) in CONFIGS.items():
            pages, top = run_config(index, it["question"], mode, rr)
            rank = pages.index(src) + 1 if src in pages else None
            # lenient relevance: page OCR contains a (>=4 char) gold answer string, or is the gold page
            if long_answers:
                len_rank = next((i + 1 for i, p in enumerate(pages)
                                 if p == src or any(a in page_text[p] for a in long_answers)), None)
            else:
                len_rank = rank
            rec[name] = {"pages": pages, "rank": rank, "lenient_rank": len_rank, "top_score": top}
        per_query.append(rec)
        print("retrieval", len(per_query), flush=True)

    def summarize(rows, rank_key):
        out = {}
        for name in CONFIGS:
            r1 = [1.0 if (q[name][rank_key] == 1) else 0.0 for q in rows]
            r5 = [1.0 if q[name][rank_key] is not None else 0.0 for q in rows]
            rr = [1.0 / q[name][rank_key] if q[name][rank_key] else 0.0 for q in rows]
            out[name] = {
                "n": len(rows),
                "recall@1": bootstrap_ci(r1), "recall@5chunks": bootstrap_ci(r5), "mrr": bootstrap_ci(rr),
            }
        return out

    subsets = {
        "filtered_unambiguous": [q for q in per_query if not q["ambiguous"]],
        "all151": per_query,
        "ambiguous_only": [q for q in per_query if q["ambiguous"]],
    }
    summary = {s: {"strict_gold_page": summarize(rows, "rank"), "lenient_answer_in_page": summarize(rows, "lenient_rank")}
               for s, rows in subsets.items()}
    summary["random_baseline_expected"] = {"recall@1": 1 / 40, "recall@5pages": 5 / 40}

    rows = subsets["filtered_unambiguous"]

    def vec(name, metric):
        if metric == "r1":
            return [1.0 if q[name]["rank"] == 1 else 0.0 for q in rows]
        return [1.0 / q[name]["rank"] if q[name]["rank"] else 0.0 for q in rows]

    pairs = [("dense+rerank", "dense-norerank"), ("hybrid+rerank", "hybrid-norerank"),
             ("hybrid+rerank", "dense+rerank"), ("hybrid-norerank", "dense-norerank"),
             ("caption_baseline", "cross_modal(CLIP)"), ("hybrid+rerank", "caption_baseline"),
             ("hybrid+rerank", "bm25-only"), ("dense+rerank", "cross_modal(CLIP)")]
    summary["paired_diffs_filtered_strict"] = {
        f"{a} - {b}": {"recall@1": paired_bootstrap_diff(vec(a, "r1"), vec(b, "r1")),
                       "mrr": paired_bootstrap_diff(vec(a, "mrr"), vec(b, "mrr"))}
        for a, b in pairs}

    (RESULTS / "retrieval_per_query.json").write_text(json.dumps(per_query, indent=1))
    (RESULTS / "retrieval_summary.json").write_text(json.dumps(summary, indent=2))
    for s in ("filtered_unambiguous", "all151"):
        for rk in ("strict_gold_page", "lenient_answer_in_page"):
            print(f"\n== {s} ({rk}) ==")
            for name, m in summary[s][rk].items():
                print(f"{name:20s} n={m['n']} R@1={m['recall@1'][0]:.3f} [{m['recall@1'][1]:.3f},{m['recall@1'][2]:.3f}]"
                      f" R@5={m['recall@5chunks'][0]:.3f} MRR={m['mrr'][0]:.3f}")
    for k, v in summary["paired_diffs_filtered_strict"].items():
        print(k, "dR@1=%.3f [%.3f,%.3f] p=%.3f" % tuple(v["recall@1"]))

    # ---------- (c) gate scores for refusal analysis ----------
    gate_rows = []
    qs = [(q["id"], q["question"], "answerable", q["ambiguous"], d2p[q["source_doc"]]) for q in gold]
    qs += [(o["id"], o["question"], o["ood_type"], False, None) for o in ood]
    for qid, qtext, kind, amb, src in qs:
        row = {"id": qid, "kind": kind, "ambiguous": amb}
        for mode in ("dense", "hybrid", "cross_modal", "caption_baseline"):
            row[f"gate_{mode}"] = _grounding_score(index, mode, qtext)
        row["bm25_norm"] = index.bm25_normalized_top1(qtext)
        # alternative confidence signal: bge-reranker top-1 logit over the hybrid pool
        res = retrieve(index, qtext, mode="hybrid", k=K, use_rerank=True)
        row["reranker_top1_logit"] = res[0]["score"] if res else float("nan")
        pages = _dedup_pages(res)
        row["hybrid_hit1"] = bool(src is not None and pages[:1] == [src])
        row["hybrid_hit5"] = bool(src is not None and src in pages)
        for mode in ("dense", "cross_modal", "caption_baseline"):
            p2 = _dedup_pages(retrieve(index, qtext, mode=mode, k=K, use_rerank=(mode == "dense")))
            row[f"{mode}_hit1"] = bool(src is not None and p2[:1] == [src])
        gate_rows.append(row)
        print("gate", len(gate_rows), flush=True)
    (RESULTS / "gate_scores.json").write_text(json.dumps(gate_rows, indent=1))
    print("wrote gate scores for", len(gate_rows), "questions")


if __name__ == "__main__":
    main()
