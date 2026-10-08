"""Review item 3: sensitivity of the retrieval headline to the (single-annotator, LLM)
ambiguity filter. Recomputed from results/retrieval_per_query.json; no models run
except reading the cached OCR text (cache/ingested.pkl) for rule (d).

Subsets reported (strict R@1 = DocVQA source page ranked first):
  (a) all 151 questions (no filter)
  (b) the 107 kept by Claude's filter (headline in earlier RESULTS.md)
  (c) (b) + the 7 dropped items the reviewer named as uniquely answerable:
      mq-103, mq-107, mq-109, mq-113, mq-104, mq-137, mq-007
  (d) (b) + every dropped item whose gold answer (>=4 normalised chars) occurs in the
      OCR of exactly one corpus page and that page is the source page -- an explicit,
      reproducible rule (it can be fooled by short generic answers; listed in output).
The reviewer reported 0.790 [0.714, 0.857] for a 12-item restoration; the 12 ids were
not provided to us, so that figure is quoted, not reproduced.
"""
import json
import pickle
import re

import common  # noqa: F401
from common import CACHE, DATA, RESULTS
from stats import bootstrap_ci, paired_bootstrap_diff

REVIEWER_7 = ["mq-103", "mq-107", "mq-109", "mq-113", "mq-104", "mq-137", "mq-007"]


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def main():
    q = json.loads((RESULTS / "retrieval_per_query.json").read_text())
    gold = {x["id"]: x for x in json.loads((DATA / "gold_all151.json").read_text(encoding="utf-8"))}
    blob = pickle.load(open(CACHE / "ingested.pkl", "rb"))
    text = {p["index"]: norm(" ".join(b["text"] for b in p["text_blocks"])) for p in blob["pages"]}
    rule_d = []
    for r in q:
        if not r["ambiguous"]:
            continue
        ans = [norm(a) for a in gold[r["id"]]["answers"] if len(norm(a)) >= 4]
        pages = {p for p, t in text.items() for a in ans if a in t}
        if len(pages) == 1 and r["source_page"] in pages:
            rule_d.append(r["id"])

    subsets = {
        "a_all151": [r for r in q],
        "b_filtered107": [r for r in q if not r["ambiguous"]],
        "c_filtered+reviewer7": [r for r in q if not r["ambiguous"] or r["id"] in REVIEWER_7],
        "d_filtered+answer_unique_rule": [r for r in q if not r["ambiguous"] or r["id"] in rule_d],
    }
    out = {"reviewer_named_7": REVIEWER_7, "rule_d_restored": rule_d, "subsets": {}}
    for name, rows in subsets.items():
        r1 = lambda c: [1.0 if x[c]["rank"] == 1 else 0.0 for x in rows]  # noqa: E731
        out["subsets"][name] = {
            "n": len(rows),
            "hybrid+rerank_R@1": bootstrap_ci(r1("hybrid+rerank")),
            "bm25_R@1": bootstrap_ci(r1("bm25-only")),
            "cross_modal_R@1": bootstrap_ci(r1("cross_modal(CLIP)")),
            "delta_hybrid+rerank_minus_bm25": paired_bootstrap_diff(r1("hybrid+rerank"), r1("bm25-only")),
            "delta_hybrid+rerank_minus_clip": paired_bootstrap_diff(r1("hybrid+rerank"), r1("cross_modal(CLIP)")),
        }
    (RESULTS / "retrieval_sensitivity.json").write_text(json.dumps(out, indent=2))
    print("rule (d) restored", len(rule_d), rule_d)
    for k, v in out["subsets"].items():
        h, d = v["hybrid+rerank_R@1"], v["delta_hybrid+rerank_minus_bm25"]
        c = v["delta_hybrid+rerank_minus_clip"]
        print(f"{k:32s} n={v['n']} R@1={h[0]:.3f} [{h[1]:.3f},{h[2]:.3f}]  dBM25={d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]  dCLIP={c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]")


if __name__ == "__main__":
    main()
