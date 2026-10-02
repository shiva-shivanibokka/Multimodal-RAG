"""Step 1: build the expanded gold set from ALL 151 DocVQA QA pairs in
backend/eval/corpus/manifest.json and flag ambiguous questions.

Ambiguity rule (applied by Claude, NOT human-verified -- see data/ambiguity_labels.csv):
  A) auto: the exact (case/space-normalized) question text occurs for >=2 different
     documents in the 40-doc corpus (e.g. "What is the name of the company?" x10);
  B) manual: the question only refers to generic document parts ("this document",
     "the title", "the page no", "the logo", "the first rectangle", "the year of
     the budget"...) and names no entity that distinguishes one of the 40 pages.
     DocVQA questions were written for a single given page; in a 40-page
     open-retrieval setting these have no unique target page.
Both lists are written to data/ so the user can spot-check / overrule them.
"""
import csv
import json
import re
from collections import Counter

from common import BACKEND, DATA

MANUAL_GENERIC = {
    1: "generic 'university' -- no distinguishing entity",
    3: "generic 'the university' -- no distinguishing entity",
    4: "'the document' -- deictic",
    7: "generic 'foundation' -- no distinguishing entity",
    21: "'the company' -- 10+ company pages in corpus",
    42: "'top of the page' -- deictic",
    53: "'this document' -- deictic",
    55: "'the logo' -- deictic",
    57: "'What is date?' -- deictic",
    78: "'the last picture' -- deictic",
    81: "'ITC Ltd.' appears on ~12 pages; 'this' is deictic",
    96: "'the last rectangle' -- deictic",
    97: "'the first rectangle' -- deictic",
    101: "'the document' -- deictic",
    103: "'the document' -- deictic",
    104: "'as per the document'; two project-assignment forms in corpus",
    105: "'the title' -- deictic",
    107: "'the budget' -- several budget pages in corpus",
    109: "'the budget' -- several budget pages in corpus",
    113: "'the expenditures' -- several budget pages in corpus",
    127: "'first set ... advertised' -- deictic, several ITC brand pages",
    128: "'second set ... advertised' -- deictic, several ITC brand pages",
    131: "'the Page Number' -- deictic",
    137: "'the year of publication' -- deictic",
    141: "'the image' -- deictic",
    142: "'the document' -- deictic",
    143: "'the chart' -- deictic",
    144: "'at the bottom' -- deictic",
    145: "'the from' -- deictic",
    148: "'the seal stamped on the page' -- deictic",
    149: "'the publication' -- deictic",
}


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def main():
    manifest = json.loads((BACKEND / "eval" / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    by_q = {}
    for x in manifest:
        by_q.setdefault(norm_q(x["question"]), set()).add(x["doc_id"])

    items, rows = [], []
    for i, x in enumerate(manifest):
        dup = len(by_q[norm_q(x["question"])]) > 1
        reason = ""
        if dup:
            reason = f"auto: identical question asked of {len(by_q[norm_q(x['question'])])} different docs"
        elif i in MANUAL_GENERIC:
            reason = "manual (Claude): " + MANUAL_GENERIC[i]
        item = {
            "id": f"mq-{i:03d}",
            "manifest_index": i,
            "question": x["question"],
            "answers": x["answers"],
            "source_doc": x["image_file"],
            "answerable": True,
            "ambiguous": bool(reason),
            "ambiguity_reason": reason,
        }
        items.append(item)
        rows.append([item["id"], x["doc_id"], x["question"], " || ".join(x["answers"]), int(item["ambiguous"]), reason])

    (DATA / "gold_all151.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(DATA / "ambiguity_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "doc_id", "question", "answers", "ambiguous_claude_label", "reason", "human_agrees(y/n)"])
        for r in rows:
            w.writerow(r + [""])
    c = Counter(it["ambiguous"] for it in items)
    print(f"total={len(items)} ambiguous={c[True]} kept={c[False]}")


if __name__ == "__main__":
    main()
