"""Step 6: validate the PRODUCTION NLI faithfulness gate (app.verify.nli.verify_claims,
cross-encoder/nli-deberta-v3-base, threshold 0.5) against HUMAN hallucination
labels from RAGTruth (Niu et al., ACL 2024).

Data: HF `wandb/RAGTruth-processed`, test split (MIT license), file
data/test-00000-of-00001.parquet, sha256 recorded in results/nli_ragtruth_summary.json.
Download (not committed; ~3.9 MB) to eval_sop/cache/ragtruth/test.parquet:
  curl -L -o eval_sop/cache/ragtruth/test.parquet \
    https://huggingface.co/datasets/wandb/RAGTruth-processed/resolve/main/data/test-00000-of-00001.parquet

Protocol (mirrors production as closely as possible):
  * task_type == "QA" only (the RAG setting); seeded random sample of N responses.
  * evidence: the RAGTruth context, windowed into ~500-word chunks with the
    product's own chunker window (app.ingest.chunk.WINDOW_TOKENS), each window a
    separate premise -- exactly like retrieved chunks in production.
  * claims: the product's split_claims() sentences of the model output.
  * HUMAN label per claim: hallucinated iff any annotated span (evident_conflict
    or baseless_info) overlaps the claim's character range in the output.
    Response label: hallucinated iff it has any annotated span.
  * gate prediction: claim flagged iff entailment prob < 0.5 (production rule).
    Response flagged iff >=1 claim flagged. Firewall-refusal operating point:
    response flagged iff ALL claims flagged (what /answer actually refuses on).
Positive class = hallucinated. Reports precision, recall, balanced accuracy,
AUROC (score = 1 - entailment prob; response score = 1 - min claim prob),
with bootstrap 95% CIs, and the trivial baselines.
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import common  # noqa: F401
from common import CACHE, RESULTS
from stats import auroc, BOOT_SEED

from app.ingest.chunk import WINDOW_TOKENS
from app.verify.nli import split_claims, verify_claims
from app.config import settings

PARQUET = CACHE / "ragtruth" / "test.parquet"


def windows(text, n=WINDOW_TOKENS):
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [""]


def claim_spans(output, claims):
    spans, pos = [], 0
    for c in claims:
        i = output.find(c, pos)
        if i < 0:
            i = output.find(c)
        if i < 0:
            spans.append(None)
            continue
        spans.append((i, i + len(c)))
        pos = i + len(c)
    return spans


def binary_metrics(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn, tn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum(), (~y & ~p).sum()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"precision": float(prec), "recall": float(rec), "specificity": float(spec),
            "balanced_acc": float((rec + spec) / 2), "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else float("nan"),
            "flag_rate": float(p.mean()), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def with_ci(y, p, s, groups, n_boot=2000):
    """Bootstrap over RESPONSES (cluster bootstrap for claim-level rows)."""
    y, p, s, groups = map(np.asarray, (y, p, s, groups))
    point = binary_metrics(y, p)
    point["auroc"] = auroc(s[y.astype(bool)], s[~y.astype(bool)])
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(BOOT_SEED)
    boots = {k: [] for k in ("precision", "recall", "balanced_acc", "auroc")}
    for _ in range(n_boot):
        gs = rng.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in gs])
        m = binary_metrics(y[ii], p[ii])
        m["auroc"] = auroc(s[ii][y[ii].astype(bool)], s[ii][~y[ii].astype(bool)])
        for k in boots:
            boots[k].append(m[k])
    for k, v in boots.items():
        v = np.array(v, float)
        v = v[~np.isnan(v)]
        point[f"{k}_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sha = hashlib.sha256(PARQUET.read_bytes()).hexdigest()
    df = pd.read_parquet(PARQUET)
    df = df[df["task_type"] == "QA"].reset_index(drop=True)
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    claim_rows, resp_rows = [], []
    for k, r in df.iterrows():
        out = r["output"]
        labels = json.loads(r["hallucination_labels"]) if isinstance(r["hallucination_labels"], str) else list(r["hallucination_labels"])
        hspans = [(int(l["start"]), int(l["end"])) for l in labels]
        ev = [{"chunk": {"text": w, "page": 0, "bbox": [0, 0, 0, 0]}} for w in windows(r["context"])]
        claims = verify_claims(out, ev)
        if not claims:
            continue
        spans = claim_spans(out, [c.text for c in claims])
        for c, sp in zip(claims, spans):
            if sp is None:
                continue
            hall = any(a < sp[1] and b > sp[0] for a, b in hspans)
            claim_rows.append({"rid": str(r["id"]), "model": r["model"], "claim": c.text, "entail": c.score,
                               "flagged": not c.supported, "human_hallucinated": hall})
        resp_rows.append({"rid": str(r["id"]), "model": r["model"], "human_hallucinated": bool(hspans),
                          "min_entail": min(c.score for c in claims), "any_flagged": any(not c.supported for c in claims),
                          "all_flagged": all(not c.supported for c in claims), "n_claims": len(claims)})
        if k % 50 == 0:
            print(k, flush=True)

    C = pd.DataFrame(claim_rows)
    R = pd.DataFrame(resp_rows)
    C.to_csv(RESULTS / "nli_ragtruth_claims.csv", index=False)
    R.to_csv(RESULTS / "nli_ragtruth_responses.csv", index=False)

    summary = {
        "dataset": "wandb/RAGTruth-processed test split, task_type=QA", "parquet_sha256": sha,
        "sample": {"n_responses": len(R), "n_claims": len(C), "seed": args.seed},
        "gate": {"model": settings.nli_model, "threshold": settings.faithfulness_threshold},
        "prevalence": {"claims_hallucinated": float(C.human_hallucinated.mean()),
                       "responses_hallucinated": float(R.human_hallucinated.mean())},
        "claim_level(any-span-overlap)": with_ci(C.human_hallucinated, C.flagged, 1 - C.entail, C.rid),
        "response_level_any_claim_flagged": with_ci(R.human_hallucinated, R.any_flagged, 1 - R.min_entail, R.rid),
        "response_level_all_claims_flagged(=/answer firewall refusal)": with_ci(R.human_hallucinated, R.all_flagged, 1 - R.min_entail, R.rid),
        "baselines": {
            "always_flag_claims": binary_metrics(C.human_hallucinated, np.ones(len(C), bool)),
            "never_flag_claims": binary_metrics(C.human_hallucinated, np.zeros(len(C), bool)),
            "always_flag_responses": binary_metrics(R.human_hallucinated, np.ones(len(R), bool)),
        },
    }
    (RESULTS / "nli_ragtruth_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("sample", "prevalence")}, indent=1))
    for k in ("claim_level(any-span-overlap)", "response_level_any_claim_flagged",
              "response_level_all_claims_flagged(=/answer firewall refusal)"):
        m = summary[k]
        print(k, {x: (round(m[x], 3) if isinstance(m[x], float) else m[x]) for x in
                  ("precision", "recall", "balanced_acc", "auroc", "flag_rate")},
              "CI P", m["precision_ci95"], "R", m["recall_ci95"], "BA", m["balanced_acc_ci95"], "AUC", m["auroc_ci95"])


if __name__ == "__main__":
    main()
