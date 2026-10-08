"""Step 3: retrieval-gate refusal as selective prediction (no LLM involved).

Positives = answerable in-corpus questions (unambiguous subset, primary).
Negatives = out-of-corpus questions: 8 original trivial OOD + 30 hard near-domain
OOD written by Claude (data/ood_questions.json, NOT human-verified).

Per confidence signal we report:
  * AUROC (answerable vs OOD), stratified-bootstrap 95% CI
  * refusal accuracy at the shipped fixed threshold 0.25 vs the always-answer baseline
  * cross-validated accuracy with a threshold *learned* on held-out folds
    (5-fold stratified, repeated over 5 seeds -> mean +- std), so the tuned
    number is not fit on its own test data
  * risk-coverage curve + AURC, where an answered item is "correct" iff it is
    answerable AND that mode's top-1 retrieved page is the gold page.
"""
import csv
import json

import numpy as np

import common  # noqa: F401
from common import RESULTS
from stats import auroc_ci, risk_coverage

FIXED_T = 0.25  # app/config.py retrieval_min_score
SIGNALS = {
    # signal -> (field, hit field used for risk-coverage)
    "gate_dense (bge cosine)": ("gate_dense", "dense_hit1"),
    "gate_hybrid (max(bge, bm25_norm))": ("gate_hybrid", "hybrid_hit1"),
    "gate_cross_modal (CLIP cosine)": ("gate_cross_modal", "cross_modal_hit1"),
    "gate_caption_baseline (bge cosine)": ("gate_caption_baseline", "caption_baseline_hit1"),
    "bm25_norm only": ("bm25_norm", "hybrid_hit1"),
    "reranker top-1 logit (not shipped)": ("reranker_top1_logit", "hybrid_hit1"),
}


def acc_at(t, pos, neg):
    """accuracy of 'answer iff score >= t'."""
    return ((pos >= t).sum() + (neg < t).sum()) / (len(pos) + len(neg))


def cv_threshold_acc(pos, neg, seeds=(0, 1, 2, 3, 4), folds=5):
    accs, baccs = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        pf = rng.permutation(len(pos)) % folds
        nf = rng.permutation(len(neg)) % folds
        correct, bal_parts = 0, []
        tp = tn = 0
        for f in range(folds):
            ptr, nte = pos[pf != f], neg[nf == f]
            ntr, pte = neg[nf != f], pos[pf == f]
            cands = np.unique(np.concatenate([ptr, ntr]))
            # choose threshold maximizing balanced accuracy on training folds
            best = max(cands, key=lambda t: 0.5 * ((ptr >= t).mean() + (ntr < t).mean()))
            tp += (pte >= best).sum()
            tn += (nte < best).sum()
        accs.append((tp + tn) / (len(pos) + len(neg)))
        baccs.append(0.5 * (tp / len(pos) + tn / len(neg)))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(baccs)), float(np.std(baccs))


def main():
    rows = json.loads((RESULTS / "gate_scores.json").read_text())
    ans = [r for r in rows if r["kind"] == "answerable" and not r["ambiguous"]]
    triv = [r for r in rows if r["kind"] == "ood_trivial"]
    hard = [r for r in rows if r["kind"] == "ood_hard"]
    out = {"n_answerable_unambiguous": len(ans), "n_ood_trivial": len(triv), "n_ood_hard": len(hard), "signals": {}}
    curves = []
    for name, (field, hitf) in SIGNALS.items():
        pos = np.array([r[field] for r in ans])
        res = {"score_range_answerable": [float(pos.min()), float(np.median(pos)), float(pos.max())]}
        for negname, negset in (("trivial", triv), ("hard", hard), ("all_ood", triv + hard)):
            neg = np.array([r[field] for r in negset])
            n_tot = len(pos) + len(neg)
            d = {
                "auroc": auroc_ci(pos, neg),
                "score_range_ood": [float(neg.min()), float(np.median(neg)), float(neg.max())],
                "always_answer_baseline_acc": len(pos) / n_tot,
                "cv_learned_threshold": dict(zip(["acc_mean", "acc_std", "balanced_acc_mean", "balanced_acc_std"],
                                                 cv_threshold_acc(pos, neg))),
            }
            if field.startswith("gate_"):
                d["fixed_0.25"] = {
                    "acc": float(acc_at(FIXED_T, pos, neg)),
                    "false_refusal_rate_on_answerable": float((pos < FIXED_T).mean()),
                    "refusal_rate_on_ood": float((neg < FIXED_T).mean()),
                }
            res[negname] = d
        # risk-coverage over answerable(unambiguous)+all OOD
        allr = ans + triv + hard
        scores = [r[field] for r in allr]
        correct = [r["kind"] == "answerable" and r[hitf] for r in allr]
        cov, risk, aurc = risk_coverage(scores, correct)
        base_err = 1 - np.mean(correct)
        oracle = risk_coverage([float(c) for c in correct], correct)[2]
        res["risk_coverage"] = {"aurc": aurc, "random_order_aurc(=full-coverage risk)": float(base_err),
                                "oracle_aurc": oracle, "risk_at_full_coverage": float(base_err)}
        for c, r_ in zip(cov, risk):
            curves.append([name, round(c, 4), round(r_, 4)])
        out["signals"][name] = res
    (RESULTS / "refusal_selective_prediction.json").write_text(json.dumps(out, indent=2))
    with open(RESULTS / "risk_coverage_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "coverage", "risk"])
        w.writerows(curves)

    print(f"n_ans={len(ans)} n_triv={len(triv)} n_hard={len(hard)}")
    for name, res in out["signals"].items():
        a = res["all_ood"]
        fx = a.get("fixed_0.25", {})
        print(f"{name:40s} AUROC triv={res['trivial']['auroc'][0]:.3f} hard={res['hard']['auroc'][0]:.3f} "
              f"[{res['hard']['auroc'][1]:.3f},{res['hard']['auroc'][2]:.3f}] all={a['auroc'][0]:.3f} | "
              f"fixed.25 acc={fx.get('acc', float('nan')):.3f} (base {a['always_answer_baseline_acc']:.3f}) "
              f"cvacc={a['cv_learned_threshold']['acc_mean']:.3f}+-{a['cv_learned_threshold']['acc_std']:.3f} "
              f"AURC={res['risk_coverage']['aurc']:.3f} (rand {res['risk_coverage']['risk_at_full_coverage']:.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in SIGNALS:
            pts = [(c, r) for n, c, r in curves if n == name]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name, lw=1.2)
        ax.set_xlabel("coverage (fraction answered)")
        ax.set_ylabel("risk (1 - top-1 correct among answered)")
        ax.set_title(f"Risk-coverage, n={len(ans)} answerable + {len(triv)+len(hard)} OOD")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(RESULTS / "risk_coverage.png", dpi=130)
    except ImportError:
        print("matplotlib not available; skipped plot")


if __name__ == "__main__":
    main()
