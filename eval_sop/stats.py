"""Pure-numpy statistics: bootstrap CIs, AUROC, risk-coverage, ANLS/EM."""
from __future__ import annotations

import re
import string

import numpy as np

BOOT_N = 10_000
BOOT_SEED = 0


def bootstrap_ci(values, n=BOOT_N, seed=BOOT_SEED, stat=np.mean):
    """Percentile bootstrap 95% CI of `stat` over per-item values."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    boots = stat(v[idx], axis=1)
    return float(stat(v)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap_diff(a, b, n=BOOT_N, seed=BOOT_SEED):
    """Mean(a-b) with 95% CI and two-sided bootstrap p-value-ish (fraction of sign flips)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(min(p, 1.0))


def auroc(scores_pos, scores_neg):
    """P(score_pos > score_neg) + 0.5 P(tie). pos = answerable (should answer)."""
    p = np.asarray(scores_pos, float)[:, None]
    n = np.asarray(scores_neg, float)[None, :]
    if p.size == 0 or n.size == 0:
        return float("nan")
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def auroc_ci(scores_pos, scores_neg, n=2000, seed=BOOT_SEED):
    """Stratified bootstrap CI for AUROC (resample pos and neg separately)."""
    p, q = np.asarray(scores_pos, float), np.asarray(scores_neg, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        vals.append(auroc(p[rng.integers(0, len(p), len(p))], q[rng.integers(0, len(q), len(q))]))
    return auroc(p, q), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def risk_coverage(scores, correct):
    """Sort by confidence desc; at each coverage level, risk = 1 - accuracy among answered.
    Returns (coverage[], risk[], AURC)."""
    s = np.asarray(scores, float)
    c = np.asarray(correct, float)
    order = np.argsort(-s, kind="stable")
    c = c[order]
    k = np.arange(1, len(c) + 1)
    risk = 1 - np.cumsum(c) / k
    cov = k / len(c)
    return cov.tolist(), risk.tolist(), float(risk.mean())


# ---------- answer metrics (DocVQA convention) ----------
def _norm_ans(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds: list[str], tau: float = 0.5) -> float:
    """Standard DocVQA ANLS (Biten et al. 2019): lowercase+strip, 1-NL if NL<tau else 0, max over golds."""
    p = " ".join(pred.lower().strip().split())
    best = 0.0
    for g in golds:
        g = " ".join(g.lower().strip().split())
        if not g and not p:
            return 1.0
        nl = _lev(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best


def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(_norm_ans(pred) == _norm_ans(g) for g in golds))


def contains_match(pred: str, golds: list[str]) -> float:
    """Lenient: normalized gold answer appears as a substring of the normalized prediction."""
    p = _norm_ans(pred)
    return float(any(_norm_ans(g) and _norm_ans(g) in p for g in golds))
