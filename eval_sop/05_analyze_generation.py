"""Step 5: analyze 04_generation.py output (answer quality, end-to-end refusal,
NLI-gate flag rate) and export ~100 claims for HUMAN labeling.

Usage: python 05_analyze_generation.py results/generation_llama3.2_latest_T0.7.jsonl [--export-claims]
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import common  # noqa: F401
from common import CACHE, DATA, RESULTS
from stats import anls, bootstrap_ci, contains_match, exact_match


def mean_std(xs):
    return float(np.mean(xs)), float(np.std(xs))


def main(path):
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    rag = [r for r in recs if r["setting"] == "rag"]
    cb = [r for r in recs if r["setting"] == "closed_book"]
    # only analyze seeds whose run is complete (a killed run leaves a partial seed)
    n_items = max(sum(1 for r in rag if r["seed"] == s) for s in {r["seed"] for r in rag})
    seeds = sorted(s for s in {r["seed"] for r in rag} if sum(1 for r in rag if r["seed"] == s) == n_items)
    out = {"source_file": Path(path).name, "model": rag[0]["model"], "temperature": rag[0]["temperature"],
           "seeds": seeds, "per_seed": {}, "errors": sum(1 for r in recs if r.get("error"))}

    per_q_metrics = defaultdict(lambda: defaultdict(list))  # metric -> qid -> [per-seed values]
    for s in seeds:
        rs = [r for r in rag if r["seed"] == s]
        ans = [r for r in rs if r["kind"] == "answerable"]
        ood = [r for r in rs if r["kind"] != "answerable"]
        m = {"n_answerable": len(ans), "n_ood": len(ood), "paths_answerable": Counter(r["path"] for r in ans),
             "paths_ood": Counter(r["path"] for r in ood)}
        # answer quality: refused / errored answers score 0 (the user got no answer)
        vals = defaultdict(list)
        for r in ans:
            pred = "" if (r["refused"] or r["answer"] is None) else r["answer"]
            v = {"anls": anls(pred, r["gold_answers"]), "em": exact_match(pred, r["gold_answers"]),
                 "contains": contains_match(pred, r["gold_answers"]),
                 "answered": float(not r["refused"] and r["answer"] is not None),
                 "cited_gold_page": float(r["source_page"] in r["cited_pages"])}
            for k, x in v.items():
                vals[k].append(x)
                per_q_metrics[k][r["id"]].append(x)
        for k, xs in vals.items():
            m[k] = float(np.mean(xs))
        # end-to-end refusal on OOD
        for t in ("ood_trivial", "ood_hard"):
            sub = [r for r in ood if r["kind"] == t]
            m[f"refusal_rate_{t}"] = float(np.mean([bool(r["refused"]) for r in sub])) if sub else None
            for r in sub:
                per_q_metrics[f"refused_{t}"][r["id"]].append(float(bool(r["refused"])))
        n = len(ans) + len(ood)
        correct = sum(1 for r in ans if not r["refused"]) + sum(1 for r in ood if r["refused"])
        m["e2e_refusal_accuracy"] = correct / n
        for r in ans:
            per_q_metrics["e2e_correct"][r["id"]].append(float(not r["refused"]))
        for r in ood:
            per_q_metrics["e2e_correct"][r["id"]].append(float(bool(r["refused"])))
        # post-hoc (analysis only, product unchanged): replies that contain the sentinel but
        # were not refused because answer.py uses exact equality (finding F1)
        def _sent(r):
            return bool(r.get("raw_llm_output")) and "not_in_documents" in r["raw_llm_output"].lower() and not r["refused"]
        m["sentinel_variant_not_refused"] = {"answerable": sum(_sent(r) for r in ans), "ood": sum(_sent(r) for r in ood)}
        ref2 = lambda r: bool(r["refused"]) or _sent(r)
        m["e2e_refusal_accuracy_if_sentinel_normalized"] = (sum(1 for r in ans if not ref2(r)) + sum(1 for r in ood if ref2(r))) / n
        for t in ("ood_trivial", "ood_hard"):
            sub = [r for r in ood if r["kind"] == t]
            m[f"refusal_rate_{t}_if_sentinel_normalized"] = float(np.mean([ref2(r) for r in sub])) if sub else None
        m["always_answer_baseline_acc"] = len(ans) / n
        # NLI gate flag rate on LLM-generated claims (table shortcut claims are hard-coded supported=True)
        llm = [r for r in rs if r["path"] in ("answered", "nli_firewall_refusal")]
        claims = [c for r in llm for c in r["claims"]]
        m["n_llm_answers"] = len(llm)
        m["n_claims"] = len(claims)
        m["claim_flag_rate(unsupported)"] = float(np.mean([not c["supported"] for c in claims])) if claims else None
        m["answers_with_>=1_flagged_claim"] = float(np.mean([any(not c["supported"] for c in r["claims"]) for r in llm])) if llm else None
        m["answers_fully_refused_by_firewall"] = float(np.mean([r["path"] == "nli_firewall_refusal" for r in llm])) if llm else None
        for kind in ("answerable", "ood"):
            sub = [c for r in llm if (r["kind"] == "answerable") == (kind == "answerable") for c in r["claims"]]
            m[f"claim_flag_rate_{kind}"] = float(np.mean([not c["supported"] for c in sub])) if sub else None
        tab = [r for r in ans if r["path"] == "table_shortcut"]
        m["table_shortcut"] = {"n": len(tab), "em": float(np.mean([exact_match(r["answer"], r["gold_answers"]) for r in tab])) if tab else None,
                               "contains": float(np.mean([contains_match(r["answer"], r["gold_answers"]) for r in tab])) if tab else None,
                               "examples": [(r["question"], r["answer"], r["gold_answers"]) for r in tab[:5]]}
        tab_ood = [r for r in ood if r["path"] == "table_shortcut"]
        m["table_shortcut_on_ood"] = [(r["question"], r["answer"]) for r in tab_ood]
        # closed-book baseline
        cbs = [r for r in cb if r["seed"] == s]
        if cbs:
            m["closed_book"] = {k: float(np.mean([f(r["answer"] or "", r["gold_answers"]) for r in cbs]))
                                for k, f in (("anls", anls), ("em", exact_match), ("contains", contains_match))}
            for r in cbs:
                per_q_metrics["cb_contains"][r["id"]].append(contains_match(r["answer"] or "", r["gold_answers"]))
                per_q_metrics["cb_anls"][r["id"]].append(anls(r["answer"] or "", r["gold_answers"]))
        prompt_toks = [u.get("prompt_tokens", 0) for r in rs for u in r["usage"]]
        m["max_prompt_tokens"] = max(prompt_toks) if prompt_toks else None
        m["paths_answerable"] = dict(m["paths_answerable"])
        m["paths_ood"] = dict(m["paths_ood"])
        out["per_seed"][s] = m

    # across seeds: mean +- std of per-seed means, and bootstrap CI over questions of per-question seed-mean
    keys = ["anls", "em", "contains", "answered", "cited_gold_page", "refusal_rate_ood_trivial", "refusal_rate_ood_hard",
            "e2e_refusal_accuracy", "e2e_refusal_accuracy_if_sentinel_normalized",
            "refusal_rate_ood_trivial_if_sentinel_normalized", "refusal_rate_ood_hard_if_sentinel_normalized",
            "claim_flag_rate(unsupported)", "answers_with_>=1_flagged_claim",
            "answers_fully_refused_by_firewall", "claim_flag_rate_answerable", "claim_flag_rate_ood"]
    out["across_seeds_mean_std"] = {k: mean_std([out["per_seed"][s][k] for s in seeds if out["per_seed"][s][k] is not None]) for k in keys}
    if all("closed_book" in out["per_seed"][s] for s in seeds):
        out["across_seeds_mean_std"].update({f"closed_book_{k}": mean_std([out["per_seed"][s]["closed_book"][k] for s in seeds]) for k in ("anls", "em", "contains")})
    out["bootstrap_ci_over_questions(seed-averaged)"] = {
        k: bootstrap_ci([np.mean(v) for v in d.values()]) for k, d in per_q_metrics.items()}
    # pooled claim flag rate over all complete seeds, cluster bootstrap over questions
    llm_all = [r for r in rag if r["seed"] in seeds and r["path"] in ("answered", "nli_firewall_refusal")]
    by_q = defaultdict(lambda: [0, 0])
    for r in llm_all:
        by_q[r["id"]][0] += sum(1 for c in r["claims"] if not c["supported"])
        by_q[r["id"]][1] += len(r["claims"])
    qs = [q for q in by_q if by_q[q][1]]
    flags = np.array([by_q[q][0] for q in qs], float)
    tots = np.array([by_q[q][1] for q in qs], float)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(10000):
        ii = rng.integers(0, len(qs), len(qs))
        boots.append(flags[ii].sum() / tots[ii].sum())
    out["claim_flag_rate_pooled_cluster_ci"] = [float(flags.sum() / tots.sum()), float(np.percentile(boots, 2.5)),
                                                float(np.percentile(boots, 97.5)), int(tots.sum()), len(qs)]
    print("pooled claim flag rate", out["claim_flag_rate_pooled_cluster_ci"])
    out["decision_metrics"] = decision_metrics(rag, seeds)
    print(json.dumps(out["decision_metrics"], indent=1))
    stem = Path(path).stem
    (RESULTS / f"{stem}_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["across_seeds_mean_std"], indent=1))
    print(json.dumps(out["bootstrap_ci_over_questions(seed-averaged)"], indent=1))
    for s in seeds:
        print(s, out["per_seed"][s]["paths_answerable"], out["per_seed"][s]["paths_ood"], "table:", out["per_seed"][s]["table_shortcut"]["n"],
              "maxprompt", out["per_seed"][s]["max_prompt_tokens"])

    if "--export-claims" in sys.argv:  # optional; no headline depends on it
        export_claims_for_labeling(rag, seeds[0])



# ---------------------------------------------------------------------------
# Review fixes (items 1 and 4): decision metrics that do not reward wrong answers,
# balanced accuracy, a post-hoc "firewall off" counterfactual, and the firewall's
# refusal rate on drafts that contain the gold answer -- all with cluster
# bootstrap CIs over QUESTIONS (each question's 3 seed replies resampled together).
# Computed only from the existing generation JSONL; no LLM calls.
# ---------------------------------------------------------------------------
def _is_sentinel(text):
    return bool(text) and "not_in_documents" in text.lower()


def _cluster_boot_ratio(num, den, groups_mask=None, n=10000, seed=0):
    """Vectorised cluster bootstrap. num/den: per-question arrays (shape [Q] or [K, Q]
    for K sub-statistics). Returns point/CI of sum(num)/sum(den) per row."""
    num, den = np.atleast_2d(num).astype(float), np.atleast_2d(den).astype(float)
    Q = num.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, Q, size=(n, Q))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = num[:, idx].sum(-1) / den[:, idx].sum(-1)  # [K, n]
        point = num.sum(-1) / den.sum(-1)
    return point, boots


def _ci(point, boots):
    b = boots[~np.isnan(boots)]
    return [float(point), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def decision_metrics(rag, seeds):
    rs = [r for r in rag if r["seed"] in seeds]
    by_q = defaultdict(list)
    for r in rs:
        by_q[r["id"]].append(r)
    kind = {q: v[0]["kind"] for q, v in by_q.items()}

    def outcome(r, policy):
        """Return (decided_correctly_lenient, decided_correctly_strict) for one reply.
        policy 'system': production behaviour as logged.
        policy 'firewall_off': post-hoc counterfactual -- the LLM draft is shown unless
          the gate refused, the reply is the exact sentinel, or the table path answered;
          i.e. the NLI firewall never refuses. Uses raw_llm_output (no new calls)."""
        if policy == "system":
            refused = bool(r["refused"])
            text = None if refused else (r["answer"] or "")
        else:
            refused = r["path"] in ("gate_refusal", "llm_not_in_documents")
            text = None if refused else (r["raw_llm_output"] or r["answer"] or "")
        if r["kind"] == "answerable":
            lenient = not refused
            strict = (not refused) and (not _is_sentinel(text)) and contains_match(text, r["gold_answers"]) == 1.0
        else:
            lenient = refused
            strict = refused or _is_sentinel(text)  # a sentinel reply on OOD is a de-facto refusal
            # keep 'lenient' = production refusal flag only, so literal sentinel replies count as answers there
        return float(lenient), float(strict)

    qids = sorted(by_q)
    is_ans = np.array([kind[q] == "answerable" for q in qids])

    def stats_for(policy, which):
        idx = 0 if which == "lenient" else 1
        corr = np.array([sum(outcome(r, policy)[idx] for r in by_q[q]) for q in qids], float)
        cnt = np.array([len(by_q[q]) for q in qids], float)
        # rows: plain, answerable-acc, ood-acc
        num = np.vstack([corr, corr * is_ans, corr * ~is_ans])
        den = np.vstack([cnt, cnt * is_ans, cnt * ~is_ans])
        point, boots = _cluster_boot_ratio(num, den)
        bal_point = 0.5 * (point[1] + point[2])
        bal_boots = 0.5 * (boots[1] + boots[2])
        return {"plain_acc": _ci(point[0], boots[0]), "balanced_acc": _ci(bal_point, bal_boots),
                "answerable_acc": _ci(point[1], boots[1]), "ood_acc": _ci(point[2], boots[2])}

    out = {"n_questions": len(by_q), "n_replies": len(rs),
           "definitions": {
               "lenient": "answerable counted correct iff not refused (any non-refusal, even a wrong answer or a literal sentinel reply); OOD correct iff refused flag set",
               "strict": "answerable correct iff not refused AND reply is not a NOT_IN_DOCUMENTS variant AND gold answer contained in reply; OOD correct iff refused or reply is a sentinel variant",
               "never_refuse_baseline": "lenient: plain = share answerable, balanced = 0.5"}}
    for policy in ("system", "firewall_off"):
        for which in ("lenient", "strict"):
            out[f"{policy}_{which}"] = stats_for(policy, which)
    n_ans = sum(1 for q in by_q if kind[q] == "answerable")
    out["never_refuse_baseline_lenient"] = {"plain_acc": n_ans / len(by_q), "balanced_acc": 0.5}

    # item 4: firewall refusal on answerable LLM drafts, split by whether the draft contains the gold answer
    drafts = [r for r in rs if r["kind"] == "answerable" and r["path"] in ("answered", "nli_firewall_refusal")]
    d_by_q = defaultdict(list)
    for r in drafts:
        d_by_q[r["id"]].append((contains_match(r["raw_llm_output"] or r["answer"] or "", r["gold_answers"]) == 1.0,
                                r["path"] == "nli_firewall_refusal"))

    dq = sorted(d_by_q)
    nc = np.array([sum(1 for c, _ in d_by_q[q] if c) for q in dq], float)
    rc = np.array([sum(1 for c, ref in d_by_q[q] if c and ref) for q in dq], float)
    nw = np.array([sum(1 for c, _ in d_by_q[q] if not c) for q in dq], float)
    rw = np.array([sum(1 for c, ref in d_by_q[q] if (not c) and ref) for q in dq], float)
    point, boots = _cluster_boot_ratio(np.vstack([rc, rw]), np.vstack([nc, nw]))
    out["firewall_refusal_on_drafts"] = {
        "n_drafts": len(drafts), "n_questions": len(dq),
        "n_drafts_containing_gold": int(nc.sum()), "n_drafts_lacking_gold": int(nw.sum()),
        "refused_given_draft_contains_gold": _ci(point[0], boots[0]),
        "refused_given_draft_lacks_gold": _ci(point[1], boots[1]),
        "gap_lacks_minus_contains": _ci(point[1] - point[0], boots[1] - boots[0]),
    }
    # literal sentinel inventory (exact strings)
    sent = [r["raw_llm_output"].strip() for r in rs if _is_sentinel(r.get("raw_llm_output")) and not r["refused"]]
    out["sentinel_replies_not_refused"] = dict(Counter(sent))
    out["nli_score_of_sentinel_claims"] = sorted({round(c["score"], 3) for r in rs if _is_sentinel(r.get("raw_llm_output")) and not r["refused"] for c in r["claims"]})
    return out


def export_claims_for_labeling(rag, seed, n_target=100):
    """Stratified sample of LLM claims (seed 0) for human labeling of the NLI gate."""
    from common import load_or_ingest
    from app.retrieve.hybrid import retrieve
    from app.verify.nli import verify_claims

    sid, index, _ = load_or_ingest()
    rows = []
    for r in rag:
        if r["seed"] != seed or r["path"] not in ("answered", "nli_firewall_refusal"):
            continue
        results = retrieve(index, r["question"], mode="hybrid", k=5, use_rerank=True)
        claims = verify_claims(r["answer"], results)  # deterministic re-run -> recover full evidence chunk
        by_key = {(res["chunk"]["page"], tuple(res["chunk"]["bbox"])): res["chunk"] for res in results}
        for j, c in enumerate(claims):
            cit = c.citations[0] if c.citations else None
            ev = by_key.get((cit.page, tuple(cit.bbox))) if cit else None
            rows.append({"claim_id": f"{r['id']}#c{j}", "qid": r["id"], "kind": r["kind"], "question": r["question"],
                         "gold_answers": " || ".join(r["gold_answers"]), "claim": c.text, "nli_entail_prob": round(c.score, 4),
                         "gate_supported": int(c.supported), "evidence_page": cit.page if cit else None,
                         "evidence_text": (ev.get("text") or ev.get("caption_text") or "") if ev else ""})
    rng = np.random.default_rng(0)
    sup = [x for x in rows if x["gate_supported"]]
    uns = [x for x in rows if not x["gate_supported"]]
    k_uns = min(len(uns), n_target // 2)
    k_sup = min(len(sup), n_target - k_uns)
    pick = [sup[i] for i in rng.choice(len(sup), k_sup, replace=False)] + [uns[i] for i in rng.choice(len(uns), k_uns, replace=False)]
    pick = [pick[i] for i in rng.permutation(len(pick))]
    cols_public = ["claim_id", "qid", "kind", "question", "gold_answers", "claim", "nli_entail_prob", "gate_supported", "evidence_page"]
    with open(DATA / "claims_for_human_labeling.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols_public + ["human_label(supported/unsupported)", "notes"])
        for x in pick:
            w.writerow([x[c] for c in cols_public] + ["", ""])
    with open(CACHE / "claims_for_human_labeling_WITH_EVIDENCE.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols_public + ["evidence_text", "human_label(supported/unsupported)", "notes"])
        for x in pick:
            w.writerow([x[c] for c in cols_public] + [x["evidence_text"], "", ""])
    print(f"exported {len(pick)} claims ({k_sup} gate-supported, {k_uns} gate-flagged) of {len(rows)} total")


if __name__ == "__main__":
    main(sys.argv[1])
