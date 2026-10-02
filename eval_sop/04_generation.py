"""Step 4: end-to-end generation through the PRODUCTION answer_question()
(gate -> hybrid retrieve+rerank -> table shortcut -> LLM -> NLI firewall),
with a LOCAL model served by Ollama (free; no paid API is ever called).

How the local model is wired in without touching product code:
  AnswerRequest.provider is Literal[openai|groq|gemini|anthropic], so we pass
  provider="openai" and, in THIS process only, point providers._OPENAI_COMPAT
  ["openai"] at Ollama's OpenAI-compatible endpoint. A guard in the _post
  wrapper aborts if any request would leave localhost. The wrapper also injects
  temperature/seed (the product sends neither) and logs token usage so prompt
  truncation can be detected.

Also runs a closed-book baseline (same model, no context) for answer quality.

Usage:  python 04_generation.py --model llama3.2:latest --seeds 0 1 2 --temperature 0.7
"""
import argparse
import json
import time

import common  # noqa: F401
from common import DATA, RESULTS, install_rerank_cache, load_or_ingest

install_rerank_cache()

import app.generate.answer as answer_mod
import app.generate.providers as providers
from app.schemas import AnswerRequest

# Ollama OpenAI-compatible endpoint; overridable with --ollama-url (must be localhost).
OLLAMA = "http://localhost:11434/v1"
_STATE = {"temperature": 0.0, "seed": 0, "usage": [], "gen_calls": 0, "table_path": False, "raw": None}

_orig_post = providers._post


def _local_post(url, headers, payload, timeout):
    if not (url.startswith("http://localhost:") or url.startswith("http://127.0.0.1:")):
        raise RuntimeError(f"refusing non-local LLM call to {url}")
    payload = dict(payload, temperature=_STATE["temperature"], seed=_STATE["seed"])
    r = _orig_post(url, headers, payload, 1800)  # local model under load can be slow; product default 120s
    try:
        _STATE["usage"].append(r.json().get("usage", {}))
    except Exception:
        pass
    return r


providers._post = _local_post
providers._OPENAI_COMPAT["openai"] = OLLAMA

_orig_generate = answer_mod.generate
_orig_table = answer_mod.try_table_answer


def _gen(*a, **k):
    _STATE["gen_calls"] += 1
    out = _orig_generate(*a, **k)
    _STATE["raw"] = out
    return out


def _table(q, results):
    out = _orig_table(q, results)
    _STATE["table_path"] = out is not None
    return out


answer_mod.generate = _gen
answer_mod.try_table_answer = _table

CLOSED_BOOK_SYS = "Answer the question as briefly as possible."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:latest")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-closed-book", action="store_true")
    ap.add_argument("--ollama-url", default=OLLAMA)
    ap.add_argument("--num-ctx-check", type=int, default=8192,
                    help="warn if a prompt's prompt_tokens reaches this (server-side truncation)")
    args = ap.parse_args()
    providers._OPENAI_COMPAT["openai"] = args.ollama_url

    sid, index, d2p = load_or_ingest()
    gold = [g for g in json.loads((DATA / "gold_all151.json").read_text(encoding="utf-8")) if not g["ambiguous"]]
    ood = json.loads((DATA / "ood_questions.json").read_text(encoding="utf-8"))
    items = [(g["id"], g["question"], "answerable", g["answers"], g["source_doc"]) for g in gold]
    items += [(o["id"], o["question"], o["ood_type"], [], None) for o in ood]
    if args.limit:
        items = items[: args.limit]

    _STATE["temperature"] = args.temperature
    out_path = RESULTS / f"generation_{args.model.replace(':', '_')}_T{args.temperature}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["id"], r["setting"]))
    f = open(out_path, "a", encoding="utf-8")

    for seed in args.seeds:
        _STATE["seed"] = seed
        for qid, q, kind, answers, src in items:
            if (seed, qid, "rag") in done:
                continue
            _STATE.update(gen_calls=0, table_path=False, raw=None, usage=[])
            t = time.time()
            req = AnswerRequest(question=q, provider="openai", model=args.model, api_key="ollama-local",
                                session_id=sid, retrieval_mode="hybrid", verified=True)
            err = None
            try:
                resp = answer_mod.answer_question(req)
            except Exception as e:  # record, don't hide
                resp, err = None, repr(e)
            if resp is None:
                path = "error"
            elif _STATE["table_path"]:
                path = "table_shortcut"
            elif _STATE["gen_calls"] == 0:
                path = "gate_refusal"
            elif _STATE["raw"] is not None and _STATE["raw"].strip() == "NOT_IN_DOCUMENTS":
                path = "llm_not_in_documents"
            elif resp.refused:
                path = "nli_firewall_refusal"
            else:
                path = "answered"
            rec = {
                "setting": "rag", "seed": seed, "temperature": args.temperature, "model": args.model,
                "id": qid, "kind": kind, "question": q, "gold_answers": answers, "source_doc": src,
                "source_page": d2p.get(src) if src else None,
                "path": path, "error": err,
                "raw_llm_output": _STATE["raw"],
                "answer": resp.answer if resp else None,
                "refused": resp.refused if resp else None,
                "claims": [{"text": c.text, "supported": c.supported, "score": c.score,
                            "cited_page": c.citations[0].page if c.citations else None} for c in (resp.claims if resp else [])],
                "cited_pages": sorted({c.page for c in resp.citations}) if resp else [],
                "usage": _STATE["usage"], "secs": round(time.time() - t, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(seed, qid, path, f"{rec['secs']}s", (rec["answer"] or "")[:60].replace("\n", " "), flush=True)

        if args.no_closed_book:
            continue
        for qid, q, kind, answers, src in items:
            if kind != "answerable" or (seed, qid, "closed_book") in done:
                continue
            _STATE.update(usage=[])
            msgs = [{"role": "system", "content": CLOSED_BOOK_SYS}, {"role": "user", "content": q}]
            try:
                txt, err = providers.generate("openai", args.model, "ollama-local", msgs), None
            except Exception as e:
                txt, err = None, repr(e)
            rec = {"setting": "closed_book", "seed": seed, "temperature": args.temperature, "model": args.model,
                   "id": qid, "kind": kind, "question": q, "gold_answers": answers, "answer": txt, "error": err}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    f.close()


if __name__ == "__main__":
    main()
