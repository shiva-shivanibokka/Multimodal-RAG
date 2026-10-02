"""Shared helpers for the SOP eval (eval_sop/).

- Puts backend/ on sys.path so `app.*` / `eval.*` import exactly as in production.
- `load_or_ingest()` runs the ORIGINAL ingestion (backend/eval/run_eval.py::ingest_corpus:
  load -> docTR OCR -> img2table -> chunk) ONCE and pickles pages+chunks to
  eval_sop/cache/ (gitignored: it contains OCR text of DocVQA images, which are
  research-use licensed and not redistributed). Later scripts rebuild the Index
  from the cache, so every experiment uses the identical OCR output.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
EVAL_SOP = ROOT / "eval_sop"
CACHE = EVAL_SOP / "cache"
RESULTS = EVAL_SOP / "results"
DATA = EVAL_SOP / "data"
CORPUS = BACKEND / "eval" / "corpus"

os.environ.setdefault("HF_HOME", "C:/mrag/.cache")
os.environ.setdefault("TORCH_HOME", "C:/mrag/.cache")
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Speed only (this machine was shared with other heavy jobs): cap torch threads to
# limit CPU oversubscription. Does not change any model output.
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("EVAL_THREADS", "6")))


def install_rerank_cache():
    """Eval-only speedup: memoize bge-reranker scores per (query, passage) pair in a
    sqlite file under cache/ so repeated (config, question) retrievals across scripts
    don't recompute the cross-encoder. Uncached pairs are still scored by the
    production model through the same CrossEncoder.predict call. Caveat: scoring a
    pair inside a different batch can change the float by ~1e-6 (padding), which
    could only flip exact ties. Product code is untouched (monkeypatch in-process)."""
    import hashlib
    import sqlite3

    import numpy as np
    import app.index.rerank as rr

    CACHE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(CACHE / "rerank_cache.sqlite"), timeout=60)
    db.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY, v REAL)")
    real_get = rr._get_model

    class _Cached:
        def __init__(self, m):
            self.m = m

        def predict(self, pairs, **kw):
            keys = [hashlib.sha1((q + "\x00" + p).encode("utf-8")).hexdigest() for q, p in pairs]
            found = {}
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                q = f"SELECT k, v FROM s WHERE k IN ({','.join('?' * len(part))})"
                found.update(dict(db.execute(q, part).fetchall()))
            miss = [i for i, k in enumerate(keys) if k not in found]
            if miss:
                vals = self.m.predict([pairs[i] for i in miss], **kw)
                db.executemany("INSERT OR REPLACE INTO s VALUES (?, ?)",
                               [(keys[i], float(v)) for i, v in zip(miss, vals)])
                db.commit()
                for i, v in zip(miss, vals):
                    found[keys[i]] = float(v)
            return np.array([found[k] for k in keys], dtype=np.float32)

    _wrapped = {}

    def _get():
        if "m" not in _wrapped:
            _wrapped["m"] = _Cached(real_get())
        return _wrapped["m"]

    rr._get_model = _get


def load_or_ingest():
    """Return (session_id, index, doc_to_page). Ingests once, then reuses cache."""
    from app.session import create_session, get_index, get_session

    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / "ingested.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            blob = pickle.load(f)
        sid = create_session(blob["pages"], blob["chunks"])
        doc_to_page = blob["doc_to_page"]
    else:
        from eval.run_eval import ingest_corpus

        sid, doc_to_page = ingest_corpus(CORPUS)
        s = get_session(sid)
        with open(cache_file, "wb") as f:
            pickle.dump({"pages": s["pages"], "chunks": s["chunks"], "doc_to_page": doc_to_page}, f)
    index = get_index(sid)
    return sid, index, doc_to_page
