# backend/app/session.py
"""In-process store for ingested documents, keyed by session id.

# ponytail: in-process single-process store; move to shared/disk only if multi-replica.
# ponytail: bounded LRU (OrderedDict) capped at settings.max_sessions -- each
# session holds page PNGs + FAISS/CLIP indexes, so an unbounded dict would OOM
# a long-lived Space. Oldest/least-recently-used session is evicted on overflow.
"""
import threading
from collections import OrderedDict
from uuid import uuid4

from app.config import settings

_sessions: "OrderedDict[str, dict]" = OrderedDict()

# Task 2: per-session lock so two concurrent get_index() calls for the SAME
# session can't both see "index" missing and both build (and embed) it --
# wasted CPU at best, a race on session["index"] at worst. `_locks_guard`
# only protects the lock-creation step itself (fast), not index building.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def create_session(pages: list[dict], chunks: list[dict]) -> str:
    """Store a document's pages + chunks and return a new session id.
    Evicts the oldest session first if already at the cap."""
    if len(_sessions) >= settings.max_sessions:
        evicted_id, _ = _sessions.popitem(last=False)
        _locks.pop(evicted_id, None)
    session_id = uuid4().hex
    _sessions[session_id] = {"pages": pages, "chunks": chunks}
    return session_id


def get_session(session_id: str) -> dict | None:
    """Return ``{"pages": [...], "chunks": [...]}`` or ``None`` if unknown."""
    session = _sessions.get(session_id)
    if session is not None:
        _sessions.move_to_end(session_id)
    return session


def get_index(session_id: str):
    """Return the session's ``Index`` (Task 2.2), building + caching it on first
    call so later calls reuse it instead of re-embedding every request.
    Returns ``None`` if the session is unknown."""
    session = _sessions.get(session_id)
    if session is None:
        return None
    _sessions.move_to_end(session_id)
    with _locks_guard:
        lock = _locks.setdefault(session_id, threading.Lock())
    with lock:
        if "index" not in session:
            from app.index.store import Index

            idx = Index()
            idx.add(session["chunks"], pages=session["pages"])
            session["index"] = idx
    return session["index"]
