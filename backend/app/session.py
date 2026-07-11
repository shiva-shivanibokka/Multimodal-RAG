# backend/app/session.py
"""In-process store for ingested documents, keyed by session id.

# ponytail: in-process single-process store; move to shared/disk only if multi-replica.
# ponytail: bounded LRU (OrderedDict) capped at settings.max_sessions -- each
# session holds page PNGs + FAISS/CLIP indexes, so an unbounded dict would OOM
# a long-lived Space. Oldest/least-recently-used session is evicted on overflow.
"""
from collections import OrderedDict
from uuid import uuid4

from app.config import settings

_sessions: "OrderedDict[str, dict]" = OrderedDict()


def create_session(pages: list[dict], chunks: list[dict]) -> str:
    """Store a document's pages + chunks and return a new session id.
    Evicts the oldest session first if already at the cap."""
    if len(_sessions) >= settings.max_sessions:
        _sessions.popitem(last=False)
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
    if "index" not in session:
        from app.index.store import Index

        idx = Index()
        idx.add(session["chunks"], pages=session["pages"])
        session["index"] = idx
    return session["index"]
