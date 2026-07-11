# backend/app/session.py
"""In-process store for ingested documents, keyed by session id.

# ponytail: in-process single-process store; move to shared/disk only if multi-replica.
"""
from uuid import uuid4

_sessions: dict[str, dict] = {}


def create_session(pages: list[dict], chunks: list[dict]) -> str:
    """Store a document's pages + chunks and return a new session id."""
    session_id = uuid4().hex
    _sessions[session_id] = {"pages": pages, "chunks": chunks}
    return session_id


def get_session(session_id: str) -> dict | None:
    """Return ``{"pages": [...], "chunks": [...]}`` or ``None`` if unknown."""
    return _sessions.get(session_id)


def get_index(session_id: str):
    """Return the session's ``Index`` (Task 2.2), building + caching it on first
    call so later calls reuse it instead of re-embedding every request.
    Returns ``None`` if the session is unknown."""
    session = _sessions.get(session_id)
    if session is None:
        return None
    if "index" not in session:
        from app.index.store import Index

        idx = Index()
        idx.add(session["chunks"], pages=session["pages"])
        session["index"] = idx
    return session["index"]
