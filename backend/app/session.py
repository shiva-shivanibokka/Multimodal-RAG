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
