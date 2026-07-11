# backend/tests/test_session.py
from app import config, session


def test_sessions_evicted_beyond_cap(monkeypatch):
    monkeypatch.setattr(config.settings, "max_sessions", 5)
    monkeypatch.setattr(session, "_sessions", type(session._sessions)())

    ids = [session.create_session(pages=[], chunks=[]) for _ in range(7)]

    assert len(session._sessions) == 5
    # earliest evicted
    assert session.get_session(ids[0]) is None
    assert session.get_session(ids[1]) is None
    # most recent survive
    for sid in ids[-5:]:
        assert session.get_session(sid) is not None


def test_get_session_refreshes_recency(monkeypatch):
    monkeypatch.setattr(config.settings, "max_sessions", 3)
    monkeypatch.setattr(session, "_sessions", type(session._sessions)())

    a = session.create_session(pages=[], chunks=[])
    b = session.create_session(pages=[], chunks=[])
    c = session.create_session(pages=[], chunks=[])

    # touch "a" so it's no longer the least-recently-used
    session.get_session(a)

    d = session.create_session(pages=[], chunks=[])  # overflow: evicts "b", not "a"

    assert session.get_session(b) is None
    assert session.get_session(a) is not None
    assert session.get_session(c) is not None
    assert session.get_session(d) is not None
