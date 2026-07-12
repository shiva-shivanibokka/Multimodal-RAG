# backend/tests/test_session.py
import threading
import time

from app import config, session
from app.index.store import Index


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


# --- Task 2: get_index concurrency ---


def test_get_index_concurrent_calls_build_index_exactly_once(monkeypatch):
    calls = {"n": 0}

    def fake_add(self, chunks, pages=None):
        calls["n"] += 1
        time.sleep(0.05)  # widen the race window so two threads actually overlap
        self._chunks = []

    monkeypatch.setattr(Index, "add", fake_add)

    session_id = session.create_session(pages=[], chunks=[])

    results: list = [None, None]

    def worker(i):
        results[i] = session.get_index(session_id)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert calls["n"] == 1
    assert results[0] is results[1]
    assert results[0] is not None


def test_locks_cleaned_up_on_session_eviction(monkeypatch):
    monkeypatch.setattr(config.settings, "max_sessions", 1)
    monkeypatch.setattr(session, "_sessions", type(session._sessions)())
    monkeypatch.setattr(session, "_locks", {})

    first = session.create_session(pages=[], chunks=[])
    session.get_index(first)  # populates _locks[first]
    assert first in session._locks

    session.create_session(pages=[], chunks=[])  # evicts "first"

    assert first not in session._locks
