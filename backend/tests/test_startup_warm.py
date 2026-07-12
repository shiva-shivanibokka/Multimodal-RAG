# backend/tests/test_startup_warm.py
"""Task 1: model singletons are warmed once, serially, at app startup
instead of racing on the first concurrent requests. Every loader is stubbed
to a call-counting function so this test needs no real model weights."""
from fastapi.testclient import TestClient

from app import main
from app.index import embedders, rerank
from app.ingest import ocr, tables
from app.verify import nli


def test_startup_warms_each_model_loader_exactly_once(monkeypatch):
    calls = {"model": 0, "clip": 0, "reranker": 0, "nli": 0, "predictor": 0, "table_ocr": 0}

    def counter(key):
        def _f():
            calls[key] += 1
        return _f

    monkeypatch.setattr(embedders, "_get_model", counter("model"))
    monkeypatch.setattr(embedders, "_get_clip_model", counter("clip"))
    monkeypatch.setattr(rerank, "_get_model", counter("reranker"))
    monkeypatch.setattr(nli, "_get_model", counter("nli"))
    monkeypatch.setattr(ocr, "_get_predictor", counter("predictor"))
    monkeypatch.setattr(tables, "_get_ocr", counter("table_ocr"))

    with TestClient(main.app):
        pass  # entering/exiting the context manager runs the lifespan startup event

    assert calls == {
        "model": 1,
        "clip": 1,
        "reranker": 1,
        "nli": 1,
        "predictor": 1,
        "table_ocr": 1,
    }
