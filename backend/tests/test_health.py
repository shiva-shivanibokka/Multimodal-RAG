# backend/tests/test_health.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_protected_requires_token(monkeypatch):
    monkeypatch.setenv("BACKEND_TOKEN", "secret")
    r = client.post("/answer", json={"question": "hi", "provider": "groq", "model": "x", "api_key": "k"})
    assert r.status_code == 401
