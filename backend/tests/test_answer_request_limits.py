# backend/tests/test_answer_request_limits.py
"""Task 5: /answer request-size cap -- oversized string fields are rejected
with 422 before any retrieval/provider work happens."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_oversized_question_rejected_with_422():
    r = client.post(
        "/answer",
        json={
            "question": "x" * 4001,
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "api_key": "fake-key",
        },
    )
    assert r.status_code == 422


def test_oversized_model_rejected_with_422():
    r = client.post(
        "/answer",
        json={
            "question": "hi",
            "provider": "groq",
            "model": "x" * 201,
            "api_key": "fake-key",
        },
    )
    assert r.status_code == 422


def test_oversized_api_key_rejected_with_422():
    r = client.post(
        "/answer",
        json={
            "question": "hi",
            "provider": "groq",
            "model": "x",
            "api_key": "k" * 501,
        },
    )
    assert r.status_code == 422


def test_question_at_cap_boundary_not_rejected_by_length_alone():
    # 4000 chars is within the cap -- should get past validation (may still
    # 401 if BACKEND_TOKEN happens to be set by another test's env, but that
    # proves it passed body validation, not that it 422'd on length).
    r = client.post(
        "/answer",
        json={
            "question": "x" * 4000,
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "api_key": "fake-key",
        },
    )
    assert r.status_code != 422
