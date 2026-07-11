# backend/tests/test_answer.py
"""Task 2.4: retrieval wired into /answer, grounding-gated refusal + citations.

No network calls: ``app.generate.answer.generate`` (the name imported into
the answer module) is always monkeypatched with a mock, so a live provider
API is never hit.
"""
from unittest.mock import MagicMock

from app.generate.answer import answer_question
from app.schemas import AnswerRequest
from app.session import create_session

# Well-grounded fixture: a finance chunk whose text directly answers the
# revenue question (dense cosine ~0.69, far above retrieval_min_score=0.25),
# plus an unrelated HR chunk (page 7) so retrieval has to pick the right one.
REVENUE_CHUNK = {
    "id": 0,
    "kind": "text",
    "text": (
        "The quarterly earnings report shows total revenue of 4.2 million "
        "dollars for Q3, driven by strong enterprise software sales."
    ),
    "page": 3,
    "bbox": [0, 0, 10, 10],
    "table_df_json": None,
}
HR_CHUNK = {
    "id": 1,
    "kind": "text",
    "text": (
        "Employee onboarding requires completing tax forms, signing the "
        "confidentiality agreement, and setting up direct deposit within "
        "the first week."
    ),
    "page": 7,
    "bbox": [0, 0, 10, 10],
    "table_df_json": None,
}
SAFETY_CHUNK = {
    "id": 2,
    "kind": "text",
    "text": (
        "New hires must complete the mandatory workplace safety training "
        "module within thirty days of their start date."
    ),
    "page": 8,
    "bbox": [0, 0, 10, 10],
    "table_df_json": None,
}


def _req(session_id, question, retrieval_mode="hybrid"):
    return AnswerRequest(
        question=question,
        provider="groq",
        model="llama-3.1-8b-instant",
        api_key="fake-key",
        session_id=session_id,
        retrieval_mode=retrieval_mode,
    )


def test_grounded_answer_has_citations_and_uses_retrieved_context(monkeypatch):
    session_id = create_session(pages=[], chunks=[REVENUE_CHUNK, HR_CHUNK])

    mock_generate = MagicMock(return_value="The answer is X.")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    resp = answer_question(_req(session_id, "What was the total revenue in Q3?"))

    assert resp.refused is False
    assert resp.answer == "The answer is X."
    assert len(resp.citations) >= 1
    assert any(c.page == REVENUE_CHUNK["page"] for c in resp.citations)

    # context actually built from retrieval, not hardcoded
    messages = mock_generate.call_args.args[3]
    user_content = messages[-1]["content"]
    assert "[page" in user_content
    assert "4.2 million dollars" in user_content


def test_refusal_when_no_session(monkeypatch):
    mock_generate = MagicMock(return_value="should not be called")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    resp = answer_question(_req(None, "What was the total revenue in Q3?"))

    assert resp.refused is True
    assert resp.answer == ""
    mock_generate.assert_not_called()


def test_refusal_on_low_grounding(monkeypatch):
    # Topic A: HR onboarding docs. Topic B: a totally unrelated art-history
    # question. Observed top dense cosine for this pair is 0.2286 (see
    # task-2.4-report.md), comfortably under retrieval_min_score=0.25 and
    # deterministic (no dropout at inference) -- not flaky.
    session_id = create_session(pages=[], chunks=[HR_CHUNK, SAFETY_CHUNK])

    mock_generate = MagicMock(return_value="should not be called")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    resp = answer_question(
        _req(session_id, "What type of paint did Vincent van Gogh use for Starry Night?")
    )

    assert resp.refused is True
    assert resp.answer == ""
    mock_generate.assert_not_called()


def test_refusal_when_model_says_not_in_documents(monkeypatch):
    session_id = create_session(pages=[], chunks=[REVENUE_CHUNK, HR_CHUNK])

    mock_generate = MagicMock(return_value="NOT_IN_DOCUMENTS")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    resp = answer_question(_req(session_id, "What was the total revenue in Q3?"))

    assert resp.refused is True
    assert resp.answer == ""
    mock_generate.assert_called_once()
