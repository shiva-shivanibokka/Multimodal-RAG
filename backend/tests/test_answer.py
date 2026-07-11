# backend/tests/test_answer.py
"""Task 2.4: retrieval wired into /answer, grounding-gated refusal + citations.

No network calls: ``app.generate.answer.generate`` (the name imported into
the answer module) is always monkeypatched with a mock, so a live provider
API is never hit.
"""
from unittest.mock import MagicMock

from app.generate.answer import answer_question
from app.ingest.loader import load_document
from app.schemas import AnswerRequest
from app.session import create_session
from tests.fixtures import make_scanned_pdf, make_solid_image

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


def _req(session_id, question, retrieval_mode="hybrid", verified=True):
    return AnswerRequest(
        question=question,
        provider="groq",
        model="llama-3.1-8b-instant",
        api_key="fake-key",
        session_id=session_id,
        retrieval_mode=retrieval_mode,
        verified=verified,
    )


def _figure_chunk(chunk_id, page, bbox):
    return {
        "id": chunk_id,
        "kind": "figure",
        "text": "",
        "page": page,
        "bbox": bbox,
        "table_df_json": None,
    }


def test_grounded_answer_has_citations_and_uses_retrieved_context(monkeypatch):
    # Pre-4.2 citation/context-routing test -- unrelated to the NLI
    # firewall, so verified=False keeps it testing the retrieval path only
    # (the canned "The answer is X." isn't literally NLI-entailed by the
    # evidence text; faithfulness itself is covered by the Task 4.2 tests
    # below).
    session_id = create_session(pages=[], chunks=[REVENUE_CHUNK, HR_CHUNK])

    mock_generate = MagicMock(return_value="The answer is X.")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    resp = answer_question(_req(session_id, "What was the total revenue in Q3?", verified=False))

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


# --- Task 3.4: mode routing + figure images to the VLM ---


def test_cross_modal_mode_passes_figure_image_and_returns_figure_citation(monkeypatch):
    # Solid-red page image + one figure chunk over it. Query "a red image"
    # observed CLIP cosine against this fixture: 0.3139 -- comfortably above
    # retrieval_min_score=0.25 (see task-3.4-report.md for how this was
    # measured; not flaky, no dropout at inference).
    pages = [{"index": 0, "image_png": make_solid_image("red"), "width": 224, "height": 224}]
    chunks = [_figure_chunk(0, 0, [0, 0, 224, 224])]
    session_id = create_session(pages=pages, chunks=chunks)

    mock_generate = MagicMock(return_value="It is red.")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    # verified=False: this test is about mode routing/citations, not the
    # NLI firewall (see Task 4.2 tests below for that).
    resp = answer_question(_req(session_id, "a red image", retrieval_mode="cross_modal", verified=False))

    assert resp.refused is False
    assert resp.answer == "It is red."
    mock_generate.assert_called_once()

    images = mock_generate.call_args.kwargs["images"]
    assert len(images) == 1
    assert isinstance(images[0], str) and images[0]

    assert len(resp.citations) >= 1
    assert any(c.page == 0 for c in resp.citations)


def test_caption_baseline_mode_routes_to_ocr_captioned_figure(monkeypatch):
    # Reuse the proven OCR fixture (INVOICE TOTAL) from test_store.py: page
    # image with legible text -> non-empty caption -> indexed for
    # caption_baseline. Observed bge cosine for "invoice total" against this
    # fixture: 0.9498, far above retrieval_min_score=0.25.
    scanned_page = load_document(make_scanned_pdf("INVOICE TOTAL"))[0]
    pages = [
        {
            "index": 0,
            "image_png": scanned_page["image_png"],
            "width": scanned_page["width"],
            "height": scanned_page["height"],
        }
    ]
    chunks = [_figure_chunk(0, 0, [0, 0, scanned_page["width"], scanned_page["height"]])]
    session_id = create_session(pages=pages, chunks=chunks)

    mock_generate = MagicMock(return_value="The invoice total is $100.")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    # verified=False: this test is about mode routing/citations, not the
    # NLI firewall (see Task 4.2 tests below for that).
    resp = answer_question(
        _req(session_id, "invoice total", retrieval_mode="caption_baseline", verified=False)
    )

    assert resp.refused is False
    mock_generate.assert_called_once()

    images = mock_generate.call_args.kwargs["images"]
    assert len(images) == 1

    assert len(resp.citations) >= 1
    assert any(c.page == 0 for c in resp.citations)


# --- Task 4.2: NLI verification + calibrated refusal ---

# Evidence chunk text is the exact string proven in test_nli.py to entail
# "Revenue was 4 million dollars." (score >= 0.5) and to NOT entail
# "The CEO resigned in March." (score < 0.5) -- reusing that already-proven
# pair keeps these tests deterministic without re-discovering NLI scores.
NLI_EVIDENCE_CHUNK = {
    "id": 20,
    "kind": "text",
    "text": "The company reported revenue of 4 million dollars in 2023.",
    "page": 4,
    "bbox": [0, 0, 10, 10],
    "table_df_json": None,
}


def test_verified_grounded_answer_has_supported_claims_and_citations(monkeypatch):
    session_id = create_session(pages=[], chunks=[NLI_EVIDENCE_CHUNK, HR_CHUNK])

    mock_generate = MagicMock(return_value="Revenue was 4 million dollars.")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    resp = answer_question(_req(session_id, "What was the revenue?"))

    assert resp.refused is False
    assert resp.answer == "Revenue was 4 million dollars."
    assert len(resp.claims) == 1
    assert resp.claims[0].supported is True
    assert resp.claims[0].score >= 0.5
    assert len(resp.citations) >= 1
    assert any(c.page == NLI_EVIDENCE_CHUNK["page"] for c in resp.citations)


def test_verified_hallucinated_answer_is_refused(monkeypatch):
    # Same grounded session, but the model invents a fact the evidence never
    # states -- this is the faithfulness firewall catching a hallucination.
    session_id = create_session(pages=[], chunks=[NLI_EVIDENCE_CHUNK, HR_CHUNK])

    mock_generate = MagicMock(return_value="The CEO resigned in March.")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    resp = answer_question(_req(session_id, "What was the revenue?"))

    assert resp.refused is True
    # the raw draft is kept (not blanked) so the frontend can red-flag it
    assert resp.answer == "The CEO resigned in March."
    assert len(resp.claims) == 1
    assert resp.claims[0].supported is False
    assert resp.claims[0].score < 0.5
    assert resp.citations == []


def test_verified_false_skips_nli_even_for_hallucinated_answer(monkeypatch):
    session_id = create_session(pages=[], chunks=[NLI_EVIDENCE_CHUNK, HR_CHUNK])

    mock_generate = MagicMock(return_value="The CEO resigned in March.")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    resp = answer_question(_req(session_id, "What was the revenue?", verified=False))

    assert resp.refused is False
    assert resp.answer == "The CEO resigned in March."
    assert resp.claims == []


def test_cross_modal_refuses_when_gate_score_too_low(monkeypatch):
    # A red image against an unrelated query should score below threshold
    # under the CLIP gate, and refuse before any generate() call.
    pages = [{"index": 0, "image_png": make_solid_image("red"), "width": 224, "height": 224}]
    chunks = [_figure_chunk(0, 0, [0, 0, 224, 224])]
    session_id = create_session(pages=pages, chunks=chunks)

    mock_generate = MagicMock(return_value="should not be called")
    monkeypatch.setattr("app.generate.answer.generate", mock_generate)

    resp = answer_question(
        _req(session_id, "quantum entanglement in superconducting circuits", retrieval_mode="cross_modal")
    )

    assert resp.refused is True
    assert resp.answer == ""
    mock_generate.assert_not_called()
