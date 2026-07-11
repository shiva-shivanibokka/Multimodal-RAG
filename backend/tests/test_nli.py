# backend/tests/test_nli.py
"""Task 4.1: claim splitting + NLI faithfulness gate.

Downloads cross-encoder/nli-deberta-v3-base (~700MB) on first run. Set
HF_HOME/TORCH_HOME to a writable cache dir before running, e.g.:
    HF_HOME=C:/mrag/.cache TORCH_HOME=C:/mrag/.cache \
        /c/mrag/.venv/Scripts/python.exe -m pytest tests/test_nli.py -v
"""
from app.verify.nli import split_claims, verify_claims

EVIDENCE_CHUNK = {
    "id": 0,
    "kind": "text",
    "text": "The company reported revenue of 4 million dollars in 2023.",
    "page": 2,
    "bbox": [0, 0, 10, 10],
    "table_df_json": None,
}
RETRIEVED = [{"chunk": EVIDENCE_CHUNK, "score": 1.0}]


def test_split_claims_splits_on_sentence_boundaries():
    claims = split_claims("The revenue was $4M. Costs fell 10%.")
    assert claims == ["The revenue was $4M.", "Costs fell 10%."]


def test_split_claims_empty_string():
    assert split_claims("") == []
    assert split_claims("   ") == []


def test_supported_claim_is_entailed_by_evidence():
    claims = verify_claims("Revenue was 4 million dollars.", RETRIEVED)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.supported is True
    assert claim.score >= 0.5
    assert len(claim.citations) == 1
    assert claim.citations[0].page == EVIDENCE_CHUNK["page"]
    assert claim.citations[0].bbox == EVIDENCE_CHUNK["bbox"]


def test_unsupported_claim_is_not_entailed_by_evidence():
    claims = verify_claims("The CEO resigned in March.", RETRIEVED)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.supported is False
    assert claim.score < 0.5


def test_mixed_answer_splits_supported_and_unsupported_claims():
    answer = "Revenue was 4 million dollars. The CEO resigned in March."
    claims = verify_claims(answer, RETRIEVED)

    assert len(claims) == 2
    supported_flags = [c.supported for c in claims]
    assert supported_flags == [True, False]


def test_no_evidence_text_means_unsupported_and_no_citations():
    figure_chunk = {
        "id": 1,
        "kind": "figure",
        "text": "",
        "page": 5,
        "bbox": [0, 0, 10, 10],
        "table_df_json": None,
    }
    claims = verify_claims("Revenue was 4 million dollars.", [{"chunk": figure_chunk, "score": 1.0}])

    assert len(claims) == 1
    assert claims[0].supported is False
    assert claims[0].score == 0.0
    assert claims[0].citations == []
