# backend/app/verify/nli.py
"""Claim splitting + NLI faithfulness gate (Task 4.1).

``verify_claims`` splits an answer into sentence-level claims and scores each
against the retrieved evidence with a ``sentence_transformers.CrossEncoder``
NLI model (premise=evidence chunk text, hypothesis=claim). The max
entailment probability across all retrieved chunks becomes the claim's
score/citation.

Verified label order for ``cross-encoder/nli-deberta-v3-base`` (checked at
runtime via ``model.config.id2label``, not hardcoded blindly):
    {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
So softmax(logits)[1] is P(entailment). We assert this order at model load
so a silent model swap can't flip the labels under us.
"""
import re

import numpy as np
from sentence_transformers import CrossEncoder

from app.config import settings
from app.schemas import Citation, Claim

_SNIPPET_LEN = 150
_ENTAILMENT_LABEL = "entailment"

_model = None  # ponytail: module-level lazy singleton, loaded once on first use
_entailment_idx = None


def _get_model():
    global _model, _entailment_idx
    if _model is None:
        model = CrossEncoder(settings.nli_model, device="cpu")
        id2label = {i: l.lower() for i, l in model.config.id2label.items()}
        idx = [i for i, l in id2label.items() if l == _ENTAILMENT_LABEL]
        if len(idx) != 1:
            raise RuntimeError(
                f"could not locate a unique '{_ENTAILMENT_LABEL}' label in "
                f"{settings.nli_model} id2label={id2label}"
            )
        _entailment_idx = idx[0]
        _model = model
    return _model


# ponytail: stdlib regex sentence splitter -- splits on .!? followed by
# whitespace/EOL. No clause-level splitting, no handling of abbreviations
# ("Mr.", "e.g.") or decimal numbers ("4.2 million") that happen to contain
# a period followed by a space -- good enough for claim-level faithfulness
# checking, upgrade to a real sentence tokenizer only if this misfires often.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_claims(answer: str) -> list[str]:
    if not answer or not answer.strip():
        return []
    parts = _SENTENCE_SPLIT_RE.split(answer.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(chunk: dict) -> str:
    return chunk.get("text") or chunk.get("caption_text") or ""


def verify_claims(answer: str, retrieved: list[dict]) -> list[Claim]:
    evidence = [
        (r["chunk"], text) for r in retrieved if (text := _chunk_text(r["chunk"]))
    ]

    claims_text = split_claims(answer)
    if not claims_text:
        return []

    if not evidence:
        return [
            Claim(text=text, supported=False, score=0.0, citations=[])
            for text in claims_text
        ]

    model = _get_model()
    pairs = [(chunk_text, claim) for chunk, chunk_text in evidence for claim in claims_text]
    logits = model.predict(pairs, apply_softmax=False)
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    entailment_probs = probs[:, _entailment_idx].reshape(len(evidence), len(claims_text))

    claims = []
    for j, claim_text in enumerate(claims_text):
        best_i = int(entailment_probs[:, j].argmax())
        score = float(entailment_probs[best_i, j])
        supported = score >= settings.faithfulness_threshold
        best_chunk, best_text = evidence[best_i]
        citations = [
            Citation(
                page=best_chunk["page"],
                bbox=best_chunk["bbox"],
                snippet=best_text[:_SNIPPET_LEN],
            )
        ]
        claims.append(
            Claim(text=claim_text, supported=supported, score=score, citations=citations)
        )
    return claims
