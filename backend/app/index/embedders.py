# backend/app/index/embedders.py
"""bge-small text embeddings (sentence-transformers), CPU only.

All rows are L2-normalized so dot product == cosine similarity — downstream
FAISS index (Task 2.2) can use a plain inner-product index.

Query vs. passage asymmetry: bge-small-en-v1.5's docs recommend prefixing
*queries* (not passages/documents) with the instruction
"Represent this sentence for searching relevant passages:" — this measurably
improves retrieval because the model was fine-tuned with this asymmetric
setup. ``embed_texts`` embeds passages plain; ``embed_query`` applies the
prefix.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None  # ponytail: module-level lazy singleton, loaded once on first use


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.text_model, device="cpu")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed passages/documents. Returns (n, dim) float32, unit-norm rows."""
    model = _get_model()
    return model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a search query with bge's recommended query instruction prefix.

    Returns a 1D float32 unit-norm vector (same dim as embed_texts).
    """
    model = _get_model()
    vec = model.encode(
        _QUERY_INSTRUCTION + text, normalize_embeddings=True, convert_to_numpy=True
    )
    return vec.astype(np.float32)
