# backend/app/index/embedders.py
"""bge-small text embeddings + CLIP image/text embeddings (sentence-transformers), CPU only.

All rows are L2-normalized so dot product == cosine similarity — downstream
FAISS index (Task 2.2) can use a plain inner-product index.

Query vs. passage asymmetry: bge-small-en-v1.5's docs recommend prefixing
*queries* (not passages/documents) with the instruction
"Represent this sentence for searching relevant passages:" — this measurably
improves retrieval because the model was fine-tuned with this asymmetric
setup. ``embed_texts`` embeds passages plain; ``embed_query`` applies the
prefix.

CLIP space is a SEPARATE, SECOND index from the bge text space above: it has
a different dimensionality and its vectors are NOT comparable to bge vectors
(different model, different training objective). What makes CLIP useful for
cross-modal retrieval is that ``embed_images`` and ``embed_query_clip`` share
*one* space with each other — a CLIP text vector can be compared directly to
a CLIP image vector. bge text vectors can only be compared to other bge text
vectors.
"""
import io

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.config import settings

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None  # ponytail: module-level lazy singleton, loaded once on first use
_clip_model = None  # ponytail: separate lazy singleton for the CLIP model/space


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.text_model, device="cpu")
    return _model


def _get_clip_model():
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer(settings.clip_model, device="cpu")
    return _clip_model


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


def embed_images(images: list[bytes]) -> np.ndarray:
    """Embed PNG-encoded images with CLIP. Returns (n, dim) float32, unit-norm rows.

    Lives in the CLIP space (see module docstring) — NOT comparable to
    embed_texts/embed_query's bge vectors.
    """
    model = _get_clip_model()
    pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
    return model.encode(
        pil_images, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)


def embed_query_clip(text: str) -> np.ndarray:
    """Embed a text query with the SAME CLIP model used by embed_images, so the
    result lands in the shared CLIP space and can be compared directly to image
    vectors (true cross-modal text -> image retrieval).

    Returns a 1D float32 unit-norm vector (same dim as embed_images). Do NOT
    use embed_query (bge) for this — bge and CLIP text vectors are not
    interchangeable even though both embed text.
    """
    model = _get_clip_model()
    vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)
