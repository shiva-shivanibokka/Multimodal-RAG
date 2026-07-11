# backend/tests/test_embedders.py
import numpy as np

from app.index.embedders import embed_query, embed_texts


def test_embed_texts_returns_normalized_semantically_meaningful_vectors():
    texts = [
        "a cat sits on the mat",
        "a kitten rests on the rug",
        "quarterly financial report",
    ]
    vecs = embed_texts(texts)

    assert vecs.dtype == np.float32
    assert vecs.shape[0] == 3
    dim = vecs.shape[1]

    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    cos_similar = float(np.dot(vecs[0], vecs[1]))
    cos_unrelated = float(np.dot(vecs[0], vecs[2]))
    assert cos_similar > cos_unrelated


def test_embed_query_returns_unit_norm_vector_matching_embed_texts_dim():
    doc_vecs = embed_texts(["a cat sits on the mat"])
    dim = doc_vecs.shape[1]

    q = embed_query("where is the cat?")

    assert q.dtype == np.float32
    assert q.shape == (dim,)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-5
