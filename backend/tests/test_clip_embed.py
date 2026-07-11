# backend/tests/test_clip_embed.py
import numpy as np

from app.index.embedders import embed_images, embed_query_clip
from tests.fixtures import make_solid_image


def test_embed_images_returns_unit_norm_vectors():
    png = make_solid_image("red")
    vecs = embed_images([png])

    assert vecs.dtype == np.float32
    assert vecs.shape[0] == 1
    assert vecs.shape[1] > 0

    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_query_clip_shares_dim_with_embed_images():
    dim = embed_images([make_solid_image("red")]).shape[1]

    q = embed_query_clip("a red image")

    assert q.dtype == np.float32
    assert q.shape == (dim,)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-5


def test_clip_cross_modal_directional_sanity():
    """A solid red image should score higher against "a red image" than
    against "a blue image", and vice versa for a solid blue image — proving
    embed_images and embed_query_clip land in one real shared space, not just
    matching-dimension noise."""
    red_vec = embed_images([make_solid_image("red")])[0]
    blue_vec = embed_images([make_solid_image("blue")])[0]

    red_query = embed_query_clip("a red image")
    blue_query = embed_query_clip("a blue image")

    cos_red_to_red = float(np.dot(red_vec, red_query))
    cos_red_to_blue = float(np.dot(red_vec, blue_query))
    assert cos_red_to_red > cos_red_to_blue

    cos_blue_to_blue = float(np.dot(blue_vec, blue_query))
    cos_blue_to_red = float(np.dot(blue_vec, red_query))
    assert cos_blue_to_blue > cos_blue_to_red
