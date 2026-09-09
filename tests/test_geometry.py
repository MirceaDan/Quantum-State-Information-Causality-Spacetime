import numpy as np

from relational_dynamics.geometry import LorentzianEmbedding


def test_minkowski_embedding_has_lorentzian_signature_and_invariants():
    coordinates = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    embedding = LorentzianEmbedding(coordinates)
    assert embedding.signature == (-1, 3)
    assert embedding.separation_squared(0, 1) == -1
    assert embedding.separation_squared(0, 2) == 1
    assert embedding.non_degenerate


def test_coordinate_transform_changes_components_but_not_invariant():
    embedding = LorentzianEmbedding(np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=float))
    transformed = embedding.transform(np.diag([1, -1, 1, 1]))
    assert transformed.separation_squared(0, 1) == embedding.separation_squared(0, 1)
