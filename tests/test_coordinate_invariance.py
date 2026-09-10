import numpy as np

from relational_dynamics.geometry import LorentzianEmbedding, lorentz_boost


def test_proper_lorentz_boost_preserves_intervals_and_causal_relations():
    coordinates = np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    embedding = LorentzianEmbedding(coordinates)
    transformed = embedding.transform(lorentz_boost(0.3))
    np.testing.assert_allclose(transformed.separation_squared(0, 1), embedding.separation_squared(0, 1), atol=1e-10)
    np.testing.assert_allclose(transformed.separation_squared(0, 2), embedding.separation_squared(0, 2), atol=1e-10)
    assert transformed.non_degenerate
