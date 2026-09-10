import numpy as np

from relational_dynamics.gr import conformal_geometry, minkowski_geometry


def test_minkowski_tensor_calculus_is_flat():
    geometry = minkowski_geometry()
    tensors = geometry.curvature_tensors(np.zeros(4))

    np.testing.assert_allclose(tensors.christoffel, 0.0, atol=1e-12)
    np.testing.assert_allclose(tensors.riemann, 0.0, atol=1e-12)
    np.testing.assert_allclose(tensors.ricci, 0.0, atol=1e-12)
    np.testing.assert_allclose(tensors.einstein, 0.0, atol=1e-12)


def test_conformal_metric_is_curved():
    geometry = conformal_geometry(scale=0.2)
    tensors = geometry.curvature_tensors(np.array([0.3, 0.1, 0.0, 0.0]))

    assert np.linalg.norm(tensors.christoffel) > 1e-8
    assert np.linalg.norm(tensors.riemann) > 1e-8
    assert np.linalg.norm(tensors.einstein) > 1e-8
    assert tensors.inverse_metric.shape == (4, 4)
