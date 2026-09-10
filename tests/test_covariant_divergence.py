import numpy as np

from relational_dynamics.gr import covariant_divergence, conformal_geometry


def test_metric_tensor_is_covariantly_conserved():
    geometry = conformal_geometry(scale=0.2)
    point = np.array([0.3, 0.1, 0.0, 0.0])
    residual = covariant_divergence(geometry, lambda x: geometry.metric(x), point)
    assert np.linalg.norm(residual) < 1e-5


def test_nonconserved_scalar_metric_tensor_is_detected():
    geometry = conformal_geometry(scale=0.2)
    point = np.array([0.3, 0.1, 0.0, 0.0])
    residual = covariant_divergence(geometry, lambda x: x[0] * geometry.metric(x), point)
    assert np.linalg.norm(residual) > 1e-3
