import numpy as np

from relational_dynamics.gr import (
    bianchi_residual,
    conformal_geometry,
    covariant_divergence,
    spatial_conformal_geometry,
)


def test_bianchi_energy_momentum_and_delta_are_separate_residuals():
    geometry = conformal_geometry(scale=0.2)
    point = np.array([0.3, 0.1, 0.0, 0.0])
    metric_tensor = lambda x: geometry.metric(x)
    nonconserved_tensor = lambda x: x[0] * geometry.metric(x)

    bianchi = bianchi_residual(geometry, point)
    energy_momentum = covariant_divergence(geometry, metric_tensor, point)
    nonconserved = covariant_divergence(geometry, nonconserved_tensor, point)
    delta_zero = np.zeros(4)

    assert np.linalg.norm(bianchi) < 1e-4
    assert np.linalg.norm(energy_momentum) < 1e-5
    assert np.linalg.norm(nonconserved) > 1e-3
    assert np.linalg.norm(delta_zero) == 0.0


def test_second_analytic_curved_geometry_has_nonzero_curvature():
    geometry = spatial_conformal_geometry(scale=0.15)
    tensors = geometry.curvature_tensors(np.array([0.2, 0.3, 0.0, 0.0]))
    assert np.linalg.norm(tensors.riemann) > 1e-8
    assert np.linalg.norm(tensors.einstein) > 1e-8
