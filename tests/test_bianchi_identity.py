import numpy as np

from relational_dynamics.gr import bianchi_residual, conformal_geometry, minkowski_geometry


def test_bianchi_identity_for_flat_and_curved_reference_metrics():
    flat = bianchi_residual(minkowski_geometry(), np.zeros(4))
    curved = bianchi_residual(conformal_geometry(scale=0.2), np.array([0.3, 0.1, 0.0, 0.0]))

    assert np.linalg.norm(flat) < 1e-10
    assert np.linalg.norm(curved) < 1e-4
