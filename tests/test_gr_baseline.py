import numpy as np

from relational_dynamics.gr import EinsteinResidual, minkowski_metric


def test_flat_minkowski_baseline_has_zero_residual():
    metric = minkowski_metric(4)
    residual = EinsteinResidual.from_metric(metric, stress_energy=np.zeros((4, 4)))
    assert residual.einstein_norm < 1e-12
    assert residual.discretization == "analytic flat metric reference"
