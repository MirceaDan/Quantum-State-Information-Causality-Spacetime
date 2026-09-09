import numpy as np

from relational_dynamics.gr import ConservationMonitor


def test_zero_delta_is_conserved():
    monitor = ConservationMonitor(delta=np.zeros((2, 4, 4)), spacing=1.0)
    assert monitor.residual_norm() < 1e-12
    assert monitor.delta_norm() == 0.0
