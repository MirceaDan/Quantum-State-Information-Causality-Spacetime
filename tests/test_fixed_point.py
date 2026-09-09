import numpy as np

from relational_dynamics.fixed_point import FixedPointExperiment
from relational_dynamics.reconstruction import depolarizing_channel, unitary_channel
from relational_dynamics.quantum import density_matrix


def test_contracting_process_has_stable_fixed_point():
    initial = density_matrix(np.array([1, 0], dtype=complex))
    report = FixedPointExperiment(depolarizing_channel(0.5), tolerance=1e-10).run(initial)

    assert report.classification == "FIXED_POINT"
    assert report.fixed_point_error < 1e-8
    assert report.stability == "convergent"
    assert report.causal_loop_candidate is False


def test_periodic_process_is_not_called_a_fixed_point_or_causal_loop():
    initial = density_matrix(np.array([1, 0], dtype=complex))
    flip = np.array([[0, 1], [1, 0]], dtype=complex)
    report = FixedPointExperiment(unitary_channel(flip), tolerance=1e-10).run(initial)

    assert report.classification == "PERIODIC_PROCESS"
    assert report.causal_loop_candidate is False
    assert report.detected_period == 2
