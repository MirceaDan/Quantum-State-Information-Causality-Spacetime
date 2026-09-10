import numpy as np

from relational_dynamics.reconstruction import (
    depolarizing_channel,
    reconstruct_state,
    unitary_channel,
)
from relational_dynamics.quantum import density_matrix


def test_reversible_channel_reconstructs_state_without_temporal_claim():
    state = density_matrix(np.array([1, 0], dtype=complex))
    flip = np.array([[0, 1], [1, 0]], dtype=complex)
    report = reconstruct_state(state, unitary_channel(flip), unitary_channel(flip))

    assert report.forward_fidelity > 1 - 1e-12
    assert report.reverse_fidelity > 1 - 1e-12
    assert report.round_trip_loss < 1e-12
    assert report.label == "STATE_RECONSTRUCTION"
    assert report.temporal_reversal_claim is False


def test_irreversible_channel_has_imperfect_reverse_reconstruction():
    state = density_matrix(np.array([1, 0], dtype=complex))
    report = reconstruct_state(state, depolarizing_channel(0.5), depolarizing_channel(0.5))

    assert report.forward_fidelity > 1 - 1e-12
    assert report.reverse_fidelity < 1 - 1e-3
    assert report.round_trip_loss > 1e-3
