import numpy as np

from relational_dynamics.causality import (
    DynamicalCausalProcess,
    identity_process,
    cnot_process,
    sampled_dynamical_influence,
)
from relational_dynamics.quantum import density_matrix


def test_independent_process_has_no_influence():
    state = density_matrix(np.array([1, 0, 0, 0], dtype=complex))
    report = sampled_dynamical_influence(state, (2, 2), identity_process(), source=0, target=1)
    assert report.influence < 1e-12


def test_cnot_process_has_direct_one_way_influence():
    state = density_matrix(np.array([1, 0, 0, 0], dtype=complex))
    forward = sampled_dynamical_influence(state, (2, 2), cnot_process(), source=0, target=1)
    reverse = sampled_dynamical_influence(state, (2, 2), cnot_process(), source=1, target=0)
    assert forward.influence > 0.99
    assert reverse.influence < 1e-12


def test_common_cause_has_correlation_without_direct_process_influence():
    state = np.zeros(8, dtype=complex)
    state[[0, 7]] = 1 / np.sqrt(2)
    report = sampled_dynamical_influence(density_matrix(state), (2, 2, 2), identity_process(), source=0, target=1)
    assert report.mutual_information > 0
    assert report.influence < 1e-12
    assert report.measure == "trace_distance"
