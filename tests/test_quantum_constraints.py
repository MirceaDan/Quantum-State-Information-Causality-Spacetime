import numpy as np

from relational_dynamics.quantum import QuantumSystem, density_matrix, partial_trace


def test_bell_state_constraints_and_partial_trace():
    state = np.zeros(4, dtype=complex)
    state[[0, 3]] = 1 / np.sqrt(2)
    rho = density_matrix(state)
    system = QuantumSystem.from_density_matrix(rho, subsystem_dims=(2, 2))

    diagnostics = system.validate_state()
    assert diagnostics.trace_error < 1e-12
    assert diagnostics.hermiticity_error < 1e-12
    assert diagnostics.minimum_eigenvalue > -1e-12
    reduced = partial_trace(rho, keep=(0,), dims=(2, 2))
    np.testing.assert_allclose(reduced, np.eye(2) / 2, atol=1e-12)


def test_entropy_mutual_information_fidelity_and_relative_entropy():
    state = np.zeros(4, dtype=complex)
    state[[0, 3]] = 1 / np.sqrt(2)
    system = QuantumSystem.from_density_matrix(density_matrix(state), (2, 2))

    assert np.isclose(system.entropy((0,)), np.log(2), atol=1e-12)
    assert np.isclose(system.mutual_information((0,), (1,)), 2 * np.log(2), atol=1e-12)
    assert np.isclose(system.fidelity(system.rho, system.rho), 1.0, atol=1e-12)
    assert np.isclose(system.relative_entropy(system.rho, system.rho), 0.0, atol=1e-12)
