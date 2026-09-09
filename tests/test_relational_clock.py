import numpy as np

from relational_dynamics.quantum import QuantumSystem, density_matrix


def test_relational_clock_conditioning_is_not_external_time():
    state = np.zeros(4, dtype=complex)
    state[[0, 3]] = 1 / np.sqrt(2)
    system = QuantumSystem.from_density_matrix(density_matrix(state), (2, 2))

    conditioned = system.conditional_matter_state(clock_outcome=1)
    assert np.isclose(np.trace(conditioned.state), 1.0)
    np.testing.assert_allclose(conditioned.state, np.array([[0, 0], [0, 1]], dtype=complex))
    assert conditioned.clock_reading == 1
    assert conditioned.external_time is None
    assert conditioned.computational_index is None


def test_hamiltonian_constraint_residual_for_kernel_state():
    constraint = np.diag([0.0, 1.0, 2.0, 3.0])
    state = np.array([1, 0, 0, 0], dtype=complex)
    system = QuantumSystem.from_density_matrix(density_matrix(state), (2, 2))
    assert system.constraint_residual(constraint) < 1e-12
