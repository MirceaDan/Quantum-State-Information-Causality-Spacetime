import numpy as np

from relational_dynamics.quantum import QuantumSystem, density_matrix


def test_basis_and_product_states_have_zero_entropy_and_zero_mutual_information():
    zero = QuantumSystem.from_density_matrix(density_matrix(np.array([1, 0], dtype=complex)), (2,))
    one = QuantumSystem.from_density_matrix(density_matrix(np.array([0, 1], dtype=complex)), (2,))
    product = QuantumSystem.from_density_matrix(density_matrix(np.array([0, 0, 0, 1], dtype=complex)), (2, 2))

    assert zero.entropy((0,)) == 0.0
    assert one.entropy((0,)) == 0.0
    assert product.mutual_information((0,), (1,)) == 0.0


def test_ghz_three_party_reductions_have_known_entropy():
    state = np.zeros(8, dtype=complex)
    state[[0, 7]] = 1 / np.sqrt(2)
    system = QuantumSystem.from_density_matrix(density_matrix(state), (2, 2, 2))

    assert np.isclose(system.entropy((0,)), np.log(2), atol=1e-12)
    assert np.isclose(system.mutual_information((0,), (1,)), np.log(2), atol=1e-12)


def test_mixed_state_entropy_and_independent_fidelity_relative_entropy_values():
    maximally_mixed = np.eye(2, dtype=complex) / 2
    pure_zero = density_matrix(np.array([1, 0], dtype=complex))
    system = QuantumSystem.from_density_matrix(maximally_mixed, (2,))

    assert np.isclose(system.entropy((0,)), np.log(2), atol=1e-12)
    assert np.isclose(QuantumSystem.fidelity(pure_zero, maximally_mixed), 0.5, atol=1e-10)
    assert np.isclose(QuantumSystem.relative_entropy(pure_zero, maximally_mixed), np.log(2), atol=1e-10)
