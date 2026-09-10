import numpy as np

from relational_dynamics.information import InformationCausalityEngine
from relational_dynamics.quantum import QuantumSystem, density_matrix


def make_system():
    state = np.zeros(8, dtype=complex)
    state[[0, 7]] = 1 / np.sqrt(2)
    return QuantumSystem.from_density_matrix(density_matrix(state), (2, 2, 2))


def test_information_engine_outputs_symmetric_matrix():
    engine = InformationCausalityEngine(make_system())
    matrix = engine.mutual_information_matrix()
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix), 0, atol=1e-12)
    assert matrix[0, 1] > 0


def test_information_distance_is_explicit_ansatz():
    engine = InformationCausalityEngine(make_system(), epsilon=1e-9)
    distance = engine.information_distance(0, 1, i_max=2 * np.log(2))
    assert distance >= 0
    assert engine.mapping_label == "MODELING_ASSUMPTION"
