import numpy as np

from relational_dynamics.causality import PauliInterventionFamily, sampled_causal_influence
from relational_dynamics.quantum import QuantumSystem, density_matrix


def test_sampled_causal_influence_reports_approximation_scope():
    state = np.zeros(4, dtype=complex)
    state[[0, 3]] = 1 / np.sqrt(2)
    system = QuantumSystem.from_density_matrix(density_matrix(state), (2, 2))
    report = sampled_causal_influence(system, PauliInterventionFamily())

    assert report.matrix.shape == (2, 2)
    assert np.all(report.matrix >= -1e-12)
    assert report.intervention_family == "Pauli operations"
    assert report.number_of_interventions == 4
    assert report.approximation_scope.startswith("finite sampled")
    assert report.is_exact_supremum is False
