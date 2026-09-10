import numpy as np

from relational_dynamics.information import InformationCausalityEngine
from relational_dynamics.quantum import QuantumSystem, density_matrix


def test_information_distance_reports_axioms_instead_of_assuming_metric():
    state = np.zeros(8, dtype=complex)
    state[[0, 7]] = 1 / np.sqrt(2)
    engine = InformationCausalityEngine(QuantumSystem.from_density_matrix(density_matrix(state), (2, 2, 2)))
    report = engine.metric_axiom_report(i_max=2 * np.log(2))

    assert report["classification"] in {"metric", "information-derived dissimilarity"}
    assert report["parameter_count"] == 1
    assert isinstance(report["triangle_inequality"], bool)
    assert report["mapping_is_physical_law"] is False
