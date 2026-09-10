"""Information quantities and explicitly exploratory geometry mappings."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .quantum import QuantumSystem

MODELING_ASSUMPTION = "MODELING_ASSUMPTION"
MATHEMATICAL_DEFINITION = "MATHEMATICAL_DEFINITION"


@dataclass
class InformationCausalityEngine:
    system: QuantumSystem
    epsilon: float = 1e-12
    mapping_label: str = MODELING_ASSUMPTION

    def entropy(self, subsystem: int) -> float:
        return self.system.entropy((subsystem,))

    def mutual_information(self, first: int, second: int) -> float:
        return self.system.mutual_information((first,), (second,))

    def mutual_information_matrix(self) -> np.ndarray:
        count = len(self.system.subsystem_dims)
        matrix = np.zeros((count, count), dtype=float)
        for first in range(count):
            for second in range(first + 1, count):
                matrix[first, second] = matrix[second, first] = self.mutual_information(first, second)
        return matrix

    def multivariate_mutual_information(self, first: int, second: int, third: int) -> float:
        """Co-information I(A:B:C) = S(A)+S(B)+S(C)-S(AB)-S(AC)-S(BC)+S(ABC).

        This is a signed quantity (interaction information); it is not itself a metric
        and pairwise mutual information is not assumed sufficient on its own.
        """
        s_a, s_b, s_c = self.system.entropy((first,)), self.system.entropy((second,)), self.system.entropy((third,))
        s_ab = self.system.entropy((first, second))
        s_ac = self.system.entropy((first, third))
        s_bc = self.system.entropy((second, third))
        s_abc = self.system.entropy((first, second, third))
        return float(s_a + s_b + s_c - s_ab - s_ac - s_bc + s_abc)

    def multipartite_information_triples(self) -> dict[tuple[int, int, int], float]:
        count = len(self.system.subsystem_dims)
        triples: dict[tuple[int, int, int], float] = {}
        for first in range(count):
            for second in range(first + 1, count):
                for third in range(second + 1, count):
                    triples[(first, second, third)] = self.multivariate_mutual_information(first, second, third)
        return triples

    def information_distance(self, first: int, second: int, i_max: float) -> float:
        """PDF eqs. (21)-(22), retained as a configurable modeling ansatz."""
        mutual_information = self.mutual_information(first, second)
        weight = (mutual_information + self.epsilon) / (i_max + self.epsilon)
        return float(-np.log(max(weight, self.epsilon)))

    def distance_matrix(self, i_max: float) -> np.ndarray:
        count = len(self.system.subsystem_dims)
        matrix = np.zeros((count, count), dtype=float)
        for first in range(count):
            for second in range(first + 1, count):
                matrix[first, second] = matrix[second, first] = self.information_distance(first, second, i_max)
        return matrix

    def metric_axiom_report(self, i_max: float, tolerance: float = 1e-10) -> dict[str, object]:
        distances = self.distance_matrix(i_max)
        nonnegative = bool(np.all(distances >= -tolerance))
        symmetric = bool(np.allclose(distances, distances.T, atol=tolerance))
        identity = bool(np.all(np.diag(distances) <= tolerance) and np.all(distances[np.triu_indices_from(distances, 1)] > tolerance))
        triangle = all(
            distances[first, third] <= distances[first, second] + distances[second, third] + tolerance
            for first in range(len(distances))
            for second in range(len(distances))
            for third in range(len(distances))
        )
        is_metric = nonnegative and symmetric and identity and triangle
        return {
            "classification": "metric" if is_metric else "information-derived dissimilarity",
            "non_negativity": nonnegative,
            "symmetry": symmetric,
            "identity_of_indiscernibles": identity,
            "triangle_inequality": triangle,
            "parameter_count": 1,
            "mapping_is_physical_law": False,
        }

    @staticmethod
    def parametric_mapping(mutual_information: float, causal_influence: float, parameters: np.ndarray, epsilon: float = 1e-12) -> float:
        """Low-dimensional F_theta basis from PDF eq. (24)."""
        values = np.array([1.0, -np.log(mutual_information + epsilon), causal_influence, causal_influence**2, mutual_information * causal_influence])
        if len(parameters) != len(values):
            raise ValueError("the interpretable basis requires five parameters")
        return float(np.dot(parameters, values))
