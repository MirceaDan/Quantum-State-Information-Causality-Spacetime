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

    def information_distance(self, first: int, second: int, i_max: float) -> float:
        """PDF eqs. (21)-(22), retained as a configurable modeling ansatz."""
        mutual_information = self.mutual_information(first, second)
        weight = (mutual_information + self.epsilon) / (i_max + self.epsilon)
        return float(-np.log(max(weight, self.epsilon)))

    @staticmethod
    def parametric_mapping(mutual_information: float, causal_influence: float, parameters: np.ndarray, epsilon: float = 1e-12) -> float:
        """Low-dimensional F_theta basis from PDF eq. (24)."""
        values = np.array([1.0, -np.log(mutual_information + epsilon), causal_influence, causal_influence**2, mutual_information * causal_influence])
        if len(parameters) != len(values):
            raise ValueError("the interpretable basis requires five parameters")
        return float(np.dot(parameters, values))
