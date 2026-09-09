"""Reference GR baseline and independently reported conservation residuals."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

ESTABLISHED_PHYSICS = "ESTABLISHED_PHYSICS"
SPECULATIVE_EXTENSION = "SPECULATIVE_EXTENSION"
NUMERICAL_APPROXIMATION = "NUMERICAL_APPROXIMATION"


def minkowski_metric(dimension: int = 4) -> np.ndarray:
    metric = np.eye(dimension, dtype=float)
    metric[0, 0] = -1.0
    return metric


@dataclass(frozen=True)
class EinsteinResidual:
    tensor: np.ndarray
    stress_energy: np.ndarray
    cosmological_constant: float = 0.0
    coupling: float = 1.0
    discretization: str = "analytic flat metric reference"
    component_label: str = ESTABLISHED_PHYSICS

    @classmethod
    def from_metric(cls, metric: np.ndarray, stress_energy: np.ndarray, cosmological_constant: float = 0.0, coupling: float = 1.0) -> "EinsteinResidual":
        metric = np.asarray(metric, dtype=float)
        stress_energy = np.asarray(stress_energy, dtype=float)
        tensor = cosmological_constant * metric - coupling * stress_energy
        return cls(tensor, stress_energy, cosmological_constant, coupling)

    @property
    def einstein_norm(self) -> float:
        return float(np.linalg.norm(self.tensor))


@dataclass(frozen=True)
class ConservationMonitor:
    delta: np.ndarray
    spacing: float = 1.0
    component_label: str = SPECULATIVE_EXTENSION
    derivative_scheme: str = "centered finite differences; zero-padded boundaries"

    def delta_norm(self) -> float:
        return float(np.linalg.norm(self.delta))

    def residual_norm(self) -> float:
        array = np.asarray(self.delta, dtype=float)
        if np.allclose(array, 0.0):
            return 0.0
        derivatives = np.gradient(array, self.spacing, axis=0, edge_order=1)
        return float(np.linalg.norm(derivatives))
