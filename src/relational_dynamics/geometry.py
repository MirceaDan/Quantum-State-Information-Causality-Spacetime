"""Discrete Lorentzian embedding and coordinate-invariance checks."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

MATHEMATICAL_DEFINITION = "MATHEMATICAL_DEFINITION"
NUMERICAL_APPROXIMATION = "NUMERICAL_APPROXIMATION"


@dataclass(frozen=True)
class LorentzianEmbedding:
    coordinates: np.ndarray
    metric: np.ndarray | None = None
    component_label: str = MATHEMATICAL_DEFINITION

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        metric = minkowski_metric(coordinates.shape[1]) if self.metric is None else np.asarray(self.metric, dtype=float)
        if coordinates.ndim != 2 or metric.shape != (coordinates.shape[1], coordinates.shape[1]):
            raise ValueError("coordinates and metric dimensions do not match")
        if not np.allclose(metric, metric.T):
            raise ValueError("candidate metric must be symmetric")
        eigenvalues = np.linalg.eigvalsh(metric)
        if np.count_nonzero(eigenvalues < -1e-10) != 1 or np.count_nonzero(eigenvalues > 1e-10) != len(eigenvalues) - 1:
            raise ValueError("metric must have Lorentzian signature (-,+,...,+)")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "metric", metric)

    @property
    def signature(self) -> tuple[int, int]:
        eigenvalues = np.linalg.eigvalsh(self.metric)
        return (-1, int(np.count_nonzero(eigenvalues > 0)))

    @property
    def non_degenerate(self) -> bool:
        return bool(abs(np.linalg.det(self.metric)) > 1e-12)

    def separation_squared(self, first: int, second: int) -> float:
        difference = self.coordinates[first] - self.coordinates[second]
        return float(difference @ self.metric @ difference)

    def transform(self, matrix: np.ndarray) -> "LorentzianEmbedding":
        matrix = np.asarray(matrix, dtype=float)
        return LorentzianEmbedding(self.coordinates @ matrix.T, matrix @ self.metric @ matrix.T)


def minkowski_metric(dimension: int = 4) -> np.ndarray:
    metric = np.eye(dimension, dtype=float)
    metric[0, 0] = -1.0
    return metric
