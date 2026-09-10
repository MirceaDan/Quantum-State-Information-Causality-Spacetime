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

    def causal_relation(self, source: int, target: int, tolerance: float = 1e-10) -> bool:
        """Return whether target is in the future light cone of source."""
        delta_t = self.coordinates[target, 0] - self.coordinates[source, 0]
        return bool(delta_t > tolerance and self.separation_squared(source, target) <= tolerance)

    def causal_matrix(self, tolerance: float = 1e-10) -> np.ndarray:
        count = len(self.coordinates)
        return np.array([[self.causal_relation(source, target, tolerance) for target in range(count)] for source in range(count)])

    def compare_causal_relations(self, truth: np.ndarray, tolerance: float = 1e-10) -> dict[str, float | int]:
        predicted = self.causal_matrix(tolerance)
        truth = np.asarray(truth, dtype=bool)
        if truth.shape != predicted.shape:
            raise ValueError("truth relation matrix shape does not match embedding")
        mask = ~np.eye(len(truth), dtype=bool)
        true_positive = int(np.count_nonzero(predicted & truth & mask))
        false_positive = int(np.count_nonzero(predicted & ~truth & mask))
        false_negative = int(np.count_nonzero(~predicted & truth & mask))
        true_negative = int(np.count_nonzero(~predicted & ~truth & mask))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_links": false_positive,
            "false_negative_links": false_negative,
            "false_positive_rate": false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0,
            "false_negative_rate": false_negative / (false_negative + true_positive) if false_negative + true_positive else 0.0,
        }

    def transform(self, matrix: np.ndarray) -> "LorentzianEmbedding":
        matrix = np.asarray(matrix, dtype=float)
        return LorentzianEmbedding(self.coordinates @ matrix.T, matrix @ self.metric @ matrix.T)


def minkowski_metric(dimension: int = 4) -> np.ndarray:
    metric = np.eye(dimension, dtype=float)
    metric[0, 0] = -1.0
    return metric


def lorentz_boost(beta: float, spatial_axis: int = 1) -> np.ndarray:
    """Return a proper boost in one time/spatial coordinate plane."""
    if abs(beta) >= 1.0:
        raise ValueError("boost velocity must satisfy |beta| < 1")
    matrix = np.eye(4, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    matrix[0, 0] = gamma
    matrix[spatial_axis, spatial_axis] = gamma
    matrix[0, spatial_axis] = -gamma * beta
    matrix[spatial_axis, 0] = -gamma * beta
    return matrix
