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


@dataclass(frozen=True)
class CurvatureTensors:
    metric: np.ndarray
    inverse_metric: np.ndarray
    christoffel: np.ndarray
    riemann: np.ndarray
    ricci: np.ndarray
    scalar_curvature: float
    einstein: np.ndarray


@dataclass(frozen=True)
class DifferentialGeometry:
    """Four-dimensional metric field with analytic first and second derivatives."""

    metric_fn: callable
    first_derivative_fn: callable
    second_derivative_fn: callable
    dimension: int = 4
    derivative_description: str = "analytic metric derivatives"

    def metric(self, point: np.ndarray) -> np.ndarray:
        return np.asarray(self.metric_fn(np.asarray(point, dtype=float)), dtype=float)

    def curvature_tensors(self, point: np.ndarray) -> CurvatureTensors:
        metric = self.metric(point)
        inverse = np.linalg.inv(metric)
        first = np.asarray(self.first_derivative_fn(point), dtype=float)
        second = np.asarray(self.second_derivative_fn(point), dtype=float)
        christoffel = np.zeros((self.dimension, self.dimension, self.dimension), dtype=float)
        for rho in range(self.dimension):
            for mu in range(self.dimension):
                for nu in range(self.dimension):
                    christoffel[rho, mu, nu] = 0.5 * sum(
                        inverse[rho, sigma]
                        * (first[mu, sigma, nu] + first[nu, sigma, mu] - first[sigma, mu, nu])
                        for sigma in range(self.dimension)
                    )

        derivative_christoffel = np.zeros((self.dimension,) * 4, dtype=float)
        for direction in range(self.dimension):
            inverse_derivative = -inverse @ first[direction] @ inverse
            for rho in range(self.dimension):
                for mu in range(self.dimension):
                    for nu in range(self.dimension):
                        derivative_christoffel[direction, rho, mu, nu] = 0.5 * sum(
                            inverse_derivative[rho, sigma]
                            * (first[mu, sigma, nu] + first[nu, sigma, mu] - first[sigma, mu, nu])
                            + inverse[rho, sigma]
                            * (second[direction, mu, sigma, nu] + second[direction, nu, sigma, mu] - second[direction, sigma, mu, nu])
                            for sigma in range(self.dimension)
                        )

        riemann = np.zeros((self.dimension,) * 4, dtype=float)
        for rho in range(self.dimension):
            for sigma in range(self.dimension):
                for mu in range(self.dimension):
                    for nu in range(self.dimension):
                        riemann[rho, sigma, mu, nu] = (
                            derivative_christoffel[mu, rho, nu, sigma]
                            - derivative_christoffel[nu, rho, mu, sigma]
                            + sum(christoffel[rho, mu, lam] * christoffel[lam, nu, sigma] - christoffel[rho, nu, lam] * christoffel[lam, mu, sigma] for lam in range(self.dimension))
                        )
        ricci = np.einsum("rsmr->sm", riemann)
        scalar = float(np.einsum("sm,sm", inverse, ricci))
        einstein = ricci - 0.5 * metric * scalar
        return CurvatureTensors(metric, inverse, christoffel, riemann, ricci, scalar, einstein)


def minkowski_geometry() -> DifferentialGeometry:
    metric = minkowski_metric(4)
    zeros_first = np.zeros((4, 4, 4), dtype=float)
    zeros_second = np.zeros((4, 4, 4, 4), dtype=float)
    return DifferentialGeometry(lambda point: metric, lambda point: zeros_first, lambda point: zeros_second)


def conformal_geometry(scale: float = 0.2) -> DifferentialGeometry:
    eta = minkowski_metric(4)

    def metric(point: np.ndarray) -> np.ndarray:
        return np.exp(2 * scale * point[0]) * eta

    def first(point: np.ndarray) -> np.ndarray:
        result = np.zeros((4, 4, 4), dtype=float)
        result[0] = 2 * scale * metric(point)
        return result

    def second(point: np.ndarray) -> np.ndarray:
        result = np.zeros((4, 4, 4, 4), dtype=float)
        result[0, 0] = 4 * scale**2 * metric(point)
        return result

    return DifferentialGeometry(metric, first, second)


def spatial_conformal_geometry(scale: float = 0.15) -> DifferentialGeometry:
    """Analytic conformal metric with a spatially varying conformal factor."""
    eta = minkowski_metric(4)

    def metric(point: np.ndarray) -> np.ndarray:
        return np.exp(2 * scale * point[1]) * eta

    def first(point: np.ndarray) -> np.ndarray:
        result = np.zeros((4, 4, 4), dtype=float)
        result[1] = 2 * scale * metric(point)
        return result

    def second(point: np.ndarray) -> np.ndarray:
        result = np.zeros((4, 4, 4, 4), dtype=float)
        result[1, 1] = 4 * scale**2 * metric(point)
        return result

    return DifferentialGeometry(metric, first, second)


def covariant_divergence(geometry: DifferentialGeometry, tensor_fn: callable, point: np.ndarray, step: float = 1e-5) -> np.ndarray:
    """Compute nabla_mu T^mu_nu with connection terms for a lower tensor T."""
    point = np.asarray(point, dtype=float)
    tensors = geometry.curvature_tensors(point)
    mixed = tensors.inverse_metric @ np.asarray(tensor_fn(point), dtype=float)
    derivative = np.zeros((geometry.dimension, geometry.dimension, geometry.dimension), dtype=float)
    for direction in range(geometry.dimension):
        plus = point.copy()
        minus = point.copy()
        plus[direction] += step
        minus[direction] -= step
        plus_mixed = geometry.curvature_tensors(plus).inverse_metric @ np.asarray(tensor_fn(plus), dtype=float)
        minus_mixed = geometry.curvature_tensors(minus).inverse_metric @ np.asarray(tensor_fn(minus), dtype=float)
        derivative[direction] = (plus_mixed - minus_mixed) / (2 * step)
    result = np.zeros(geometry.dimension, dtype=float)
    for nu in range(geometry.dimension):
        result[nu] = sum(
            derivative[mu, mu, nu]
            + sum(tensors.christoffel[mu, mu, lam] * mixed[lam, nu] - tensors.christoffel[lam, mu, nu] * mixed[mu, lam] for lam in range(geometry.dimension))
            for mu in range(geometry.dimension)
        )
    return result


def bianchi_residual(geometry: DifferentialGeometry, point: np.ndarray, step: float = 1e-5) -> np.ndarray:
    return covariant_divergence(geometry, lambda x: geometry.curvature_tensors(x).einstein, point, step)


@dataclass(frozen=True)
class ConvergenceReport:
    steps: tuple[float, ...]
    residuals: tuple[float, ...]
    observed_order: float


def bianchi_convergence(geometry: DifferentialGeometry, point: np.ndarray, steps: tuple[float, ...]) -> ConvergenceReport:
    residuals = tuple(float(np.linalg.norm(bianchi_residual(geometry, point, step))) for step in steps)
    if len(steps) < 2 or residuals[-1] == 0.0 or residuals[0] == 0.0:
        order = 0.0
    else:
        order = float(np.log(residuals[0] / residuals[-1]) / np.log(steps[0] / steps[-1]))
    return ConvergenceReport(steps, residuals, order)
