"""Finite-dimensional quantum primitives and relational clock conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import fractional_matrix_power

ESTABLISHED_PHYSICS = "ESTABLISHED_PHYSICS"
MATHEMATICAL_DEFINITION = "MATHEMATICAL_DEFINITION"
NUMERICAL_APPROXIMATION = "NUMERICAL_APPROXIMATION"


def density_matrix(state: np.ndarray) -> np.ndarray:
    """Return |psi><psi| after validating a normalized state vector."""
    vector = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError("state vector must be normalized")
    return np.outer(vector, vector.conj())


def _validate_density_matrix(rho: np.ndarray, tolerance: float = 1e-10) -> None:
    matrix = np.asarray(rho, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density matrix must be square")
    if not np.isclose(np.trace(matrix), 1.0, atol=tolerance):
        raise ValueError("density matrix must have unit trace")
    if np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).min() < -tolerance:
        raise ValueError("density matrix must be positive semidefinite")


def partial_trace(rho: np.ndarray, keep: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """Trace out all tensor factors except those listed in ``keep``."""
    dims = tuple(int(dim) for dim in dims)
    keep = tuple(keep)
    if sorted(keep) != list(keep) or len(set(keep)) != len(keep):
        raise ValueError("keep must contain sorted unique subsystem indices")
    if np.prod(dims) != np.asarray(rho).shape[0]:
        raise ValueError("dims do not match density matrix")
    tensor = np.asarray(rho, dtype=complex).reshape(*dims, *dims)
    trace_out = [index for index in range(len(dims)) if index not in keep]
    for index in reversed(trace_out):
        tensor = np.trace(tensor, axis1=index, axis2=index + tensor.ndim // 2)
    kept_dims = tuple(dims[index] for index in keep)
    size = int(np.prod(kept_dims))
    return tensor.reshape(size, size)


def _entropy(rho: np.ndarray, tolerance: float = 1e-12) -> float:
    eigenvalues = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    positive = eigenvalues[eigenvalues > tolerance]
    return float(-np.sum(positive * np.log(positive)))


@dataclass(frozen=True)
class StateDiagnostics:
    trace_error: float
    hermiticity_error: float
    minimum_eigenvalue: float
    positivity_violation: float


@dataclass(frozen=True)
class ConditionalState:
    state: np.ndarray
    clock_reading: object
    probability: float
    computational_index: int | None = None
    external_time: float | None = None


@dataclass(frozen=True)
class ConstraintSolution:
    status: str
    kernel_dimension: int
    state: np.ndarray | None
    residual: float | None
    eigenvalues: np.ndarray


class HamiltonianConstraintSolver:
    """Finite-dimensional solver for the physical-state constraint C|Psi> = 0."""

    component_label = MATHEMATICAL_DEFINITION

    def __init__(self, constraint: np.ndarray, tolerance: float = 1e-10) -> None:
        operator = np.asarray(constraint, dtype=complex)
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError("constraint must be a square matrix")
        if not np.allclose(operator, operator.conj().T, atol=tolerance):
            raise ValueError("the reference solver requires a Hermitian constraint")
        self.constraint = (operator + operator.conj().T) / 2
        self.tolerance = tolerance

    def solve(self, preferred_state: np.ndarray | None = None) -> ConstraintSolution:
        eigenvalues, eigenvectors = np.linalg.eigh(self.constraint)
        kernel = np.flatnonzero(np.abs(eigenvalues) <= self.tolerance)
        if len(kernel) == 0:
            return ConstraintSolution("NO_PHYSICAL_STATE", 0, None, None, eigenvalues)

        basis = eigenvectors[:, kernel]
        if preferred_state is None:
            state = basis[:, 0]
        else:
            vector = np.asarray(preferred_state, dtype=complex).reshape(-1)
            if vector.shape[0] != self.constraint.shape[0]:
                raise ValueError("preferred state dimension does not match constraint")
            projection = basis @ (basis.conj().T @ vector)
            if np.linalg.norm(projection) <= self.tolerance:
                state = basis[:, 0]
            else:
                state = projection / np.linalg.norm(projection)
        residual = float(np.linalg.norm(self.constraint @ state))
        status = "EXACT_KERNEL" if residual <= self.tolerance else "APPROXIMATE_KERNEL"
        return ConstraintSolution(status, len(kernel), state, residual, eigenvalues)


@dataclass
class QuantumSystem:
    rho: np.ndarray
    subsystem_dims: tuple[int, ...]
    labels: tuple[str, ...] = ("matter", "clock", "observer")
    component_label: str = ESTABLISHED_PHYSICS

    def __post_init__(self) -> None:
        self.rho = np.asarray(self.rho, dtype=complex)
        self.subsystem_dims = tuple(self.subsystem_dims)
        if len(self.subsystem_dims) != len(self.labels):
            self.labels = tuple(f"subsystem_{i}" for i in range(len(self.subsystem_dims)))
        _validate_density_matrix(self.rho)

    @classmethod
    def from_density_matrix(cls, rho: np.ndarray, subsystem_dims: Sequence[int], labels: Sequence[str] | None = None) -> "QuantumSystem":
        dimensions = tuple(subsystem_dims)
        chosen_labels = tuple(labels) if labels is not None else ("matter", "clock", "observer")
        return cls(np.asarray(rho, dtype=complex), dimensions, chosen_labels[: len(dimensions)])

    def validate_state(self) -> StateDiagnostics:
        hermiticity = float(np.linalg.norm(self.rho - self.rho.conj().T))
        minimum = float(np.linalg.eigvalsh((self.rho + self.rho.conj().T) / 2).min())
        return StateDiagnostics(
            trace_error=float(abs(np.trace(self.rho) - 1)),
            hermiticity_error=hermiticity,
            minimum_eigenvalue=minimum,
            positivity_violation=max(0.0, -minimum),
        )

    def reduced_state(self, subsystems: Sequence[int]) -> np.ndarray:
        return partial_trace(self.rho, tuple(subsystems), self.subsystem_dims)

    def entropy(self, subsystems: Sequence[int]) -> float:
        return _entropy(self.reduced_state(subsystems))

    def mutual_information(self, first: Sequence[int], second: Sequence[int]) -> float:
        union = tuple(sorted(set(first) | set(second)))
        return self.entropy(first) + self.entropy(second) - self.entropy(union)

    @staticmethod
    def fidelity(first: np.ndarray, second: np.ndarray) -> float:
        root = fractional_matrix_power((first + first.conj().T) / 2, 0.5)
        middle = root @ second @ root
        value = np.trace(fractional_matrix_power((middle + middle.conj().T) / 2, 0.5)).real
        return float(max(0.0, min(1.0, value * value)))

    @staticmethod
    def relative_entropy(first: np.ndarray, second: np.ndarray) -> float:
        """D(first || second), returning infinity for unsupported support."""
        first_eigenvalues, first_vectors = np.linalg.eigh((first + first.conj().T) / 2)
        second_eigenvalues = np.linalg.eigvalsh((second + second.conj().T) / 2)
        if np.any((first_eigenvalues > 1e-12) & (second_eigenvalues <= 1e-12)):
            return float("inf")
        first_log = first_vectors @ np.diag(np.log(np.maximum(first_eigenvalues, 1e-12))) @ first_vectors.conj().T
        second_log = np.linalg.eigh((second + second.conj().T) / 2)
        second_log_matrix = second_log[1] @ np.diag(np.log(np.maximum(second_log[0], 1e-12))) @ second_log[1].conj().T
        return float(np.trace(first @ (first_log - second_log_matrix)).real)

    def conditional_matter_state(self, clock_outcome: int) -> ConditionalState:
        """Condition matter on a clock computational-basis projector.

        The clock reading is metadata; no external simulation-time variable is created.
        """
        if len(self.subsystem_dims) < 2 or self.subsystem_dims[1] <= clock_outcome:
            raise ValueError("clock subsystem or clock outcome is invalid")
        projector = np.zeros((self.subsystem_dims[1], self.subsystem_dims[1]), dtype=complex)
        projector[clock_outcome, clock_outcome] = 1
        operator = projector
        for dim in self.subsystem_dims[2:]:
            operator = np.kron(operator, np.eye(dim))
        full_operator = np.kron(np.eye(self.subsystem_dims[0]), operator)
        projected = full_operator @ self.rho @ full_operator
        probability = float(np.trace(projected).real)
        if probability <= 1e-12:
            raise ValueError("clock outcome has negligible probability")
        matter = partial_trace(projected / probability, (0,), self.subsystem_dims)
        return ConditionalState(matter, clock_outcome, probability)

    def constraint_residual(self, constraint: np.ndarray) -> float:
        vector_density_residual = constraint @ self.rho
        return float(np.linalg.norm(vector_density_residual))
