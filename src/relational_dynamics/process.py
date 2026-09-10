"""Fixed-order process abstraction; no causal reversal is inferred."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

MATHEMATICAL_DEFINITION = "MATHEMATICAL_DEFINITION"


@dataclass(frozen=True)
class FixedOrderProcess:
    operations: tuple[Callable[[np.ndarray], np.ndarray], ...]
    component_label: str = MATHEMATICAL_DEFINITION

    def apply(self, state: np.ndarray) -> np.ndarray:
        result = np.asarray(state)
        for operation in self.operations:
            result = operation(result)
        return result


@dataclass(frozen=True)
class FixedOrderQuantumProcess:
    """Fixed-order channel represented by explicit Kraus operators."""

    kraus_operators: tuple[np.ndarray, ...]
    causal_order: str = "fixed"
    component_label: str = "ESTABLISHED_PHYSICS"

    @classmethod
    def from_kraus(cls, kraus_operators: tuple[np.ndarray, ...]) -> "FixedOrderQuantumProcess":
        operators = tuple(np.asarray(operator, dtype=complex) for operator in kraus_operators)
        if not operators or any(operator.ndim != 2 or operator.shape[0] != operator.shape[1] for operator in operators):
            raise ValueError("Kraus operators must be nonempty square matrices")
        return cls(operators)

    def apply(self, rho: np.ndarray) -> np.ndarray:
        return sum(operator @ rho @ operator.conj().T for operator in self.kraus_operators)

    @property
    def is_completely_positive(self) -> bool:
        return True

    @property
    def trace_preserving_error(self) -> float:
        dimension = self.kraus_operators[0].shape[0]
        completeness = sum(operator.conj().T @ operator for operator in self.kraus_operators)
        return float(np.linalg.norm(completeness - np.eye(dimension)))


@dataclass(frozen=True)
class ProcessMatrix:
    """Reserved interface for process matrices; indefinite order is not implemented."""

    array: np.ndarray

    @classmethod
    def from_array(cls, array: np.ndarray) -> "ProcessMatrix":
        raise NotImplementedError("indefinite causal order is not implemented in the reference phase")
