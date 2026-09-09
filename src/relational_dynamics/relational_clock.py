"""A controlled Page-Wootters relational-clock experiment."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .quantum import QuantumSystem, density_matrix, partial_trace

MATHEMATICAL_DEFINITION = "MATHEMATICAL_DEFINITION"
ESTABLISHED_PHYSICS = "ESTABLISHED_PHYSICS"


@dataclass(frozen=True)
class RelationalClockReport:
    conditional_states: tuple[np.ndarray, np.ndarray]
    clock_probabilities: tuple[float, float]
    clock_distinguishability: float
    conditional_fidelity: float
    constraint_residual: float
    consistent_with_page_wootters_toy_model: bool
    external_time_used: bool = False
    computational_index_used_as_time: bool = False


@dataclass(frozen=True)
class PageWoottersExperiment:
    matter_hamiltonian: np.ndarray
    clock_hamiltonian: np.ndarray
    physical_state: np.ndarray
    clock_states: tuple[np.ndarray, np.ndarray]
    component_label: str = MATHEMATICAL_DEFINITION

    @classmethod
    def two_level_controlled(cls) -> "PageWoottersExperiment":
        matter_hamiltonian = np.diag([0.0, 1.0])
        clock_hamiltonian = np.diag([0.0, -1.0])
        state = np.zeros(4, dtype=complex)
        state[[0, 3]] = 1 / np.sqrt(2)
        clock_states = (
            np.array([1.0, 1.0], dtype=complex) / np.sqrt(2),
            np.array([1.0, -1.0], dtype=complex) / np.sqrt(2),
        )
        return cls(matter_hamiltonian, clock_hamiltonian, state, clock_states)

    @property
    def total_constraint(self) -> np.ndarray:
        return np.kron(self.matter_hamiltonian, np.eye(2)) + np.kron(np.eye(2), self.clock_hamiltonian)

    def conditional_matter_state(self, clock_state: np.ndarray) -> tuple[np.ndarray, float]:
        clock_state = np.asarray(clock_state, dtype=complex)
        projector = np.outer(clock_state, clock_state.conj())
        projected = np.kron(np.eye(2), projector) @ density_matrix(self.physical_state)
        probability = float(np.trace(projected).real)
        if probability <= 1e-14:
            raise ValueError("clock projector has zero probability")
        conditional = partial_trace(projected / probability, (0,), (2, 2))
        return conditional, probability

    def run(self) -> RelationalClockReport:
        constraint_residual = float(np.linalg.norm(self.total_constraint @ self.physical_state))
        conditioned = tuple(self.conditional_matter_state(clock_state) for clock_state in self.clock_states)
        states = (conditioned[0][0], conditioned[1][0])
        probabilities = (conditioned[0][1], conditioned[1][1])
        fidelity = QuantumSystem.fidelity(states[0], states[1])
        distinguishability = 1.0 - fidelity
        consistent = constraint_residual < 1e-10 and fidelity < 1e-10 and all(probability > 0 for probability in probabilities)
        return RelationalClockReport(states, probabilities, distinguishability, fidelity, constraint_residual, consistent)
