"""Numerical fixed-point and periodic-process diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .reconstruction import QuantumChannel


@dataclass(frozen=True)
class FixedPointReport:
    classification: str
    fixed_point: np.ndarray
    fixed_point_error: float
    stability: str
    detected_period: int | None
    causal_loop_candidate: bool = False


@dataclass(frozen=True)
class FixedPointExperiment:
    process: QuantumChannel
    tolerance: float = 1e-10
    max_iterations: int = 1000

    def run(self, initial_state: np.ndarray) -> FixedPointReport:
        current = np.asarray(initial_state, dtype=complex)
        orbit = [current.copy()]
        for iteration in range(1, self.max_iterations + 1):
            next_state = self.process.apply(current)
            error = float(np.linalg.norm(next_state - current))
            if error < self.tolerance:
                return FixedPointReport("FIXED_POINT", next_state, error, "convergent", None)
            for period in range(1, min(iteration, 8) + 1):
                if np.linalg.norm(next_state - orbit[-period]) < self.tolerance:
                    return FixedPointReport("PERIODIC_PROCESS", next_state, error, "periodic", period)
            orbit.append(next_state.copy())
            current = next_state
        error = float(np.linalg.norm(self.process.apply(current) - current))
        return FixedPointReport("NO_CONVERGENCE", current, error, "unstable_or_unresolved", None)
