"""Finite intervention estimates of operational causal influence."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .quantum import QuantumSystem, partial_trace

NUMERICAL_APPROXIMATION = "NUMERICAL_APPROXIMATION"


@dataclass(frozen=True)
class PauliInterventionFamily:
    name: str = "Pauli operations"
    component_label: str = NUMERICAL_APPROXIMATION

    @property
    def unitaries(self) -> tuple[np.ndarray, ...]:
        return (
            np.eye(2, dtype=complex),
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        )


@dataclass(frozen=True)
class CausalInfluenceReport:
    matrix: np.ndarray
    intervention_family: str
    number_of_interventions: int
    approximation_scope: str
    is_exact_supremum: bool = False


@dataclass(frozen=True)
class DynamicalCausalProcess:
    """A completely specified finite-dimensional state transformation."""

    name: str
    channel: callable
    component_label: str = "MATHEMATICAL_DEFINITION"

    def apply(self, rho: np.ndarray) -> np.ndarray:
        output = np.asarray(self.channel(rho), dtype=complex)
        if output.shape != rho.shape:
            raise ValueError("process channel changed the Hilbert-space dimension")
        return output


@dataclass(frozen=True)
class DynamicalCausalReport:
    influence: float
    mutual_information: float
    measure: str = "trace_distance"
    approximation_scope: str = "finite I/X intervention family"


def identity_process() -> DynamicalCausalProcess:
    return DynamicalCausalProcess("identity", lambda rho: rho.copy())


def cnot_process() -> DynamicalCausalProcess:
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    return DynamicalCausalProcess("CNOT A->B", lambda rho: cnot @ rho @ cnot.conj().T)


def _trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    difference = (first - second + (first - second).conj().T) / 2
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def sampled_dynamical_influence(
    rho: np.ndarray,
    dims: tuple[int, ...],
    process: DynamicalCausalProcess,
    source: int,
    target: int,
) -> DynamicalCausalReport:
    """Estimate direct operational influence through a known dynamical process."""
    if any(dim != 2 for dim in dims):
        raise ValueError("the reference dynamical experiment uses qubits")
    interventions = (
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
    )
    outputs = []
    for operation in interventions:
        factors = [np.eye(dim, dtype=complex) for dim in dims]
        factors[source] = operation
        full = factors[0]
        for factor in factors[1:]:
            full = np.kron(full, factor)
        evolved = process.apply(full @ rho @ full.conj().T)
        outputs.append(partial_trace(evolved, (target,), dims))
    influence = _trace_distance(outputs[0], outputs[1])
    system = QuantumSystem.from_density_matrix(rho, dims)
    return DynamicalCausalReport(influence, system.mutual_information((source,), (target,)))


def _apply_local_unitary(rho: np.ndarray, unitary: np.ndarray, subsystem: int, dims: tuple[int, ...]) -> np.ndarray:
    factors = [np.eye(dim, dtype=complex) for dim in dims]
    factors[subsystem] = unitary
    full = factors[0]
    for factor in factors[1:]:
        full = np.kron(full, factor)
    return full @ rho @ full.conj().T


def sampled_causal_influence(system: QuantumSystem, family: PauliInterventionFamily) -> CausalInfluenceReport:
    """Estimate C_ij over a finite intervention family, never an exact supremum."""
    count = len(system.subsystem_dims)
    matrix = np.zeros((count, count), dtype=float)
    for source in range(count):
        for target in range(count):
            target_dims = tuple(index for index in range(count) if index == target)
            baseline = system.reduced_state(target_dims)
            observed = []
            for unitary in family.unitaries:
                state = _apply_local_unitary(system.rho, unitary, source, system.subsystem_dims)
                observed.append(QuantumSystem.relative_entropy(partial_trace(state, target_dims, system.subsystem_dims), baseline))
            finite_values = [value for value in observed if np.isfinite(value)]
            matrix[source, target] = max(finite_values, default=0.0)
    return CausalInfluenceReport(matrix, family.name, len(family.unitaries), "finite sampled maximum; not the exact supremum")


@dataclass(frozen=True)
class InterventionResponseTensorReport:
    """The full finite intervention-response tensor C_{i,j,a}, eq. (10) of the spec.

    Unlike ``sampled_causal_influence``, no max is taken over interventions: every
    (source, target, intervention) trace-distance response is retained.
    """

    tensor: np.ndarray
    intervention_family: str
    number_of_interventions: int
    approximation_scope: str = "finite sampled trace-distance response; not the exact operational supremum"


def intervention_response_tensor(system: QuantumSystem, family: PauliInterventionFamily) -> InterventionResponseTensorReport:
    """Compute C_{i,j,a} = D_tr(rho_j^(a), rho_j^(0)) for every source i, target j, intervention a."""
    count = len(system.subsystem_dims)
    unitaries = family.unitaries
    tensor = np.zeros((count, count, len(unitaries)), dtype=float)
    baselines = [system.reduced_state((target,)) for target in range(count)]
    for source in range(count):
        for intervention_index, unitary in enumerate(unitaries):
            state = _apply_local_unitary(system.rho, unitary, source, system.subsystem_dims)
            for target in range(count):
                response = partial_trace(state, (target,), system.subsystem_dims)
                tensor[source, target, intervention_index] = _trace_distance(response, baselines[target])
    return InterventionResponseTensorReport(tensor, family.name, len(unitaries))

