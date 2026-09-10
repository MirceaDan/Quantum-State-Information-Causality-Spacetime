"""Geometry-coupled relational process generator for the Reverse-Theseus phase.

MODEL ASSUMPTION (declared, not established physics): the only place geometric
information enters the forward construction is the two-qubit coupling angle
between adjacent relational events, set by a monotonically decreasing function
of the magnitude of their Lorentzian invariant interval. No metric component,
coordinate, or world/time label is ever written into the quantum state, the
Kraus operators, or the observable vectors. This is the ``historically coupled
relational information`` construction contrasted with the independent-observable
null baseline in ``geometry_reconstruction.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .ground_truth import GroundTruthWorld
from .quantum import QuantumSystem, density_matrix

MODELING_ASSUMPTION = "MODELING_ASSUMPTION"

COUPLING_MECHANISM = "geometry_dependent_channel_strength_from_invariant_interval"
COUPLING_LENGTH_SCALE = 1.0
MAX_COUPLING_ANGLE = np.pi / 4
MAX_INFORMATION_LOSS_PROBABILITY = 0.5

_PAULI = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def coupling_angle(interval: float, length_scale: float = COUPLING_LENGTH_SCALE) -> float:
    """Declared model assumption: stronger relational coupling for smaller |interval|."""
    return float(MAX_COUPLING_ANGLE * np.exp(-abs(interval) / max(length_scale, 1e-12)))


def _partial_swap(theta: float) -> np.ndarray:
    """U(theta) = cos(theta) I + i sin(theta) SWAP; unitary because SWAP^2 = I and SWAP is Hermitian."""
    return np.cos(theta) * np.eye(4, dtype=complex) + 1j * np.sin(theta) * _SWAP


def _embed_local(operator: np.ndarray, position: int, event_count: int) -> np.ndarray:
    factors = [np.eye(2, dtype=complex)] * position + [operator] + [np.eye(2, dtype=complex)] * (event_count - position - 1)
    full = factors[0]
    for factor in factors[1:]:
        full = np.kron(full, factor)
    return full


def _embed_adjacent_pair(operator: np.ndarray, first: int, event_count: int) -> np.ndarray:
    left_dim = 2**first
    right_dim = 2 ** (event_count - first - 2)
    return np.kron(np.kron(np.eye(left_dim, dtype=complex), operator), np.eye(right_dim, dtype=complex))


def _apply_local_depolarizing(rho: np.ndarray, position: int, event_count: int, probability: float) -> np.ndarray:
    """Standard single-qubit depolarizing channel embedded at ``position``; models information loss."""
    if probability <= 0.0:
        return rho
    weights = (np.sqrt(max(1 - 3 * probability / 4, 0.0)),) + (np.sqrt(probability / 4),) * 3
    result = np.zeros_like(rho)
    for weight, pauli in zip(weights, _PAULI):
        operator = weight * _embed_local(pauli, position, event_count)
        result = result + operator @ rho @ operator.conj().T
    return result


@dataclass(frozen=True)
class HistoricalProcessResult:
    system: QuantumSystem
    coupling_mechanism: str
    coupling_angles: tuple[float, ...]
    retention: float
    event_count: int
    length_scale: float = COUPLING_LENGTH_SCALE
    component_label: str = MODELING_ASSUMPTION


def build_geometry_coupled_history(
    world: GroundTruthWorld,
    retention: float = 1.0,
    length_scale: float = COUPLING_LENGTH_SCALE,
    bit_flips: frozenset[int] = frozenset(),
) -> HistoricalProcessResult:
    """Build rho_T = E_T(...E_1(rho_0)) with coupling set only by ``world``'s invariant intervals.

    ``retention`` in [0, 1] controls how much of the geometry-coupled correlation
    survives: retention=1 applies no extra information-destroying channel;
    retention=0 applies the maximal local depolarizing probability after every
    coupling step, destroying the relational pairing built up by the coupling.

    ``bit_flips`` applies an X intervention on the listed event indices at the
    moment each event enters the causal chain (event 0 before any coupling,
    event k right after the (k-1, k) coupling step and before the (k, k+1)
    step). This keeps the intervention operationally causal: it can influence
    later events through subsequent coupling steps but cannot signal earlier
    events whose coupling steps already completed. Used by
    ``historical_intervention_response`` to probe causal propagation through
    the actual dynamics (a frozen final state cannot signal across disjoint
    subsystems, by the no-communication theorem).
    """
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must lie in [0, 1]")
    event_count = len(world.coordinates)
    if event_count < 2:
        raise ValueError("a historical process needs at least two relational events")
    flip_operator = np.array([[0, 1], [1, 0]], dtype=complex)

    def _maybe_flip(state: np.ndarray, index: int) -> np.ndarray:
        if index not in bit_flips:
            return state
        operator = _embed_local(flip_operator, index, event_count)
        return operator @ state @ operator.conj().T

    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    state_vector = zero if 0 % 2 == 0 else one
    for index in range(1, event_count):
        state_vector = np.kron(state_vector, zero if index % 2 == 0 else one)
    rho = density_matrix(state_vector)
    rho = _maybe_flip(rho, 0)
    loss_probability = MAX_INFORMATION_LOSS_PROBABILITY * (1.0 - retention)
    angles: list[float] = []
    for first in range(event_count - 1):
        second = first + 1
        interval = float(world.invariant_intervals[first, second])
        theta = coupling_angle(interval, length_scale)
        angles.append(theta)
        unitary = _embed_adjacent_pair(_partial_swap(theta), first, event_count)
        rho = unitary @ rho @ unitary.conj().T
        rho = _apply_local_depolarizing(rho, first, event_count, loss_probability)
        rho = _apply_local_depolarizing(rho, second, event_count, loss_probability)
        rho = _maybe_flip(rho, second)
    labels = tuple(f"event_{index}" for index in range(event_count))
    system = QuantumSystem.from_density_matrix(rho, (2,) * event_count, labels=labels)
    return HistoricalProcessResult(system, COUPLING_MECHANISM, tuple(angles), retention, event_count, length_scale)


def _trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    difference = (first - second + (first - second).conj().T) / 2
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


@dataclass(frozen=True)
class HistoricalCausalResponseReport:
    """Operational C_{i,j,a}: replay the SAME geometry-coupled process with an
    initial bit-flip intervention at event i and compare event j's final marginal
    to the unperturbed baseline. Unlike a post-hoc intervention on the frozen
    final state, this respects no-signaling and can show genuine propagation
    through the geometry-coupled dynamics.
    """

    tensor: np.ndarray
    intervention_family: str = "initial_bit_flip"
    number_of_interventions: int = 1
    approximation_scope: str = "finite single-intervention replay of the fixed-order geometry-coupled process"


def historical_intervention_response(
    world: GroundTruthWorld,
    retention: float = 1.0,
    length_scale: float = COUPLING_LENGTH_SCALE,
) -> HistoricalCausalResponseReport:
    baseline = build_geometry_coupled_history(world, retention, length_scale)
    event_count = baseline.event_count
    baseline_marginals = [baseline.system.reduced_state((target,)) for target in range(event_count)]
    tensor = np.zeros((event_count, event_count), dtype=float)
    for source in range(event_count):
        perturbed = build_geometry_coupled_history(world, retention, length_scale, bit_flips=frozenset({source}))
        for target in range(event_count):
            if target == source:
                continue
            response = perturbed.system.reduced_state((target,))
            tensor[source, target] = _trace_distance(response, baseline_marginals[target])
    return HistoricalCausalResponseReport(tensor)
