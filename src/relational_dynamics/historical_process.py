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

FIXED_ORDER_CONTROL = "FIXED_ORDER_CONTROL"
ZERO_COUPLING = "ZERO_COUPLING"
SHUFFLED_GEOMETRY = "SHUFFLED_GEOMETRY"
RANDOMIZED_ORDER = "RANDOMIZED_ORDER"
COUPLED = "COUPLED"

_PAULI = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_PAULI_X = _PAULI[1]
_PAULI_Z = _PAULI[3]


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
    control_label: str = COUPLED
    schedule: tuple[int, ...] = ()
    interval_permutation: tuple[int, ...] = ()
    component_label: str = MODELING_ASSUMPTION


def build_geometry_coupled_history(
    world: GroundTruthWorld,
    retention: float = 1.0,
    length_scale: float = COUPLING_LENGTH_SCALE,
    bit_flips: frozenset[int] = frozenset(),
    force_zero_coupling: bool = False,
    interval_permutation: tuple[int, ...] | None = None,
    schedule: tuple[int, ...] | None = None,
    intervention_operator: np.ndarray | None = None,
    control_label: str | None = None,
    angle_override: tuple[float, ...] | None = None,
) -> HistoricalProcessResult:
    """Build rho_T = E_T(...E_1(rho_0)) with coupling set only by ``world``'s invariant intervals.

    ``retention`` in [0, 1] controls how much of the geometry-coupled correlation
    survives: retention=1 applies no extra information-destroying channel;
    retention=0 applies the maximal local depolarizing probability after every
    coupling step, destroying the relational pairing built up by the coupling.

    Experimental-integrity controls (section 2 of the integrity-audit spec), all
    using the SAME adjacent-pair architecture and event count as the default
    (``FIXED_ORDER_CONTROL``) construction:

    - ``force_zero_coupling``: every coupling angle is set to 0 (``ZERO_COUPLING``),
      so no geometric information enters the process at all.
    - ``interval_permutation``: a permutation of step indices ``0..event_count-2``
      that reassigns which of the world's true invariant intervals feeds which
      coupling step (``SHUFFLED_GEOMETRY``). The ground-truth geometry used for
      evaluation is never touched; only the coupling-step-to-interval mapping is
      shuffled.
    - ``schedule``: a permutation of step indices giving the ORDER in which the
      fixed set of adjacent-pair gates is executed (``RANDOMIZED_ORDER``). The
      same set of adjacent qubit pairs is coupled exactly once each; only the
      execution order changes.

    ``bit_flips`` applies an intervention (``intervention_operator``, default X)
    on the listed event indices at the moment each event enters the causal chain
    (event 0 before any coupling, event k right after the (k-1, k) coupling step,
    independent of ``schedule``). This keeps the intervention operationally
    causal and is used by ``historical_intervention_response`` to probe causal
    propagation through the actual dynamics (a frozen final state cannot signal
    across disjoint subsystems, by the no-communication theorem).

    ``angle_override`` is a DIAGNOSTIC-ONLY escape hatch for the root-cause
    audit (``historical_diagnostics.py``, control B): it replaces the
    geometry-derived coupling angles with the given values while keeping the
    same adjacent-pair coupling graph. It does not change ``coupling_angle`` or
    the default construction and defaults to ``None`` (no effect).
    """
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must lie in [0, 1]")
    event_count = len(world.coordinates)
    if event_count < 2:
        raise ValueError("a historical process needs at least two relational events")
    step_indices = list(range(event_count - 1))
    if interval_permutation is not None and sorted(interval_permutation) != step_indices:
        raise ValueError("interval_permutation must be a permutation of the coupling step indices")
    if schedule is not None and sorted(schedule) != step_indices:
        raise ValueError("schedule must be a permutation of the coupling step indices")
    if angle_override is not None and len(angle_override) != len(step_indices):
        raise ValueError("angle_override must supply one angle per coupling step")
    execution_order = list(schedule) if schedule is not None else step_indices
    flip_operator = _PAULI_X if intervention_operator is None else np.asarray(intervention_operator, dtype=complex)
    if control_label is None:
        if force_zero_coupling:
            control_label = ZERO_COUPLING
        elif interval_permutation is not None:
            control_label = SHUFFLED_GEOMETRY
        elif schedule is not None:
            control_label = RANDOMIZED_ORDER
        else:
            control_label = FIXED_ORDER_CONTROL

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
    raw_intervals = [float(world.invariant_intervals[step, step + 1]) for step in step_indices]
    if interval_permutation is not None:
        raw_intervals = [raw_intervals[interval_permutation[step]] for step in step_indices]
    angles: list[float] = [0.0] * len(step_indices)
    for step in execution_order:
        first, second = step, step + 1
        if angle_override is not None:
            theta = float(angle_override[step])
        else:
            theta = 0.0 if force_zero_coupling else coupling_angle(raw_intervals[step], length_scale)
        angles[step] = theta
        unitary = _embed_adjacent_pair(_partial_swap(theta), first, event_count)
        rho = unitary @ rho @ unitary.conj().T
        rho = _apply_local_depolarizing(rho, first, event_count, loss_probability)
        rho = _apply_local_depolarizing(rho, second, event_count, loss_probability)
        rho = _maybe_flip(rho, second)
    labels = tuple(f"event_{index}" for index in range(event_count))
    system = QuantumSystem.from_density_matrix(rho, (2,) * event_count, labels=labels)
    return HistoricalProcessResult(
        system,
        COUPLING_MECHANISM,
        tuple(angles),
        retention,
        event_count,
        length_scale,
        control_label,
        tuple(execution_order),
        tuple(interval_permutation) if interval_permutation is not None else (),
    )


def _trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    difference = (first - second + (first - second).conj().T) / 2
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


@dataclass(frozen=True)
class HistoricalCausalResponseReport:
    """Operational C_{i,j,a}: replay the SAME process with an intervention at
    event i and compare event j's final marginal to the unperturbed baseline.
    Unlike a post-hoc intervention on the frozen final state, this respects
    no-signaling and can show genuine propagation through the dynamics.
    """

    tensor: np.ndarray
    intervention_family: str = "single_qubit_pauli"
    number_of_interventions: int = 1
    approximation_scope: str = "finite single-intervention replay of the fixed-order geometry-coupled process"


def historical_intervention_response(
    world: GroundTruthWorld,
    retention: float = 1.0,
    length_scale: float = COUPLING_LENGTH_SCALE,
    intervention_operator: np.ndarray | None = None,
    intervention_label: str = "X",
    force_zero_coupling: bool = False,
    interval_permutation: tuple[int, ...] | None = None,
    schedule: tuple[int, ...] | None = None,
) -> HistoricalCausalResponseReport:
    """Replay-based causal probe; ``intervention_operator`` defaults to Pauli-X.

    Accepts the same experimental-integrity controls as
    ``build_geometry_coupled_history`` so every control variant (COUPLED,
    ZERO_COUPLING, SHUFFLED_GEOMETRY, RANDOMIZED_ORDER) gets a causal observable
    built the same way.
    """
    common = dict(
        retention=retention,
        length_scale=length_scale,
        force_zero_coupling=force_zero_coupling,
        interval_permutation=interval_permutation,
        schedule=schedule,
    )
    baseline = build_geometry_coupled_history(world, **common)
    event_count = baseline.event_count
    baseline_marginals = [baseline.system.reduced_state((target,)) for target in range(event_count)]
    tensor = np.zeros((event_count, event_count), dtype=float)
    for source in range(event_count):
        perturbed = build_geometry_coupled_history(world, bit_flips=frozenset({source}), intervention_operator=intervention_operator, **common)
        for target in range(event_count):
            if target == source:
                continue
            response = perturbed.system.reduced_state((target,))
            tensor[source, target] = _trace_distance(response, baseline_marginals[target])
    return HistoricalCausalResponseReport(tensor, intervention_family=intervention_label)
