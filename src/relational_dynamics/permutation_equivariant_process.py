"""Milestone 4: permutation-equivariant quantum process-memory simulation.

CONVENTION (must be understood before reading the code): because a relabeling
permutation acts on the SAME label set (e.g. {0,...,N-1} to itself), the
canonical tensor-factor layout ``sorted(topology.node_labels)`` is *always*
{0,...,N-1} regardless of which permutation was applied. This lets LABEL and
TENSOR POSITION coincide numerically, which is what makes every observable
below directly comparable across relabelings via simple array gathers
(``M[np.ix_(permutation, permutation)]``) instead of ad-hoc re-sorting.

This module is deliberately independent of ``historical_process.py``: it does
not import it and does not reuse its (index-adjacency-confounded) internals.
It exists so the M4 process-memory substrate can be validated without
inheriting the Reverse-Theseus confounds (see docs/AUDIT.md, "Milestone 4").

Q = 0 and Delta_munu = 0: this module contains no speculative sector at all.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np

from .process_topology import ProcessTopology
from .quantum import QuantumSystem, density_matrix, partial_trace

MATHEMATICAL_DEFINITION = "MATHEMATICAL_DEFINITION"
NUMERICAL_APPROXIMATION = "NUMERICAL_APPROXIMATION"

_PAULI = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)
PAULI_X = _PAULI[1]
PAULI_Z = _PAULI[3]
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


# ---------------------------------------------------------------------------
# Permutation operator on a composite qubit Hilbert space (pure linear algebra)
# ---------------------------------------------------------------------------


def permutation_operator(permutation: tuple[int, ...], num_qubits: int) -> np.ndarray:
    """Unitary U with U|combo> = |combo'>, combo'[permutation[k]] = combo[k].

    In words: the basis-state amplitude that was at tensor position ``k`` ends
    up at tensor position ``permutation[k]``. Restricted to qubits (dimension 2
    per factor); see LIMITATIONS.md for the uniform-dimension restriction.
    """
    if sorted(permutation) != list(range(num_qubits)):
        raise ValueError("permutation must be a bijection on range(num_qubits)")
    dimension = 2**num_qubits
    operator = np.zeros((dimension, dimension), dtype=complex)
    for combo in itertools.product((0, 1), repeat=num_qubits):
        input_index = 0
        for bit in combo:
            input_index = input_index * 2 + bit
        output_bits = [0] * num_qubits
        for old_position, bit in enumerate(combo):
            output_bits[permutation[old_position]] = bit
        output_index = 0
        for bit in output_bits:
            output_index = output_index * 2 + bit
        operator[output_index, input_index] = 1.0
    return operator


def _embed_local(operator: np.ndarray, position: int, num_qubits: int) -> np.ndarray:
    factors = [np.eye(2, dtype=complex)] * position + [operator] + [np.eye(2, dtype=complex)] * (num_qubits - position - 1)
    full = factors[0]
    for factor in factors[1:]:
        full = np.kron(full, factor)
    return full


def _embed_two_qubit(unitary4: np.ndarray, position_first: int, position_second: int, num_qubits: int) -> np.ndarray:
    """Embed a 4x4 two-qubit unitary acting on ARBITRARY (not necessarily adjacent) positions."""
    others = [position for position in range(num_qubits) if position not in (position_first, position_second)]
    order = [position_first, position_second] + others  # order[new_position] = old_position
    permutation = [0] * num_qubits
    for new_position, old_position in enumerate(order):
        permutation[old_position] = new_position
    reorder = permutation_operator(tuple(permutation), num_qubits)
    embedded = np.kron(unitary4, np.eye(2 ** (num_qubits - 2), dtype=complex))
    return reorder.conj().T @ embedded @ reorder


def _partial_swap(theta: float) -> np.ndarray:
    return np.cos(theta) * np.eye(4, dtype=complex) + 1j * np.sin(theta) * _SWAP


def _apply_local_depolarizing(rho: np.ndarray, position: int, num_qubits: int, probability: float) -> np.ndarray:
    """Deterministic CPTP depolarizing channel (Kraus sum, no RNG): equivariance can be
    tested to floating precision even with this 'noise' enabled (section 13)."""
    if probability <= 0.0:
        return rho
    weights = (np.sqrt(max(1 - 3 * probability / 4, 0.0)),) + (np.sqrt(probability / 4),) * 3
    result = np.zeros_like(rho)
    for weight, pauli in zip(weights, _PAULI):
        operator = weight * _embed_local(pauli, position, num_qubits)
        result = result + operator @ rho @ operator.conj().T
    return result


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessMemoryResult:
    final_system: QuantumSystem
    canonical_order: tuple[int, ...]
    node_position: dict[int, int]
    trajectory: tuple[np.ndarray, ...]
    applied_relation_order: tuple[int, ...]
    component_label: str = NUMERICAL_APPROXIMATION


def simulate_process(
    topology: ProcessTopology,
    retention: float = 1.0,
    interventions: dict[int, np.ndarray] | None = None,
    intervention_at_order: int | float = -np.inf,
) -> ProcessMemoryResult:
    """Simulate rho_0 -> E_1 -> rho_1 -> E_2 -> ... using ONLY topology-declared data.

    Node/tensor position is ``sorted(topology.node_labels).index(label)``; since
    a relabeling permutation acts on the same label SET, this canonical order
    is always {0,...,N-1} and label/position coincide numerically (see module
    docstring). Relations are applied in ascending ``relation.order`` (an
    explicit declared field), never in raw list/dict iteration order.

    ``interventions`` maps a node label to a single-qubit unitary to apply at
    the schedule point ``intervention_at_order`` (before any relation whose
    ``order >= intervention_at_order``; default -inf applies interventions to
    the initial state, before any relation).
    """
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must lie in [0, 1]")
    canonical_order = tuple(sorted(topology.node_labels))
    num_qubits = len(canonical_order)
    node_position = {label: position for position, label in enumerate(canonical_order)}
    for label in canonical_order:
        if topology.node_attributes[label].dimension != 2:
            raise ValueError("this reference implementation supports qubit (dimension=2) nodes only")

    def _apply_interventions(state: np.ndarray) -> np.ndarray:
        if not interventions:
            return state
        for label, operator in interventions.items():
            embedded = _embed_local(np.asarray(operator, dtype=complex), node_position[label], num_qubits)
            state = embedded @ state @ embedded.conj().T
        return state

    vectors = []
    for label in canonical_order:
        angle = topology.node_attributes[label].initial_state_angle
        vectors.append(np.array([np.cos(angle / 2), np.sin(angle / 2)], dtype=complex))
    state_vector = vectors[0]
    for vector in vectors[1:]:
        state_vector = np.kron(state_vector, vector)
    rho = density_matrix(state_vector)

    ordered_relations = sorted(topology.relations, key=lambda relation: relation.order)
    applied_intervention = False
    if intervention_at_order == -np.inf:
        rho = _apply_interventions(rho)
        applied_intervention = True
    trajectory = [rho]
    for relation in ordered_relations:
        if not applied_intervention and relation.order >= intervention_at_order:
            rho = _apply_interventions(rho)
            applied_intervention = True
        position_first, position_second = node_position[relation.first], node_position[relation.second]
        unitary = _embed_two_qubit(_partial_swap(relation.strength), position_first, position_second, num_qubits)
        rho = unitary @ rho @ unitary.conj().T
        loss_first = topology.node_attributes[relation.first].noise_parameter * (1.0 - retention)
        loss_second = topology.node_attributes[relation.second].noise_parameter * (1.0 - retention)
        rho = _apply_local_depolarizing(rho, position_first, num_qubits, loss_first)
        rho = _apply_local_depolarizing(rho, position_second, num_qubits, loss_second)
        trajectory.append(rho)
    if not applied_intervention:
        rho = _apply_interventions(rho)
        trajectory.append(rho)

    labels = tuple(f"node_{label}" for label in canonical_order)
    final_system = QuantumSystem.from_density_matrix(rho, (2,) * num_qubits, labels=labels)
    return ProcessMemoryResult(
        final_system,
        canonical_order,
        node_position,
        tuple(trajectory),
        tuple(relation.order for relation in ordered_relations),
    )
