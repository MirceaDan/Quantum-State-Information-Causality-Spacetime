"""Deliberately broken negative control for the M4 equivariance tests (section 9).

Contains exactly one artificial label-dependent rule (index-parity initial
state), matching the confound already found in the Reverse-Theseus phase. The
M4 equivariance checks MUST fail against this model; if they do not, the
checks are not a meaningful scientific guard.
"""

from __future__ import annotations

import numpy as np

from .permutation_equivariant_process import ProcessMemoryResult, _apply_local_depolarizing, _embed_two_qubit, _partial_swap, permutation_operator
from .process_memory_observables import EquivarianceCheck, _pass
from .process_topology import ProcessTopology, permute_process
from .quantum import QuantumSystem, density_matrix

MODELING_ASSUMPTION = "MODELING_ASSUMPTION"


def simulate_process_broken_label_dependent(topology: ProcessTopology, retention: float = 1.0) -> ProcessMemoryResult:
    """Identical to ``simulate_process`` EXCEPT the initial state ignores each
    node's declared ``initial_state_angle`` and instead uses raw label parity:
    ``|0> if label % 2 == 0 else |1>``. This is the single injected bug."""
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must lie in [0, 1]")
    canonical_order = tuple(sorted(topology.node_labels))
    num_qubits = len(canonical_order)
    node_position = {label: position for position, label in enumerate(canonical_order)}

    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    state_vector = zero if canonical_order[0] % 2 == 0 else one  # <-- the injected label-dependent bug
    for label in canonical_order[1:]:
        state_vector = np.kron(state_vector, zero if label % 2 == 0 else one)  # <-- and here
    rho = density_matrix(state_vector)

    ordered_relations = sorted(topology.relations, key=lambda relation: relation.order)
    trajectory = [rho]
    for relation in ordered_relations:
        position_first, position_second = node_position[relation.first], node_position[relation.second]
        unitary = _embed_two_qubit(_partial_swap(relation.strength), position_first, position_second, num_qubits)
        rho = unitary @ rho @ unitary.conj().T
        loss_first = topology.node_attributes[relation.first].noise_parameter * (1.0 - retention)
        loss_second = topology.node_attributes[relation.second].noise_parameter * (1.0 - retention)
        rho = _apply_local_depolarizing(rho, position_first, num_qubits, loss_first)
        rho = _apply_local_depolarizing(rho, position_second, num_qubits, loss_second)
        trajectory.append(rho)

    labels = tuple(f"node_{label}" for label in canonical_order)
    final_system = QuantumSystem.from_density_matrix(rho, (2,) * num_qubits, labels=labels)
    return ProcessMemoryResult(final_system, canonical_order, node_position, tuple(trajectory), tuple(relation.order for relation in ordered_relations))


def check_broken_state_equivariance(topology: ProcessTopology, permutation: tuple[int, ...], retention: float = 1.0, tolerance: float = 1e-8) -> EquivarianceCheck:
    """The same equivariance comparison as ``check_state_equivariance``, but against
    the deliberately broken simulator. This MUST fail (negative control, section 9)."""
    original = simulate_process_broken_label_dependent(topology, retention=retention)
    permuted = simulate_process_broken_label_dependent(permute_process(topology, permutation), retention=retention)
    operator = permutation_operator(permutation, len(topology.node_labels))
    recovered = operator.conj().T @ permuted.final_system.rho @ operator
    difference = np.abs(recovered - original.final_system.rho)
    return EquivarianceCheck(float(difference.max()), float(difference.mean()), _pass(float(difference.max()), tolerance))
