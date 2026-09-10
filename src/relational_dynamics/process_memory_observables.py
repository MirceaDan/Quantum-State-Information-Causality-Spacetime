"""Memory/relational observables for the M4 permutation-equivariant process, and the
equivariance comparison utilities used by the M4 test suite and runner.

Every observable here is indexed by NODE LABEL (never by an arbitrary Python
list/dict iteration position), so the equivariance check reduces to a direct
array gather: for permutation pi, ``O_after[i, j] == O_before[pi[i], pi[j]]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np

from .permutation_equivariant_process import (
    PAULI_X,
    ProcessMemoryResult,
    permutation_operator,
    simulate_process,
)
from .process_topology import ProcessTopology, permute_process
from .quantum import QuantumSystem

MODELING_ASSUMPTION = "MODELING_ASSUMPTION"
NUMERICAL_APPROXIMATION = "NUMERICAL_APPROXIMATION"


def _trace_distance(first: np.ndarray, second: np.ndarray) -> float:
    difference = (first - second + (first - second).conj().T) / 2
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


# ---------------------------------------------------------------------------
# Relational observables, all label-indexed
# ---------------------------------------------------------------------------


def pairwise_mutual_information(result: ProcessMemoryResult) -> np.ndarray:
    """M[i, j] = I(node i : node j) on the final state; indexed by LABEL, i == position."""
    labels = result.canonical_order
    count = len(labels)
    matrix = np.zeros((count, count), dtype=float)
    for first, second in itertools.combinations(labels, 2):
        value = result.final_system.mutual_information((result.node_position[first],), (result.node_position[second],))
        matrix[first, second] = matrix[second, first] = value
    return matrix


def multipartite_information(result: ProcessMemoryResult) -> dict[tuple[int, int, int], float]:
    """Co-information I(i:j:k), keyed by SORTED LABEL triples."""
    labels = result.canonical_order
    values: dict[tuple[int, int, int], float] = {}
    for first, second, third in itertools.combinations(labels, 3):
        positions = (result.node_position[first], result.node_position[second], result.node_position[third])
        system = result.final_system
        s_a, s_b, s_c = system.entropy((positions[0],)), system.entropy((positions[1],)), system.entropy((positions[2],))
        s_ab, s_ac, s_bc = system.entropy(positions[:2]), system.entropy((positions[0], positions[2])), system.entropy(positions[1:])
        s_abc = system.entropy(positions)
        values[(first, second, third)] = float(s_a + s_b + s_c - s_ab - s_ac - s_bc + s_abc)
    return values


@dataclass(frozen=True)
class InterventionResponseReport:
    """R(X -> Y): replay-based response, keyed by (source_label, target_label)."""

    tensor: dict[tuple[int, int], float]
    intervention_family: str = "single_qubit_pauli"
    approximation_scope: str = "finite single-intervention replay; not the exact operational supremum"


def intervention_response_tensor(
    topology: ProcessTopology,
    retention: float = 1.0,
    intervention_operator_matrix: np.ndarray | None = None,
    intervention_label: str = "X",
    intervention_at_order: int | float = -np.inf,
) -> InterventionResponseReport:
    """R(source -> target) = D_tr(rho_target^(intervened), rho_target^(baseline)), all label-indexed."""
    operator = PAULI_X if intervention_operator_matrix is None else intervention_operator_matrix
    baseline = simulate_process(topology, retention=retention)
    labels = baseline.canonical_order
    baseline_marginals = {label: baseline.final_system.reduced_state((baseline.node_position[label],)) for label in labels}
    tensor: dict[tuple[int, int], float] = {}
    for source in labels:
        perturbed = simulate_process(
            topology, retention=retention, interventions={source: operator}, intervention_at_order=intervention_at_order
        )
        for target in labels:
            if target == source:
                continue
            response = perturbed.final_system.reduced_state((perturbed.node_position[target],))
            tensor[(source, target)] = _trace_distance(response, baseline_marginals[target])
    return InterventionResponseReport(tensor, intervention_label)


def process_memory_score(topology: ProcessTopology, retention: float = 1.0) -> float:
    """A neutral, explicitly-defined finite diagnostic -- NOT a claim of a rigorous
    non-Markovianity measure (see docs/METHODS.md, Milestone 4).

    Definition: the mean intervention response R(X -> Y) restricted to pairs
    where the intervention is inserted BEFORE X's own relations execute and Y
    is only reachable through at least one further relation step (i.e. the
    response must survive propagation through the declared schedule, not just
    a single local step). A value near zero means the process behaves as if
    only the immediately preceding step mattered; a larger value means
    information from an earlier declared step still shows up later -- a
    simple, literal notion of "the future depends on more than the last step",
    which is the property this score is intended to capture and nothing more.
    """
    response = intervention_response_tensor(topology, retention=retention)
    if not response.tensor:
        return 0.0
    return float(np.mean(list(response.tensor.values())))


# ---------------------------------------------------------------------------
# Equivariance comparison utilities (section 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquivarianceCheck:
    max_error: float
    mean_error: float
    passed: bool


def _pass(max_error: float, tolerance: float) -> bool:
    return bool(max_error <= tolerance)


def check_state_equivariance(topology: ProcessTopology, permutation: tuple[int, ...], retention: float = 1.0, tolerance: float = 1e-8) -> EquivarianceCheck:
    original = simulate_process(topology, retention=retention)
    permuted_topology = permute_process(topology, permutation)
    permuted = simulate_process(permuted_topology, retention=retention)
    operator = permutation_operator(permutation, len(topology.node_labels))
    recovered = operator.conj().T @ permuted.final_system.rho @ operator
    difference = np.abs(recovered - original.final_system.rho)
    return EquivarianceCheck(float(difference.max()), float(difference.mean()), _pass(float(difference.max()), tolerance))


def check_pairwise_observable_equivariance(topology: ProcessTopology, permutation: tuple[int, ...], retention: float = 1.0, tolerance: float = 1e-8) -> EquivarianceCheck:
    original = pairwise_mutual_information(simulate_process(topology, retention=retention))
    permuted = pairwise_mutual_information(simulate_process(permute_process(topology, permutation), retention=retention))
    recovered = permuted[np.ix_(permutation, permutation)]
    difference = np.abs(recovered - original)
    return EquivarianceCheck(float(difference.max()), float(difference.mean()), _pass(float(difference.max()), tolerance))


def check_multipartite_observable_equivariance(topology: ProcessTopology, permutation: tuple[int, ...], retention: float = 1.0, tolerance: float = 1e-8) -> EquivarianceCheck:
    original = multipartite_information(simulate_process(topology, retention=retention))
    permuted = multipartite_information(simulate_process(permute_process(topology, permutation), retention=retention))
    errors = []
    for (first, second, third), value in original.items():
        mapped_key = tuple(sorted((permutation[first], permutation[second], permutation[third])))
        errors.append(abs(permuted[mapped_key] - value))
    if not errors:
        return EquivarianceCheck(0.0, 0.0, True)
    return EquivarianceCheck(float(max(errors)), float(np.mean(errors)), _pass(float(max(errors)), tolerance))


def check_intervention_tensor_equivariance(topology: ProcessTopology, permutation: tuple[int, ...], retention: float = 1.0, tolerance: float = 1e-8) -> EquivarianceCheck:
    original = intervention_response_tensor(topology, retention=retention).tensor
    permuted = intervention_response_tensor(permute_process(topology, permutation), retention=retention).tensor
    errors = [abs(permuted[(permutation[source], permutation[target])] - value) for (source, target), value in original.items()]
    if not errors:
        return EquivarianceCheck(0.0, 0.0, True)
    return EquivarianceCheck(float(max(errors)), float(np.mean(errors)), _pass(float(max(errors)), tolerance))


def check_memory_score_invariance(topology: ProcessTopology, permutation: tuple[int, ...], retention: float = 1.0, tolerance: float = 1e-8) -> EquivarianceCheck:
    """A GLOBAL SCALAR must be exactly invariant (not merely equivariant) under relabeling."""
    original = process_memory_score(topology, retention=retention)
    permuted = process_memory_score(permute_process(topology, permutation), retention=retention)
    error = abs(permuted - original)
    return EquivarianceCheck(error, error, _pass(error, tolerance))


def check_global_scalar_invariants(topology: ProcessTopology, permutation: tuple[int, ...], retention: float = 1.0, tolerance: float = 1e-8) -> dict[str, EquivarianceCheck]:
    """Section 8, Test D: entropy of the full state, eigenvalue spectrum, trace/positivity diagnostics."""
    original = simulate_process(topology, retention=retention).final_system
    permuted = simulate_process(permute_process(topology, permutation), retention=retention).final_system

    original_entropy = original.entropy(tuple(range(len(topology.node_labels))))
    permuted_entropy = permuted.entropy(tuple(range(len(topology.node_labels))))
    entropy_error = abs(permuted_entropy - original_entropy)

    original_spectrum = np.sort(np.linalg.eigvalsh((original.rho + original.rho.conj().T) / 2))
    permuted_spectrum = np.sort(np.linalg.eigvalsh((permuted.rho + permuted.rho.conj().T) / 2))
    spectrum_error = float(np.max(np.abs(original_spectrum - permuted_spectrum)))

    original_diag = original.validate_state()
    permuted_diag = permuted.validate_state()
    diagnostics_error = max(
        abs(original_diag.trace_error - permuted_diag.trace_error),
        abs(original_diag.hermiticity_error - permuted_diag.hermiticity_error),
        abs(original_diag.positivity_violation - permuted_diag.positivity_violation),
    )

    return {
        "full_state_entropy": EquivarianceCheck(entropy_error, entropy_error, _pass(entropy_error, tolerance)),
        "density_matrix_spectrum": EquivarianceCheck(spectrum_error, spectrum_error, _pass(spectrum_error, tolerance)),
        "quantum_state_diagnostics": EquivarianceCheck(diagnostics_error, diagnostics_error, _pass(diagnostics_error, tolerance)),
    }
