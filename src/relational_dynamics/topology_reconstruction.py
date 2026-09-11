"""Milestone 5: distances, observable ablations, and the two inverse-experiment tasks
(M5.1 cycle-vs-DAG classification, M5.2 exact topology recovery). No ML, no learned
weights, no gradient-based fitting: every decision here is a fixed, deterministic
distance comparison (section 21).
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np

from .causal_topology import CausalTopology, canonical_topology_key, has_cycle, is_isomorphic, valid_topological_node_orders
from .topology_observables import TopologyObservableBundle, build_observable_bundle, default_process_parameters

MI_ONLY = "MI_ONLY"
MULTIPARTITE_ONLY = "MULTIPARTITE_ONLY"
CAUSAL_ONLY = "CAUSAL_ONLY"
MEMORY_ONLY = "MEMORY_ONLY"
CAUSAL_PLUS_MEMORY = "CAUSAL_PLUS_MEMORY"
ALL_OBSERVABLES = "ALL"

OBSERVABLE_WEIGHTS = {
    MI_ONLY: (1.0, 0.0, 0.0, 0.0),
    MULTIPARTITE_ONLY: (0.0, 1.0, 0.0, 0.0),
    CAUSAL_ONLY: (0.0, 0.0, 1.0, 0.0),
    MEMORY_ONLY: (0.0, 0.0, 0.0, 1.0),
    CAUSAL_PLUS_MEMORY: (0.0, 0.0, 0.5, 0.5),
    ALL_OBSERVABLES: (0.25, 0.25, 0.25, 0.25),
}

# fixed candidate-signature seeds for M5.2 (never overlap TEST_SEEDS; section 20)
# 12 seeds chosen a priori as a minimally-adequate Monte-Carlo sample size for
# averaging out per-instance parameter dependence (see LIMITATIONS.md); not
# tuned against any specific test query.
TRAIN_SEEDS = tuple(range(100, 112))
TEST_SEEDS = (7, 8, 9)

# Per the observable-ablation study (section 13, documented in RESULTS.md): the
# causal intervention-response observable was consistently the most
# discriminative single component for topology recovery in this small-N
# regime, well ahead of MI/multipartite/memory alone. This constant records
# that a-priori finding; it is NOT re-tuned per test case.
RECOMMENDED_MODE = CAUSAL_ONLY


def assert_no_seed_leakage(test_seeds: tuple[int, ...] = TEST_SEEDS, train_seeds: tuple[int, ...] = TRAIN_SEEDS) -> None:
    overlap = set(test_seeds) & set(train_seeds)
    if overlap:
        raise ValueError(f"train/test seed leakage detected: {overlap}")


# ---------------------------------------------------------------------------
# Distances (section 12)
# ---------------------------------------------------------------------------


def distance_mutual_information(first: TopologyObservableBundle, second: TopologyObservableBundle) -> float:
    upper = np.triu_indices(first.node_count, k=1)
    return float(np.linalg.norm(first.pairwise_information[upper] - second.pairwise_information[upper]))


def distance_multipartite(first: TopologyObservableBundle, second: TopologyObservableBundle) -> float:
    common = set(first.multipartite_information) & set(second.multipartite_information)
    if not common:
        return 0.0
    return float(np.mean([abs(first.multipartite_information[key] - second.multipartite_information[key]) for key in common]))


def distance_causal(first: TopologyObservableBundle, second: TopologyObservableBundle) -> float:
    common = set(first.intervention_tensor) & set(second.intervention_tensor)
    if not common:
        return 0.0
    return float(np.mean([abs(first.intervention_tensor[key] - second.intervention_tensor[key]) for key in common]))


def distance_memory(first: TopologyObservableBundle, second: TopologyObservableBundle) -> float:
    return abs(first.memory_score - second.memory_score)


@dataclass(frozen=True)
class ComponentScales:
    """Documented normalization rule (section 12): each component is divided by its
    own mean pairwise distance across a REFERENCE (train-only) ensemble, computed
    once and never re-tuned on a test/query topology."""

    mi: float
    multipartite: float
    causal: float
    memory: float


def compute_component_scales(reference_bundles: list[TopologyObservableBundle]) -> ComponentScales:
    mi_values, multi_values, causal_values, memory_values = [], [], [], []
    for first_index in range(len(reference_bundles)):
        for second_index in range(first_index + 1, len(reference_bundles)):
            first, second = reference_bundles[first_index], reference_bundles[second_index]
            mi_values.append(distance_mutual_information(first, second))
            multi_values.append(distance_multipartite(first, second))
            causal_values.append(distance_causal(first, second))
            memory_values.append(distance_memory(first, second))

    def _scale(values: list[float]) -> float:
        mean_value = float(np.mean(values)) if values else 0.0
        return mean_value if mean_value > 1e-12 else 1.0

    return ComponentScales(_scale(mi_values), _scale(multi_values), _scale(causal_values), _scale(memory_values))


def combined_distance(
    first: TopologyObservableBundle,
    second: TopologyObservableBundle,
    mode: str = ALL_OBSERVABLES,
    scales: ComponentScales | None = None,
) -> float:
    weights = OBSERVABLE_WEIGHTS[mode]
    scales = scales or ComponentScales(1.0, 1.0, 1.0, 1.0)
    d_mi = distance_mutual_information(first, second) / scales.mi
    d_multi = distance_multipartite(first, second) / scales.multipartite
    d_causal = distance_causal(first, second) / scales.causal
    d_memory = distance_memory(first, second) / scales.memory
    return weights[0] * d_mi + weights[1] * d_multi + weights[2] * d_causal + weights[3] * d_memory


def _permute_bundle(bundle: TopologyObservableBundle, permutation: tuple[int, ...]) -> TopologyObservableBundle:
    """Relabel an ALREADY-COMPUTED bundle by ``permutation`` (same gather convention as M4)."""
    mi = bundle.pairwise_information[np.ix_(permutation, permutation)]
    multi = {tuple(sorted((permutation[a], permutation[b], permutation[c]))): value for (a, b, c), value in bundle.multipartite_information.items()}
    causal = {(permutation[a], permutation[b]): value for (a, b), value in bundle.intervention_tensor.items()}
    return TopologyObservableBundle(mi, multi, causal, bundle.memory_score, bundle.node_count)


def _bundle_relabelings(bundle: TopologyObservableBundle) -> tuple[TopologyObservableBundle, ...]:
    return tuple(_permute_bundle(bundle, permutation) for permutation in itertools.permutations(range(bundle.node_count)))


def isomorphism_aware_distance(
    query_bundle: TopologyObservableBundle,
    candidate_bundle: TopologyObservableBundle,
    mode: str = ALL_OBSERVABLES,
    scales: "ComponentScales | None" = None,
) -> float:
    """Section 4/12: the query's labeling relative to any canonical candidate labeling is
    UNKNOWN, so the observable distance must be minimized over every relabeling of the
    candidate (graph-isomorphism-aware comparison), never a raw fixed-label distance.
    Feasible because M5 is restricted to small N (<=5-6, exhaustive permutations)."""
    best = float("inf")
    for permuted in _bundle_relabelings(candidate_bundle):
        best = min(best, combined_distance(query_bundle, permuted, mode, scales))
    return best


# ---------------------------------------------------------------------------
# Bundle construction helpers (train/reference pools; section 20 no-leakage)
# ---------------------------------------------------------------------------


def _reference_node_order(topology: CausalTopology) -> tuple[int, ...] | None:
    return None if has_cycle(topology) else valid_topological_node_orders(topology)[0]


def build_reference_bundles(topology: CausalTopology, seeds: tuple[int, ...] = TRAIN_SEEDS) -> list[TopologyObservableBundle]:
    node_order = _reference_node_order(topology)
    return [build_observable_bundle(topology, default_process_parameters(topology, seed=seed), node_order) for seed in seeds]


@dataclass(frozen=True)
class ReferenceLibrary:
    candidate_topologies: tuple[CausalTopology, ...]
    bundles_by_key: dict[str, tuple[TopologyObservableBundle, ...]]
    relabelings_by_key: dict[str, tuple[tuple[TopologyObservableBundle, ...], ...]]
    scales: ComponentScales
    train_seeds: tuple[int, ...]


def build_reference_library(candidate_topologies: list[CausalTopology], train_seeds: tuple[int, ...] = TRAIN_SEEDS) -> ReferenceLibrary:
    assert_no_seed_leakage(test_seeds=TEST_SEEDS, train_seeds=train_seeds)
    ordered = tuple(sorted(candidate_topologies, key=canonical_topology_key))
    bundles_by_key = {
        str(canonical_topology_key(topology)): tuple(build_reference_bundles(topology, train_seeds)) for topology in ordered
    }
    relabelings_by_key = {
        key: tuple(_bundle_relabelings(bundle) for bundle in bundles) for key, bundles in bundles_by_key.items()
    }
    all_bundles = [bundle for bundles in bundles_by_key.values() for bundle in bundles]
    return ReferenceLibrary(ordered, bundles_by_key, relabelings_by_key, compute_component_scales(all_bundles), train_seeds)


def build_query_bundle(topology: CausalTopology, seed: int = TEST_SEEDS[0], node_order: tuple[int, ...] | None = "auto") -> TopologyObservableBundle:
    resolved_order = _reference_node_order(topology) if node_order == "auto" else node_order
    return build_observable_bundle(topology, default_process_parameters(topology, seed=seed), resolved_order)


# ---------------------------------------------------------------------------
# M5.1: cycle vs DAG classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleClassificationResult:
    predicted_label: str
    dag_min_distance: float
    cyclic_min_distance: float


def classify_cycle_vs_dag(
    query_bundle: TopologyObservableBundle,
    dag_reference_bundles: list[TopologyObservableBundle],
    cyclic_reference_bundles: list[TopologyObservableBundle],
    mode: str = ALL_OBSERVABLES,
    scales: ComponentScales | None = None,
) -> CycleClassificationResult:
    dag_distances = [isomorphism_aware_distance(query_bundle, reference, mode, scales) for reference in dag_reference_bundles]
    cyclic_distances = [isomorphism_aware_distance(query_bundle, reference, mode, scales) for reference in cyclic_reference_bundles]
    dag_min = min(dag_distances) if dag_distances else float("inf")
    cyclic_min = min(cyclic_distances) if cyclic_distances else float("inf")
    predicted = "CYCLIC" if cyclic_min < dag_min else "DAG"
    return CycleClassificationResult(predicted, dag_min, cyclic_min)


# ---------------------------------------------------------------------------
# M5.2: exact topology recovery (deterministic nearest-candidate matching)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconstructionResult:
    recovered_topology: CausalTopology
    recovered_distance: float
    all_candidate_distances: dict[str, float]
    is_isomorphic_to_truth: bool
    ambiguous: bool = False
    tied_canonical_keys: tuple[str, ...] = ()


def reconstruct_topology(
    query_bundle: TopologyObservableBundle,
    candidate_topologies: list[CausalTopology],
    ground_truth: CausalTopology | None = None,
    mode: str = ALL_OBSERVABLES,
    scales: ComponentScales | None = None,
    candidate_seeds: tuple[int, ...] = TRAIN_SEEDS,
) -> ReconstructionResult:
    """Deterministic exhaustive candidate matching (section 11); no ML."""
    assert_no_seed_leakage(test_seeds=TEST_SEEDS, train_seeds=candidate_seeds)
    all_distances: dict[str, float] = {}
    candidate_by_key: dict[str, CausalTopology] = {}
    for candidate in sorted(candidate_topologies, key=canonical_topology_key):
        reference_bundles = build_reference_bundles(candidate, candidate_seeds)
        candidate_distance = float(np.mean([isomorphism_aware_distance(query_bundle, reference, mode, scales) for reference in reference_bundles]))
        key = str(canonical_topology_key(candidate))
        all_distances[key] = candidate_distance
        candidate_by_key[key] = candidate
    best_distance = min(all_distances.values())
    tied_keys = tuple(sorted(key for key, distance in all_distances.items() if np.isclose(distance, best_distance, atol=1e-12, rtol=0.0)))
    best_topology = candidate_by_key[tied_keys[0]]
    isomorphic = is_isomorphic(best_topology, ground_truth) if ground_truth is not None else False
    return ReconstructionResult(best_topology, best_distance, all_distances, isomorphic, len(tied_keys) > 1, tied_keys)


def reconstruct_with_library(
    query_bundle: TopologyObservableBundle,
    library: ReferenceLibrary,
    ground_truth: CausalTopology | None = None,
    mode: str = ALL_OBSERVABLES,
) -> ReconstructionResult:
    """Same inverse rule as ``reconstruct_topology``, using frozen training-only references."""
    all_distances: dict[str, float] = {}
    candidate_by_key = {str(canonical_topology_key(topology)): topology for topology in library.candidate_topologies}
    for key, reference_relabelings in library.relabelings_by_key.items():
        per_reference = [
            min(combined_distance(query_bundle, relabeling, mode, library.scales) for relabeling in relabelings)
            for relabelings in reference_relabelings
        ]
        all_distances[key] = float(np.mean(per_reference))
    best_distance = min(all_distances.values())
    tied_keys = tuple(sorted(key for key, distance in all_distances.items() if np.isclose(distance, best_distance, atol=1e-12, rtol=0.0)))
    recovered = candidate_by_key[tied_keys[0]]
    isomorphic = is_isomorphic(recovered, ground_truth) if ground_truth is not None else False
    return ReconstructionResult(recovered, best_distance, all_distances, isomorphic, len(tied_keys) > 1, tied_keys)


# ---------------------------------------------------------------------------
# Negative controls (section 18)
# ---------------------------------------------------------------------------


def shuffle_observable_bundle(bundle: TopologyObservableBundle, seed: int) -> TopologyObservableBundle:
    """Control A: permute observable VALUES while preserving marginals, breaking pairing."""
    rng = np.random.default_rng(seed)
    upper = np.triu_indices(bundle.node_count, k=1)
    values = bundle.pairwise_information[upper].copy()
    rng.shuffle(values)
    shuffled_mi = np.zeros_like(bundle.pairwise_information)
    shuffled_mi[upper] = values
    shuffled_mi[(upper[1], upper[0])] = values

    multi_keys = list(bundle.multipartite_information.keys())
    multi_values = np.array(list(bundle.multipartite_information.values()), dtype=float)
    rng.shuffle(multi_values)
    shuffled_multi = dict(zip(multi_keys, multi_values))

    causal_keys = list(bundle.intervention_tensor.keys())
    causal_values = np.array(list(bundle.intervention_tensor.values()), dtype=float)
    rng.shuffle(causal_values)
    shuffled_causal = dict(zip(causal_keys, causal_values))

    return TopologyObservableBundle(shuffled_mi, shuffled_multi, shuffled_causal, bundle.memory_score, bundle.node_count)


def topology_independent_bundle(node_count: int, seed: int) -> TopologyObservableBundle:
    """Control B: a process with NO declared relations at all (topology information removed)."""
    empty_topology = CausalTopology(tuple(range(node_count)), ())
    node_order = _reference_node_order(empty_topology)
    return build_observable_bundle(empty_topology, default_process_parameters(empty_topology, seed=seed), node_order)
