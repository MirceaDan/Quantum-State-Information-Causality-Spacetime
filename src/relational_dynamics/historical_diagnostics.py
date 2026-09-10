"""Root-cause analysis for the order/label confound found by the integrity audit.

This module performs DIAGNOSIS ONLY. It does not change the coupling law, the
synthetic geometries, Q, Delta_munu, or the reconstruction optimizer. It exists
to answer one question: what information other than geometry lets the
reconstruction pipeline distinguish or fit event configurations?

Q = 0 and Delta_munu = 0 throughout; see ``speculative_gate.BaselineGate``.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np

from .ground_truth import GroundTruthWorld
from .historical_process import (
    COUPLED,
    RANDOMIZED_ORDER,
    build_geometry_coupled_history,
)
from .historical_reconstruction import (
    FLEXIBLE,
    build_historical_observables,
    build_relabeled_world,
    compare_historical_baselines,
    randomize_historical_observables,
    run_label_permutation_control,
    run_process_variant_experiment,
)
from .speculative_gate import BaselineGate

LABEL_CONFUND = "LABEL_CONFUND"
ORDER_CONFUND = "ORDER_CONFUND"
PROCESS_STRUCTURE_CONFUND = "PROCESS_STRUCTURE_CONFUND"
SPLIT_DESIGN_CONFUND = "SPLIT_DESIGN_CONFUND"
MULTIPLE_CONFUNDS = "MULTIPLE_CONFUNDS"
NO_CONFUND_FOUND = "NO_CONFUND_FOUND"


# ---------------------------------------------------------------------------
# 1-3: what changes under a pure relabeling, component by component
# ---------------------------------------------------------------------------


def coupling_graph_edges(event_count: int) -> list[frozenset[int]]:
    """Which PHYSICAL event pairs the fixed-order construction couples, by raw index adjacency."""
    return [frozenset({index, index + 1}) for index in range(event_count - 1)]


def relabeled_coupling_graph_edges(permutation: tuple[int, ...]) -> list[frozenset[int]]:
    """The same construction on a relabeled world, expressed back in ORIGINAL event identities."""
    return [frozenset({permutation[index], permutation[index + 1]}) for index in range(len(permutation) - 1)]


def preserves_adjacency(permutation: tuple[int, ...]) -> bool:
    """True only for the identity and full reversal of a path graph (the two automorphisms of a line)."""
    event_count = len(permutation)
    original = set(coupling_graph_edges(event_count))
    relabeled = set(relabeled_coupling_graph_edges(permutation))
    return original == relabeled


def _unpermute(matrix: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    """Map a relabeled-world matrix back into original-event index space."""
    inverse = tuple(int(index) for index in np.argsort(permutation))
    return matrix[np.ix_(inverse, inverse)]


def _unpermute_triples(triples: dict[tuple[int, int, int], float], permutation: tuple[int, ...]) -> dict[tuple[int, int, int], float]:
    inverse = {new: old for new, old in enumerate(permutation)}
    remapped = {}
    for (first, second, third), value in triples.items():
        mapped = tuple(sorted((inverse[first], inverse[second], inverse[third])))
        remapped[mapped] = value
    return remapped


@dataclass(frozen=True)
class ObservableInvarianceReport:
    """Component-by-component comparison of O(world) vs O(relabeled_world), mapped back to
    original event identity. A truly geometry-only pipeline should make every field below
    equal (up to floating tolerance); any "changed" entry is direct evidence of a confound."""

    permutation: tuple[int, ...]
    coupling_graph_preserved: bool
    original_edges: tuple[frozenset[int], ...]
    relabeled_edges: tuple[frozenset[int], ...]
    shared_edges: tuple[frozenset[int], ...]
    angles_on_shared_edges_match: bool
    pairwise_information_max_abs_diff: float
    pairwise_information_changed: bool
    multipartite_information_max_abs_diff: float
    multipartite_information_changed: bool
    causal_estimate_changed: bool
    causal_estimate_hamming_distance: int
    intervention_tensor_max_abs_diff: float
    intervention_tensor_changed: bool


def compare_observables_under_relabeling(world: GroundTruthWorld, permutation: tuple[int, ...], retention: float = 1.0, tolerance: float = 1e-9) -> ObservableInvarianceReport:
    relabeled_world = build_relabeled_world(world, permutation)

    original_process = build_geometry_coupled_history(world, retention=retention)
    relabeled_process = build_geometry_coupled_history(relabeled_world, retention=retention)
    original_observables = build_historical_observables(original_process, world)
    relabeled_observables = build_historical_observables(relabeled_process, relabeled_world)

    original_edges = coupling_graph_edges(len(world.coordinates))
    relabeled_edges_in_original_identity = relabeled_coupling_graph_edges(permutation)
    shared_edges = [edge for edge in original_edges if edge in relabeled_edges_in_original_identity]

    # angle comparison restricted to edges physically present in BOTH processes
    original_angle_by_edge = {frozenset({step, step + 1}): angle for step, angle in enumerate(original_process.coupling_angles)}
    relabeled_angle_by_edge = {frozenset({permutation[step], permutation[step + 1]}): angle for step, angle in enumerate(relabeled_process.coupling_angles)}
    angles_match = all(abs(original_angle_by_edge[edge] - relabeled_angle_by_edge[edge]) < tolerance for edge in shared_edges) if shared_edges else False

    unpermuted_information = _unpermute(relabeled_observables.pairwise_information, permutation)
    information_diff = float(np.max(np.abs(unpermuted_information - original_observables.pairwise_information)))

    unpermuted_causal = _unpermute(relabeled_observables.causal_estimate, permutation)
    causal_hamming = int(np.sum(unpermuted_causal != original_observables.causal_estimate))

    unpermuted_multipartite = _unpermute_triples(relabeled_observables.multipartite_information, permutation)
    common_triples = set(unpermuted_multipartite) & set(original_observables.multipartite_information)
    multipartite_diff = (
        float(max(abs(unpermuted_multipartite[triple] - original_observables.multipartite_information[triple]) for triple in common_triples))
        if common_triples
        else float("nan")
    )

    unpermuted_tensor = _unpermute(relabeled_observables.intervention_tensor[:, :, 0], permutation)
    tensor_diff = float(np.max(np.abs(unpermuted_tensor - original_observables.intervention_tensor[:, :, 0])))

    return ObservableInvarianceReport(
        permutation,
        set(original_edges) == set(relabeled_edges_in_original_identity),
        tuple(original_edges),
        tuple(relabeled_edges_in_original_identity),
        tuple(shared_edges),
        angles_match,
        information_diff,
        information_diff > tolerance,
        multipartite_diff if not np.isnan(multipartite_diff) else 0.0,
        (not np.isnan(multipartite_diff)) and multipartite_diff > tolerance,
        causal_hamming > 0,
        causal_hamming,
        tensor_diff,
        tensor_diff > tolerance,
    )


# ---------------------------------------------------------------------------
# 4/6: is reconstruction loss predictable from non-geometric features?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationLossPoint:
    permutation: tuple[int, ...]
    preserves_adjacency: bool
    kendall_displacement: int
    test_loss_mean: float
    causal_f1_mean: float


def _kendall_displacement(permutation: tuple[int, ...]) -> int:
    """Sum of absolute index displacement from identity; a simple non-geometric 'how shuffled' feature."""
    return int(sum(abs(value - index) for index, value in enumerate(permutation)))


def enumerate_permutation_losses(world: GroundTruthWorld, seeds: tuple[int, ...] = tuple(range(5))) -> list[PermutationLossPoint]:
    """Exhaustive for event_count<=5 (<=120 permutations); every permutation is a pure relabeling."""
    event_count = len(world.coordinates)
    if event_count > 5:
        raise ValueError("exhaustive permutation enumeration is restricted to event_count <= 5")
    points = []
    for permutation in itertools.permutations(range(event_count)):
        relabeled_world = build_relabeled_world(world, permutation) if permutation != tuple(range(event_count)) else world
        report = run_process_variant_experiment(relabeled_world, COUPLED, seeds)
        points.append(
            PermutationLossPoint(
                permutation,
                preserves_adjacency(permutation),
                _kendall_displacement(permutation),
                report.seed_statistics[FLEXIBLE].mean,
                report.baselines[FLEXIBLE].causal_f1,
            )
        )
    return points


@dataclass(frozen=True)
class NonGeometricPredictabilityReport:
    world_id: str
    permutation_count: int
    adjacency_preserving_loss_mean: float
    adjacency_breaking_loss_mean: float
    adjacency_effect_ratio: float
    displacement_loss_correlation: float
    loss_variance_explained_by_adjacency: float


def analyze_non_geometric_predictability(world: GroundTruthWorld, seeds: tuple[int, ...] = tuple(range(5))) -> NonGeometricPredictabilityReport:
    """Section 4/6: can loss be predicted from index/order/adjacency alone, without any geometric observable?

    ``preserves_adjacency`` and ``kendall_displacement`` are computed from the
    permutation ONLY (pure combinatorics on integer labels); no geometric
    quantity (interval, coordinate, metric) is used to compute them.
    """
    points = enumerate_permutation_losses(world, seeds)
    preserving = [point.test_loss_mean for point in points if point.preserves_adjacency]
    breaking = [point.test_loss_mean for point in points if not point.preserves_adjacency]
    preserving_mean = float(np.mean(preserving)) if preserving else float("nan")
    breaking_mean = float(np.mean(breaking)) if breaking else float("nan")

    displacements = np.array([point.kendall_displacement for point in points], dtype=float)
    losses = np.array([point.test_loss_mean for point in points], dtype=float)
    if np.std(displacements) > 0 and np.std(losses) > 0:
        correlation = float(np.corrcoef(displacements, losses)[0, 1])
    else:
        correlation = 0.0

    group_labels = np.array([1.0 if point.preserves_adjacency else 0.0 for point in points])
    total_variance = float(np.var(losses))
    if total_variance > 0 and len(set(group_labels.tolist())) > 1:
        between_group_variance = float(np.var([preserving_mean] * len(preserving) + [breaking_mean] * len(breaking)))
        explained = between_group_variance / total_variance
    else:
        explained = 0.0

    return NonGeometricPredictabilityReport(
        world.world_id,
        len(points),
        preserving_mean,
        breaking_mean,
        breaking_mean / preserving_mean if preserving_mean not in (0.0,) and not np.isnan(preserving_mean) else float("nan"),
        correlation,
        explained,
    )


# ---------------------------------------------------------------------------
# 5: explicit non-geometric controls A-D
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonGeometricControlsReport:
    coupled_test_loss_mean: float
    control_a_randomized_labels_test_loss_mean: float
    control_b_randomized_strengths_test_loss_mean: float
    control_c_randomized_order_test_loss_mean: float
    control_d_randomized_indexing_test_loss_mean: float


def run_non_geometric_controls(world: GroundTruthWorld, seeds: tuple[int, ...] = tuple(range(10)), retention: float = 1.0, control_seed: int = 0) -> NonGeometricControlsReport:
    """Controls A-D from the root-cause-analysis request, all through the identical pipeline.

    A. same observable values, randomized event labels (pairwise-permutation
       control on the observable bundle itself; ``randomize_historical_observables``).
    B. same causal graph (adjacency), randomized coupling strengths
       (``angle_override`` with values unrelated to any interval).
    C. same coupling strengths, randomized causal/execution order (``RANDOMIZED_ORDER``).
    D. same geometry, randomized event indexing (``build_relabeled_world``).
    """
    BaselineGate().validate()
    event_count = len(world.coordinates)

    coupled_process = build_geometry_coupled_history(world, retention=retention)
    coupled_observables = build_historical_observables(coupled_process, world)
    coupled_losses = [compare_historical_baselines(world, coupled_observables, seed)[FLEXIBLE].test_loss for seed in seeds]

    control_a_losses = [
        compare_historical_baselines(world, randomize_historical_observables(coupled_observables, seed), seed)[FLEXIBLE].test_loss
        for seed in seeds
    ]

    rng = np.random.default_rng(control_seed)
    random_angles = tuple(rng.uniform(0.0, np.pi / 4, size=event_count - 1))
    control_b_process = build_geometry_coupled_history(world, retention=retention, angle_override=random_angles)
    control_b_observables = build_historical_observables(control_b_process, world)
    control_b_losses = [compare_historical_baselines(world, control_b_observables, seed)[FLEXIBLE].test_loss for seed in seeds]

    control_c_report = run_process_variant_experiment(world, RANDOMIZED_ORDER, seeds, retention, schedule_seed=control_seed)
    control_c_losses = tuple(control_c_report.seed_statistics[FLEXIBLE].per_seed)

    permutation = tuple(int(index) for index in np.random.default_rng(control_seed).permutation(event_count))
    relabeled_world = build_relabeled_world(world, permutation)
    control_d_process = build_geometry_coupled_history(relabeled_world, retention=retention)
    control_d_observables = build_historical_observables(control_d_process, relabeled_world)
    control_d_losses = [compare_historical_baselines(relabeled_world, control_d_observables, seed)[FLEXIBLE].test_loss for seed in seeds]

    return NonGeometricControlsReport(
        float(np.mean(coupled_losses)),
        float(np.mean(control_a_losses)),
        float(np.mean(control_b_losses)),
        float(np.mean(control_c_losses)),
        float(np.mean(control_d_losses)),
    )


# ---------------------------------------------------------------------------
# 10: root-cause classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RootCauseReport:
    process_structure_confund: bool
    label_confund: bool
    order_confund: bool
    split_design_confund: bool
    classification: str
    evidence: dict[str, object]


def run_root_cause_analysis(world: GroundTruthWorld, seeds: tuple[int, ...] = tuple(range(5)), permutation: tuple[int, ...] | None = None) -> RootCauseReport:
    """Diagnosis only: does not repair anything. Q=0, Delta_munu=0 unchanged throughout."""
    BaselineGate().validate()
    event_count = len(world.coordinates)
    if permutation is None:
        rng = np.random.default_rng(0)
        permutation = tuple(int(index) for index in rng.permutation(event_count))

    invariance = compare_observables_under_relabeling(world, permutation)
    predictability = analyze_non_geometric_predictability(world, seeds)
    controls = run_non_geometric_controls(world, seeds)
    label_report = run_label_permutation_control(world, seeds)

    # PROCESS_STRUCTURE_CONFUND: the fixed-order coupling graph is index-adjacency-defined,
    # so a relabeling changes WHICH physical event pairs are coupled at all.
    process_structure_confund = not invariance.coupling_graph_preserved and (
        invariance.pairwise_information_changed or invariance.causal_estimate_changed or invariance.intervention_tensor_changed
    )

    # LABEL_CONFUND: the initial computational-basis state is assigned by index
    # parity, independent of geometry; detect via the observable-level effect
    # (observables change under relabeling even when restricted to shared edges)
    # and via control A (does randomizing labels while keeping observable
    # values fixed change the fit outcome at all).
    label_confund = invariance.pairwise_information_changed or invariance.causal_estimate_changed

    # ORDER_CONFUND: loss correlates with a purely combinatorial (non-geometric)
    # displacement-from-identity feature, or differs strongly between
    # adjacency-preserving and adjacency-breaking permutations.
    order_confund = abs(predictability.displacement_loss_correlation) > 0.3 or predictability.loss_variance_explained_by_adjacency > 0.3

    # SPLIT_DESIGN_CONFUND: the train/validation/test pair split is defined on
    # raw integer pair indices, so which REAL relations end up in which split
    # changes under relabeling even though the split *procedure* is fixed.
    # Detect indirectly: control D (randomized indexing, same geometry) should
    # reproduce the label_permutation_control finding.
    split_design_confund = label_report.classification == "LABEL_OR_ORDER_CONFOUND_DETECTED" and controls.control_d_randomized_indexing_test_loss_mean != controls.coupled_test_loss_mean

    confound_flags = [process_structure_confund, label_confund, order_confund, split_design_confund]
    active = sum(1 for flag in confound_flags if flag)
    if active == 0:
        classification = NO_CONFUND_FOUND
    elif active == 1:
        classification = (
            PROCESS_STRUCTURE_CONFUND
            if process_structure_confund
            else LABEL_CONFUND
            if label_confund
            else ORDER_CONFUND
            if order_confund
            else SPLIT_DESIGN_CONFUND
        )
    else:
        classification = MULTIPLE_CONFUNDS

    evidence = {
        "invariance_report": invariance,
        "predictability_report": predictability,
        "controls_report": controls,
        "label_permutation_report": label_report,
    }
    return RootCauseReport(process_structure_confund, label_confund, order_confund, split_design_confund, classification, evidence)
