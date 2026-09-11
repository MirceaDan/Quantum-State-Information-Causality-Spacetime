"""Clean M5.1/M5.2 validation experiments.

Separates three questions that must not be mixed:
1. same frozen physical process under node relabeling (structural correctness),
2. same frozen process under another valid execution schedule (schedule sensitivity),
3. same topology with independently resampled nuisance parameters (robustness).
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np

from .causal_topology import CausalTopology, graph_recovery_metrics, has_cycle, is_isomorphic, valid_topological_node_orders
from .topology_observables import (
    PhysicalProcessInstance,
    build_observable_bundle_from_instance,
    create_physical_process_instance,
    relabel_physical_process,
    reschedule_physical_process,
    same_process_relabeling_report,
)
from .topology_reconstruction import (
    ALL_OBSERVABLES,
    CAUSAL_ONLY,
    CAUSAL_PLUS_MEMORY,
    MEMORY_ONLY,
    MI_ONLY,
    MULTIPARTITE_ONLY,
    ComponentScales,
    ReferenceLibrary,
    ReconstructionResult,
    combined_distance,
    reconstruct_topology,
    reconstruct_with_library,
)

OBSERVABLE_MODES = (MI_ONLY, MULTIPARTITE_ONLY, CAUSAL_ONLY, MEMORY_ONLY, CAUSAL_PLUS_MEMORY, ALL_OBSERVABLES)


def _reconstruct(query, candidates, truth, mode, scales):
    if isinstance(candidates, ReferenceLibrary):
        return reconstruct_with_library(query, candidates, truth, mode)
    return reconstruct_topology(query, candidates, truth, mode, scales)


@dataclass(frozen=True)
class ReconstructionRelabelingReport:
    permutations_tested: int
    observable_max_error: float
    prediction_invariant: bool
    truth_correctness_invariant: bool
    per_permutation: tuple[dict[str, object], ...]


def reconstruction_relabeling_report(
    instance: PhysicalProcessInstance,
    candidates: list[CausalTopology] | ReferenceLibrary,
    mode: str,
    scales: ComponentScales | None = None,
) -> ReconstructionRelabelingReport:
    """Reconstruct every relabeling of the SAME physical instance; never resample."""
    original_bundle = build_observable_bundle_from_instance(instance)
    original = _reconstruct(original_bundle, candidates, instance.topology, mode, scales)
    structural = same_process_relabeling_report(instance)
    records = []
    predictions_invariant = True
    correctness_invariant = True
    for permutation in itertools.permutations(instance.topology.nodes):
        relabeled = relabel_physical_process(instance, permutation)
        result = _reconstruct(build_observable_bundle_from_instance(relabeled), candidates, relabeled.topology, mode, scales)
        same_prediction = is_isomorphic(original.recovered_topology, result.recovered_topology)
        same_correctness = original.is_isomorphic_to_truth == result.is_isomorphic_to_truth
        predictions_invariant &= same_prediction
        correctness_invariant &= same_correctness
        records.append({
            "permutation": permutation,
            "recovered_class": result.recovered_topology.topology_class,
            "isomorphic_to_truth": result.is_isomorphic_to_truth,
            "prediction_invariant": same_prediction,
            "ambiguous": result.ambiguous,
        })
    observable_max_error = max(
        structural.state_max_error,
        structural.pairwise_max_error,
        structural.multipartite_max_error,
        structural.intervention_max_error,
        structural.memory_max_error,
        structural.global_scalar_max_error,
    )
    return ReconstructionRelabelingReport(len(records), observable_max_error, predictions_invariant, correctness_invariant, tuple(records))


@dataclass(frozen=True)
class ScheduleRobustnessReport:
    schedules_tested: int
    max_observable_distance: float
    recovery_accuracy: float
    prediction_invariant: bool
    per_schedule: tuple[dict[str, object], ...]


def schedule_robustness_report(
    instance: PhysicalProcessInstance,
    candidates: list[CausalTopology] | ReferenceLibrary,
    mode: str,
    scales: ComponentScales | None = None,
) -> ScheduleRobustnessReport:
    """Hold topology and nuisance parameters fixed; vary only valid DAG schedules."""
    if has_cycle(instance.topology):
        return ScheduleRobustnessReport(1, 0.0, 1.0, True, ({"schedule": instance.relation_schedule, "operational_cycle_pass": True},))
    orders = valid_topological_node_orders(instance.topology)
    bundles = []
    records = []
    predictions = []
    correct = []
    for node_order in orders:
        scheduled = reschedule_physical_process(instance, node_order)
        bundle = build_observable_bundle_from_instance(scheduled)
        result = _reconstruct(bundle, candidates, instance.topology, mode, scales)
        bundles.append(bundle)
        predictions.append(result.recovered_topology)
        correct.append(result.is_isomorphic_to_truth)
        records.append({
            "node_order": node_order,
            "relation_schedule": scheduled.relation_schedule,
            "recovered_class": result.recovered_topology.topology_class,
            "isomorphic_to_truth": result.is_isomorphic_to_truth,
        })
    distances = [combined_distance(bundles[first], bundles[second], mode, scales) for first in range(len(bundles)) for second in range(first + 1, len(bundles))]
    invariant = all(is_isomorphic(predictions[0], prediction) for prediction in predictions[1:]) if predictions else True
    return ScheduleRobustnessReport(len(orders), max(distances, default=0.0), float(np.mean(correct)), invariant, tuple(records))


@dataclass(frozen=True)
class NuisanceRobustnessReport:
    instances_tested: int
    recovery_accuracy: float
    per_instance: tuple[dict[str, object], ...]


def nuisance_robustness_report(
    topology: CausalTopology,
    candidates: list[CausalTopology] | ReferenceLibrary,
    seeds: tuple[int, ...],
    mode: str,
    scales: ComponentScales | None = None,
    retention_values: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5),
) -> NuisanceRobustnessReport:
    """Independently resample nuisance parameters; this is NOT label invariance."""
    node_order = None if has_cycle(topology) else valid_topological_node_orders(topology)[0]
    records = []
    for seed in seeds:
        for retention in retention_values:
            instance = create_physical_process_instance(topology, seed, node_order, retention=retention)
            result = _reconstruct(build_observable_bundle_from_instance(instance), candidates, topology, mode, scales)
            metrics = graph_recovery_metrics(result.recovered_topology, topology)
            records.append({
                "seed": seed,
                "retention": retention,
                "strengths": instance.relation_strengths,
                "initial_state_angles": instance.node_initial_state_angles,
                "noise_parameters": instance.node_noise_parameters,
                "schedule": instance.relation_schedule,
                "recovered_class": result.recovered_topology.topology_class,
                "isomorphic_to_truth": result.is_isomorphic_to_truth,
                "edge_precision": metrics.precision,
                "edge_recall": metrics.recall,
                "edge_f1": metrics.f1,
                "graph_edit_distance": metrics.graph_edit_distance,
            })
    accuracy = float(np.mean([record["isomorphic_to_truth"] for record in records])) if records else 0.0
    return NuisanceRobustnessReport(len(records), accuracy, tuple(records))


@dataclass(frozen=True)
class ConfusionMetrics:
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: dict[str, int]


def cycle_confusion_metrics(truth_and_prediction: list[tuple[str, str]]) -> ConfusionMetrics:
    tp = sum(truth == "CYCLIC" and prediction == "CYCLIC" for truth, prediction in truth_and_prediction)
    tn = sum(truth == "DAG" and prediction == "DAG" for truth, prediction in truth_and_prediction)
    fp = sum(truth == "DAG" and prediction == "CYCLIC" for truth, prediction in truth_and_prediction)
    fn = sum(truth == "CYCLIC" and prediction == "DAG" for truth, prediction in truth_and_prediction)
    total = tp + tn + fp + fn
    return ConfusionMetrics(
        (tp + tn) / total if total else 0.0,
        fp / (fp + tn) if fp + tn else 0.0,
        fn / (fn + tp) if fn + tp else 0.0,
        {"true_cyclic": tp, "true_dag": tn, "false_cyclic": fp, "false_dag": fn},
    )
