"""M5.1/M5.2 clean validation with frozen-instance relabeling and held-out seeds.

No ML, geometry, Lorentzian data, Q, or Delta_munu. This runner separates
same-process relabeling, schedule variation, and nuisance resampling.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from relational_dynamics.causal_topology import canonical_topology_key, graph_recovery_metrics, has_cycle, is_isomorphic, valid_topological_node_orders
from relational_dynamics.m5_validation import (
    OBSERVABLE_MODES,
    cycle_confusion_metrics,
    nuisance_robustness_report,
    reconstruction_relabeling_report,
    schedule_robustness_report,
)
from relational_dynamics.topology_generation import generate_ensemble
from relational_dynamics.topology_identifiability import run_identifiability_test
from relational_dynamics.topology_observables import (
    audit_directionality,
    build_observable_bundle_from_instance,
    create_physical_process_instance,
    relabel_physical_process,
    reschedule_physical_process,
    same_process_relabeling_report,
)
from relational_dynamics.topology_reconstruction import (
    CAUSAL_ONLY,
    CAUSAL_PLUS_MEMORY,
    TRAIN_SEEDS,
    build_reference_library,
    classify_cycle_vs_dag,
    reconstruct_with_library,
    shuffle_observable_bundle,
)

CLEAN_TEST_SEEDS = tuple(range(1000, 1010))
PRIMARY_MODE = CAUSAL_PLUS_MEMORY
TOLERANCE = 1e-8
TOPOLOGY_RECOVERY_PASS_THRESHOLD = 0.75
NUISANCE_ROBUSTNESS_PASS_THRESHOLD = 0.75
ENSEMBLE = generate_ensemble(edge_counts=(2, 3, 4), max_size=1000)


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _default_order(topology):
    return None if has_cycle(topology) else valid_topological_node_orders(topology)[0]


def _instance(topology, seed, retention=1.0):
    return create_physical_process_instance(topology, seed, _default_order(topology), retention=retention)


def _cycle_validation(library):
    dag_references = [bundle for topology in library.candidate_topologies if not has_cycle(topology) for bundle in library.bundles_by_key[str(canonical_topology_key(topology))]]
    cyclic_references = [bundle for topology in library.candidate_topologies if has_cycle(topology) for bundle in library.bundles_by_key[str(canonical_topology_key(topology))]]
    records = []
    truth_prediction = []
    for topology in ENSEMBLE:
        truth = "CYCLIC" if has_cycle(topology) else "DAG"
        for seed in CLEAN_TEST_SEEDS:
            instance = _instance(topology, seed)
            query = build_observable_bundle_from_instance(instance)
            result = classify_cycle_vs_dag(query, dag_references, cyclic_references, CAUSAL_ONLY, library.scales)
            truth_prediction.append((truth, result.predicted_label))
            records.append({
                "topology_key": canonical_topology_key(topology),
                "seed": seed,
                "schedule": instance.relation_schedule,
                "strengths": instance.relation_strengths,
                "truth": truth,
                "prediction": result.predicted_label,
                "dag_distance": result.dag_min_distance,
                "cyclic_distance": result.cyclic_min_distance,
            })
    return cycle_confusion_metrics(truth_prediction), records


def _cycle_controls(library):
    dag_references = [bundle for topology in library.candidate_topologies if not has_cycle(topology) for bundle in library.bundles_by_key[str(canonical_topology_key(topology))]]
    cyclic_references = [bundle for topology in library.candidate_topologies if has_cycle(topology) for bundle in library.bundles_by_key[str(canonical_topology_key(topology))]]
    truth_predictions = {name: [] for name in ("random_labels", "random_schedule", "random_strengths", "zero_coupling")}
    records = []
    for topology_index, topology in enumerate(ENSEMBLE):
        truth = "CYCLIC" if has_cycle(topology) else "DAG"
        for seed in CLEAN_TEST_SEEDS:
            instance = _instance(topology, seed)
            rng = np.random.default_rng(7000 + topology_index * 100 + seed)
            permutation = tuple(int(index) for index in rng.permutation(topology.nodes))
            variants = {"random_labels": relabel_physical_process(instance, permutation)}
            if has_cycle(topology):
                variants["random_schedule"] = replace(instance, relation_schedule=tuple(reversed(instance.relation_schedule)))
            else:
                orders = valid_topological_node_orders(topology)
                variants["random_schedule"] = reschedule_physical_process(instance, orders[int(rng.integers(0, len(orders)))])
            strength_values = list(instance.relation_strengths.values())
            rng.shuffle(strength_values)
            variants["random_strengths"] = replace(instance, relation_strengths=dict(zip(instance.relation_strengths, strength_values)))
            variants["zero_coupling"] = replace(instance, relation_strengths={edge: 0.0 for edge in instance.relation_strengths})
            for control, variant in variants.items():
                prediction = classify_cycle_vs_dag(
                    build_observable_bundle_from_instance(variant), dag_references, cyclic_references, CAUSAL_ONLY, library.scales
                ).predicted_label
                truth_predictions[control].append((truth, prediction))
                records.append({"control": control, "topology_key": canonical_topology_key(topology), "seed": seed, "truth": truth, "prediction": prediction})
    return {control: cycle_confusion_metrics(values) for control, values in truth_predictions.items()}, records


def _exact_recovery(library):
    queries = []
    for topology in ENSEMBLE:
        for seed in CLEAN_TEST_SEEDS:
            instance = _instance(topology, seed)
            queries.append((topology, seed, instance, build_observable_bundle_from_instance(instance)))
    output = {}
    for mode in OBSERVABLE_MODES:
        records = []
        for topology, seed, instance, query_bundle in queries:
            result = reconstruct_with_library(query_bundle, library, topology, mode)
            metrics = graph_recovery_metrics(result.recovered_topology, topology)
            records.append({
                "topology_key": canonical_topology_key(topology),
                "seed": seed,
                "schedule": instance.relation_schedule,
                "strengths": instance.relation_strengths,
                "initial_state_angles": instance.node_initial_state_angles,
                "noise_parameters": instance.node_noise_parameters,
                "retention": instance.retention,
                "recovered_key": canonical_topology_key(result.recovered_topology),
                "correct_up_to_isomorphism": result.is_isomorphic_to_truth,
                "distance": result.recovered_distance,
                "ambiguous": result.ambiguous,
                "edge_precision": metrics.precision,
                "edge_recall": metrics.recall,
                "edge_f1": metrics.f1,
                "graph_edit_distance": metrics.graph_edit_distance,
            })
        per_topology = {}
        confusion_matrix = {}
        for topology in ENSEMBLE:
            key = str(canonical_topology_key(topology))
            relevant = [record for record in records if str(tuple(record["topology_key"])) == key]
            per_topology[key] = float(np.mean([record["correct_up_to_isomorphism"] for record in relevant]))
        for record in records:
            truth = str(tuple(record["topology_key"]))
            predicted = str(tuple(record["recovered_key"]))
            confusion_matrix.setdefault(truth, {})[predicted] = confusion_matrix.setdefault(truth, {}).get(predicted, 0) + 1
        output[mode] = {
            "accuracy": float(np.mean([record["correct_up_to_isomorphism"] for record in records])),
            "per_topology_accuracy": per_topology,
            "confusion_matrix": confusion_matrix,
            "records": records,
        }
    return output


def _same_process_relabeling(library):
    structural = []
    reconstruction = []
    for topology in ENSEMBLE:
        instance = _instance(topology, CLEAN_TEST_SEEDS[0])
        structural_report = same_process_relabeling_report(instance, TOLERANCE)
        reconstruction_report = reconstruction_relabeling_report(instance, library, PRIMARY_MODE, library.scales)
        structural.append({"topology_key": canonical_topology_key(topology), "report": structural_report})
        reconstruction.append({"topology_key": canonical_topology_key(topology), "report": reconstruction_report})
    return structural, reconstruction


def _schedule_validation(library):
    records = []
    for topology in ENSEMBLE:
        if has_cycle(topology):
            continue
        orders = valid_topological_node_orders(topology)
        if len(orders) < 2:
            continue
        instance = _instance(topology, CLEAN_TEST_SEEDS[0])
        report = schedule_robustness_report(instance, library, PRIMARY_MODE, library.scales)
        records.append({"topology_key": canonical_topology_key(topology), "report": report})
    return records


def _schedule_ablation(library):
    summaries = {mode: [] for mode in OBSERVABLE_MODES}
    for topology in ENSEMBLE:
        if has_cycle(topology):
            continue
        orders = valid_topological_node_orders(topology)
        if len(orders) < 2:
            continue
        instance = _instance(topology, CLEAN_TEST_SEEDS[0])
        bundles = [build_observable_bundle_from_instance(reschedule_physical_process(instance, order)) for order in orders]
        for mode in OBSERVABLE_MODES:
            predictions = [reconstruct_with_library(bundle, library, topology, mode) for bundle in bundles]
            summaries[mode].append({
                "topology_key": canonical_topology_key(topology),
                "schedules_tested": len(orders),
                "prediction_invariant": all(is_isomorphic(predictions[0].recovered_topology, prediction.recovered_topology) for prediction in predictions[1:]),
                "recovery_accuracy": float(np.mean([prediction.is_isomorphic_to_truth for prediction in predictions])),
            })
    return {
        mode: {
            "topologies_tested": len(records),
            "prediction_invariance_rate": float(np.mean([record["prediction_invariant"] for record in records])) if records else 1.0,
            "recovery_accuracy": float(np.mean([record["recovery_accuracy"] for record in records])) if records else 1.0,
            "records": records,
        }
        for mode, records in summaries.items()
    }


def _nuisance_validation(library):
    return [
        {
            "topology_key": canonical_topology_key(topology),
            "report": nuisance_robustness_report(
                topology,
                library,
                CLEAN_TEST_SEEDS[:5],
                PRIMARY_MODE,
                library.scales,
            ),
        }
        for topology in ENSEMBLE
    ]


def _negative_controls(library):
    records = []
    for topology in ENSEMBLE:
        instance = _instance(topology, CLEAN_TEST_SEEDS[0])
        clean_bundle = build_observable_bundle_from_instance(instance)
        clean = reconstruct_with_library(clean_bundle, library, topology, PRIMARY_MODE)
        shuffled = reconstruct_with_library(shuffle_observable_bundle(clean_bundle, CLEAN_TEST_SEEDS[0]), library, topology, PRIMARY_MODE)
        zero_instance = replace(instance, relation_strengths={edge: 0.0 for edge in instance.relation_strengths})
        zero = reconstruct_with_library(build_observable_bundle_from_instance(zero_instance), library, topology, PRIMARY_MODE)
        records.append({
            "topology_key": canonical_topology_key(topology),
            "clean_correct": clean.is_isomorphic_to_truth,
            "shuffled_observable_correct": shuffled.is_isomorphic_to_truth,
            "zero_coupling_correct": zero.is_isomorphic_to_truth,
        })
    return {
        "records": records,
        "clean_accuracy": float(np.mean([record["clean_correct"] for record in records])),
        "shuffled_observable_accuracy": float(np.mean([record["shuffled_observable_correct"] for record in records])),
        "zero_coupling_accuracy": float(np.mean([record["zero_coupling_correct"] for record in records])),
    }


def _identifiability_audit():
    jobs = []
    for first in range(len(ENSEMBLE)):
        for second in range(first + 1, len(ENSEMBLE)):
            jobs.append((ENSEMBLE[first], ENSEMBLE[second], 5000 + first * len(ENSEMBLE) + second))
    with ProcessPoolExecutor(max_workers=4) as executor:
        return list(executor.map(_identifiability_job, jobs))


def _identifiability_job(job):
    first, second, seed = job
    return run_identifiability_test(first, second, tolerance=0.05, trials=10, seed=seed, mode=PRIMARY_MODE)


def main():
    existing_suite_pass = "--existing-suite-passed" in sys.argv
    if set(TRAIN_SEEDS) & set(CLEAN_TEST_SEEDS):
        raise RuntimeError("TRAIN_TEST_LEAKAGE: clean test seeds overlap reference seeds")
    started = time.perf_counter()
    library = build_reference_library(ENSEMBLE, TRAIN_SEEDS)
    cycle_metrics, cycle_records = _cycle_validation(library)
    cycle_controls, cycle_control_records = _cycle_controls(library)
    exact_recovery = _exact_recovery(library)
    structural_relabeling, reconstruction_relabeling = _same_process_relabeling(library)
    schedule = _schedule_validation(library)
    schedule_ablation = _schedule_ablation(library)
    nuisance = _nuisance_validation(library)
    controls = _negative_controls(library)
    identifiability = _identifiability_audit()

    process_relabeling_pass = all(entry["report"].passed for entry in structural_relabeling)
    label_invariance_pass = all(entry["report"].prediction_invariant and entry["report"].truth_correctness_invariant for entry in reconstruction_relabeling)
    schedule_pass = all(entry["report"].prediction_invariant for entry in schedule)
    nuisance_accuracy = float(np.mean([entry["report"].recovery_accuracy for entry in nuisance]))
    topology_accuracy = exact_recovery[PRIMARY_MODE]["accuracy"]
    identifiability_complete = len(identifiability) == len(ENSEMBLE) * (len(ENSEMBLE) - 1) // 2
    probe_bundle = build_observable_bundle_from_instance(_instance(ENSEMBLE[0], CLEAN_TEST_SEEDS[0]))
    causal_leakage_pass = set(probe_bundle.__dataclass_fields__) == {
        "pairwise_information", "multipartite_information", "intervention_tensor", "memory_score", "node_count"
    }
    gates = {
        "M5_PROCESS_RELABELING_PASS": process_relabeling_pass,
        "M5_LABEL_INVARIANCE_PASS": label_invariance_pass,
        "M5_TRAIN_TEST_LEAKAGE_PASS": True,
        "M5_CAUSAL_LEAKAGE_PASS": causal_leakage_pass,
        "M5_SCHEDULE_ROBUSTNESS_PASS": schedule_pass,
        "M5_NUISANCE_ROBUSTNESS_PASS": nuisance_accuracy >= NUISANCE_ROBUSTNESS_PASS_THRESHOLD,
        "M5_TOPOLOGY_RECOVERY_PASS": topology_accuracy >= TOPOLOGY_RECOVERY_PASS_THRESHOLD,
        "M5_IDENTIFIABILITY_AUDIT_PASS": identifiability_complete,
        "EXISTING_TEST_SUITE_PASS": existing_suite_pass,
    }
    classification = "PASS" if all(gates.values()) else "IMPLEMENTED_BUT_NOT_FULLY_VALIDATED"
    result = {
        "milestone": "M5_CLEAN_VALIDATION",
        "classification": classification,
        "gates": gates,
        "directionality_audit": audit_directionality(),
        "topology_ensemble": [{"canonical_key": canonical_topology_key(topology), "edges": topology.directed_relations, "cyclic": has_cycle(topology)} for topology in ENSEMBLE],
        "parameter_domain": {"strength_range": [0.35, 0.75], "initial_state_angle_range": [0.0, float(2 * np.pi)], "noise_range": [0.0, 0.1], "retention_values": [1.0, 0.9, 0.75, 0.5]},
        "train_seeds": TRAIN_SEEDS,
        "clean_test_seeds": CLEAN_TEST_SEEDS,
        "primary_mode_preregistered": PRIMARY_MODE,
        "pass_thresholds": {
            "topology_recovery_accuracy": TOPOLOGY_RECOVERY_PASS_THRESHOLD,
            "nuisance_robustness_accuracy": NUISANCE_ROBUSTNESS_PASS_THRESHOLD,
        },
        "m51_cycle_metrics": cycle_metrics,
        "m51_records": cycle_records,
        "m51_control_metrics": cycle_controls,
        "m51_control_records": cycle_control_records,
        "m52_exact_recovery": exact_recovery,
        "same_process_structural_relabeling": structural_relabeling,
        "same_process_reconstruction_relabeling": reconstruction_relabeling,
        "schedule_robustness": schedule,
        "schedule_ablation": schedule_ablation,
        "nuisance_robustness": nuisance,
        "negative_controls": controls,
        "identifiability": identifiability,
        "runtime_seconds": time.perf_counter() - started,
        "q_norm": 0.0,
        "delta_norm": 0.0,
        "claims_new_physics": False,
        "claims_physical_ctc": False,
    }
    output_path = ROOT / "results" / "historical" / "m5_clean_validation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps({"classification": classification, "gates": gates, "cycle_metrics": _jsonable(cycle_metrics), "topology_accuracy": topology_accuracy, "nuisance_accuracy": nuisance_accuracy, "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
