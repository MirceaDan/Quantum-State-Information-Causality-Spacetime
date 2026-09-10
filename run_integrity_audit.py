"""Run the Reverse-Theseus EXPERIMENTAL INTEGRITY / IDENTIFIABILITY AUDIT.

This is the follow-up phase to run_historical_reconstruction.py. It does NOT
tune the coupling law, model capacity, or hyperparameters; it audits whether
the previously observed positive signal survives after eliminating
train/test leakage and trivial causal-order/label confounds, and adds the
required control matrix, observable-ablation attribution, N-scaling study,
and intervention-family robustness check.

The classification logic in ``classify_historical_reconstruction`` is fixed
BEFORE this script is run against the final experiment matrix, per section 9
of the integrity-audit specification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from relational_dynamics.ground_truth import build_ground_truth_worlds
from relational_dynamics.historical_reconstruction import (
    classify_historical_reconstruction,
    compute_scientific_gate,
    run_identifiability_test,
    run_information_retention_curve,
    run_intervention_family_robustness,
    run_label_permutation_control,
    run_observable_ablation,
    run_process_variant_experiment,
    run_randomized_control_experiment,
    run_scaling_study,
)
from relational_dynamics.historical_process import build_geometry_coupled_history
from relational_dynamics.historical_reconstruction import build_historical_observables
from relational_dynamics.historical_scaling import FEASIBLE_SCALING_EVENT_COUNTS, SCALING_EVENT_COUNTS

SEEDS = tuple(range(10))


def _to_jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    worlds = build_ground_truth_worlds()
    target_world = worlds[1]  # WORLD_2, the curved world used for the primary audit

    print("=== Section 1/2: control matrix ===")
    coupled = run_process_variant_experiment(target_world, "COUPLED", SEEDS)
    zero_coupling = run_process_variant_experiment(target_world, "ZERO_COUPLING", SEEDS)
    shuffled_geometry = run_process_variant_experiment(target_world, "SHUFFLED_GEOMETRY", SEEDS)
    randomized_order = run_process_variant_experiment(target_world, "RANDOMIZED_ORDER", SEEDS)
    randomized_observable = run_randomized_control_experiment(target_world, SEEDS)
    for label, report in (
        ("COUPLED", coupled), ("ZERO_COUPLING", zero_coupling), ("SHUFFLED_GEOMETRY", shuffled_geometry),
        ("RANDOMIZED_ORDER", randomized_order), ("RANDOMIZED_OBSERVABLE", randomized_observable),
    ):
        stats = report.seed_statistics["FLEXIBLE"]
        print(f"  {label}: FLEXIBLE test_loss mean={stats.mean:.4f} std={stats.std:.4f} median={stats.median:.4f} min={stats.minimum:.4f} max={stats.maximum:.4f}")

    print("=== Section 3: observable attribution ablation ===")
    process = build_geometry_coupled_history(target_world, retention=1.0)
    observables = build_historical_observables(process, target_world)
    ablation = run_observable_ablation(target_world, observables, seed=1234)
    for mode, report in ablation.items():
        print(f"  {mode}: test_loss={report.test_loss:.4f} causal_f1={report.causal_f1:.3f} curvature_error={report.curvature_invariant_error:.4f}")

    print("=== Section 4: identifiability (weak, observable-distance only) ===")
    identifiability = []
    for first in range(len(worlds)):
        for second in range(first + 1, len(worlds)):
            report = run_identifiability_test(worlds[first], worlds[second])
            identifiability.append(report)
            print(f"  {report.world_a} vs {report.world_b}: distance={report.observable_distance:.6f} -> {report.classification}")

    print("=== Section 5: label/order confound ===")
    label_report = run_label_permutation_control(target_world, SEEDS)
    print(f"  permutation={label_report.permutation} original_mean={label_report.original_test_loss_mean:.4f} relabeled_mean={label_report.relabeled_test_loss_mean:.4f} -> {label_report.classification}")

    print("=== Section 6: N-scaling study (feasible sizes only; see LIMITATIONS.md) ===")
    scaling_points = run_scaling_study(FEASIBLE_SCALING_EVENT_COUNTS, SEEDS)
    for point in scaling_points:
        print(f"  N={point.event_count} {point.world_id}: test_loss mean={point.test_loss.mean:.4f} std={point.test_loss.std:.4f} coord_err={point.coordinate_error_mean:.4f} causal_f1={point.causal_f1_mean:.3f}")
    skipped_sizes = [n for n in SCALING_EVENT_COUNTS if n not in FEASIBLE_SCALING_EVENT_COUNTS]

    print("=== Section 7: intervention-family robustness (X, Z) ===")
    intervention_reports = run_intervention_family_robustness(target_world, SEEDS)
    for label, report in intervention_reports.items():
        stats = report.seed_statistics["FLEXIBLE"]
        print(f"  {label}: FLEXIBLE test_loss mean={stats.mean:.4f} std={stats.std:.4f}")

    print("=== Section 8: retention curve (control only) ===")
    retention_curve = run_information_retention_curve(target_world, seed=1234)
    print(f"  eta: {retention_curve.etas}")
    print(f"  F_geom (coupled): {retention_curve.scores}")
    print(f"  F_geom (zero-coupling control): {retention_curve.zero_coupling_scores}")

    print("=== Section 11: scientific gate ===")
    gate = compute_scientific_gate(
        target_world,
        label_report,
        identifiability,
        scaling_study_present=True,
        intervention_robustness_present=True,
    )
    print(f"  train_test_leakage={gate.train_test_leakage}")
    print(f"  causal_leakage={gate.causal_leakage}")
    print(f"  order_confound={gate.order_confound}")
    print(f"  label_confound={gate.label_confound}")
    print(f"  geometry_leakage={gate.geometry_leakage}")
    print(f"  scaling_study_present={gate.scaling_study_present}")
    print(f"  intervention_robustness_present={gate.intervention_robustness_present}")

    classification = classify_historical_reconstruction(gate, coupled, zero_coupling, shuffled_geometry, randomized_order, randomized_observable)
    print(f"=== FINAL CLASSIFICATION: {classification} ===")

    output = {
        "gate": _to_jsonable(gate),
        "classification": classification,
        "q_norm": 0.0,
        "delta_norm": 0.0,
        "claims_new_physics": False,
        "claims_time_travel": False,
        "control_matrix": {
            "COUPLED": _to_jsonable(coupled),
            "ZERO_COUPLING": _to_jsonable(zero_coupling),
            "SHUFFLED_GEOMETRY": _to_jsonable(shuffled_geometry),
            "RANDOMIZED_ORDER": _to_jsonable(randomized_order),
            "RANDOMIZED_OBSERVABLE": _to_jsonable(randomized_observable),
        },
        "observable_ablation": _to_jsonable(ablation),
        "identifiability": _to_jsonable(identifiability),
        "label_permutation_control": _to_jsonable(label_report),
        "scaling_study": {
            "feasible_event_counts": FEASIBLE_SCALING_EVENT_COUNTS,
            "skipped_event_counts": skipped_sizes,
            "skipped_reason": "dense qubit simulation cost scales as 4**event_count; not executed in this reference phase",
            "points": _to_jsonable(scaling_points),
        },
        "intervention_family_robustness": _to_jsonable(intervention_reports),
        "retention_curve": _to_jsonable(retention_curve),
    }
    output_root = ROOT / "results" / "historical"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "integrity_audit_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
