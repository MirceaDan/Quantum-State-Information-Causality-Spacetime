"""Root-cause analysis of the order/label confound found by the integrity audit.

DIAGNOSIS ONLY: does not change the coupling law, synthetic geometries, Q,
Delta_munu, or optimizer capacity. See historical_diagnostics.py.
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
from relational_dynamics.historical_diagnostics import (
    analyze_non_geometric_predictability,
    compare_observables_under_relabeling,
    run_non_geometric_controls,
    run_root_cause_analysis,
)

SEEDS = tuple(range(5))


def _to_jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main() -> None:
    world = build_ground_truth_worlds()[1]  # WORLD_2, where the confound was found
    permutation = (2, 0, 1, 3)

    print("=== 1-3: observable invariance under a pure relabeling ===")
    invariance = compare_observables_under_relabeling(world, permutation)
    print(f"  permutation={invariance.permutation}")
    print(f"  coupling_graph_preserved={invariance.coupling_graph_preserved}")
    print(f"  original_edges={list(invariance.original_edges)}")
    print(f"  relabeled_edges(original identity)={list(invariance.relabeled_edges)}")
    print(f"  shared_edges={list(invariance.shared_edges)}  angles_match_on_shared_edges={invariance.angles_on_shared_edges_match}")
    print(f"  pairwise_information_changed={invariance.pairwise_information_changed} (max abs diff {invariance.pairwise_information_max_abs_diff:.4f})")
    print(f"  multipartite_information_changed={invariance.multipartite_information_changed} (max abs diff {invariance.multipartite_information_max_abs_diff:.4f})")
    print(f"  causal_estimate_changed={invariance.causal_estimate_changed} (hamming {invariance.causal_estimate_hamming_distance})")
    print(f"  intervention_tensor_changed={invariance.intervention_tensor_changed} (max abs diff {invariance.intervention_tensor_max_abs_diff:.4f})")

    print("\n=== 4/6: is loss predictable from non-geometric features alone? ===")
    predictability = analyze_non_geometric_predictability(world, SEEDS)
    print(f"  permutations enumerated: {predictability.permutation_count}")
    print(f"  mean loss | adjacency-preserving permutations: {predictability.adjacency_preserving_loss_mean:.4f}")
    print(f"  mean loss | adjacency-breaking permutations:   {predictability.adjacency_breaking_loss_mean:.4f}")
    print(f"  adjacency effect ratio (breaking/preserving):  {predictability.adjacency_effect_ratio:.4f}")
    print(f"  correlation(loss, index-displacement):         {predictability.displacement_loss_correlation:.4f}")
    print(f"  loss variance explained by adjacency grouping: {predictability.loss_variance_explained_by_adjacency:.4f}")

    print("\n=== 5/6: explicit non-geometric controls A-D ===")
    controls = run_non_geometric_controls(world, SEEDS)
    print(f"  COUPLED (baseline):                          {controls.coupled_test_loss_mean:.4f}")
    print(f"  A. randomized event labels (same values):    {controls.control_a_randomized_labels_test_loss_mean:.4f}")
    print(f"  B. same causal graph, random strengths:      {controls.control_b_randomized_strengths_test_loss_mean:.4f}")
    print(f"  C. same strengths, randomized order:         {controls.control_c_randomized_order_test_loss_mean:.4f}")
    print(f"  D. same geometry, randomized indexing:       {controls.control_d_randomized_indexing_test_loss_mean:.4f}")

    print("\n=== 10: root-cause classification ===")
    root_cause = run_root_cause_analysis(world, SEEDS, permutation=permutation)
    print(f"  process_structure_confund = {root_cause.process_structure_confund}")
    print(f"  label_confund             = {root_cause.label_confund}")
    print(f"  order_confund             = {root_cause.order_confund}")
    print(f"  split_design_confund      = {root_cause.split_design_confund}")
    print(f"  CLASSIFICATION            = {root_cause.classification}")
    print("\n  Scientific classification remains EXPERIMENTAL_CONFOUND_PRESENT (unchanged) until this is resolved.")

    output = {
        "invariance_report": _to_jsonable(invariance),
        "predictability_report": _to_jsonable(predictability),
        "controls_report": _to_jsonable(controls),
        "root_cause_report": {
            "process_structure_confund": root_cause.process_structure_confund,
            "label_confund": root_cause.label_confund,
            "order_confund": root_cause.order_confund,
            "split_design_confund": root_cause.split_design_confund,
            "classification": root_cause.classification,
        },
        "scientific_classification_unchanged": "EXPERIMENTAL_CONFOUND_PRESENT",
        "q_norm": 0.0,
        "delta_norm": 0.0,
    }
    result_root = ROOT / "results" / "historical"
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "root_cause_analysis_results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
