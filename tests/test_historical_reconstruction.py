import numpy as np
import pytest

from relational_dynamics.ground_truth import build_ground_truth_worlds
from relational_dynamics.historical_process import build_geometry_coupled_history
from relational_dynamics.historical_reconstruction import (
    ALL_OBSERVABLES,
    CAUSAL_ONLY,
    FLEXIBLE,
    MI_ONLY,
    MI_PLUS_CAUSAL,
    MULTIPARTITE_ONLY,
    NULL,
    RESTRICTED,
    ObservableSplit,
    assert_causal_subset,
    build_historical_observables,
    build_observable_split,
    build_relabeled_world,
    classify_historical_reconstruction,
    compare_historical_baselines,
    compute_scientific_gate,
    randomize_historical_observables,
    run_coupled_experiment,
    run_identifiability_test,
    run_information_retention_curve,
    run_intervention_family_robustness,
    run_label_permutation_control,
    run_null_baseline_experiment,
    run_observable_ablation,
    run_process_variant_experiment,
    run_randomized_control_experiment,
    run_scaling_study,
)
from relational_dynamics.historical_scaling import FEASIBLE_SCALING_EVENT_COUNTS, build_scaled_worlds


WORLDS = build_ground_truth_worlds()
SMALL_SEEDS = tuple(range(3))


# ---------------------------------------------------------------------------
# Section 1: train/validation/test leakage
# ---------------------------------------------------------------------------


def test_observable_split_pairs_are_disjoint():
    process = build_geometry_coupled_history(WORLDS[1], retention=1.0)
    observables = build_historical_observables(process, WORLDS[1])
    split = build_observable_split(observables, seed=7)
    assert isinstance(split, ObservableSplit)
    train, validation, test = set(split.train_pairs), set(split.validation_pairs), set(split.test_pairs)
    assert not (train & validation)
    assert not (train & test)
    assert not (validation & test)
    assert train | validation | test == {(first, second) for first in range(4) for second in range(first + 1, 4)}


def test_train_causal_matrix_never_contains_validation_or_test_pairs():
    process = build_geometry_coupled_history(WORLDS[1], retention=1.0)
    observables = build_historical_observables(process, WORLDS[1])
    split = build_observable_split(observables, seed=7)
    assert_causal_subset(split.train_causal, split.train_pairs)
    with pytest.raises(ValueError):
        assert_causal_subset(observables.causal_estimate, split.train_pairs)  # full matrix must NOT be a training-only subset in general


def test_assert_causal_subset_rejects_validation_and_test_leakage():
    process = build_geometry_coupled_history(WORLDS[1], retention=1.0)
    observables = build_historical_observables(process, WORLDS[1])
    split = build_observable_split(observables, seed=7)
    if np.any(split.validation_causal):
        with pytest.raises(ValueError):
            assert_causal_subset(split.validation_causal, split.train_pairs)
    if np.any(split.test_causal):
        with pytest.raises(ValueError):
            assert_causal_subset(split.test_causal, split.train_pairs)


def test_triples_are_only_assigned_to_train_when_all_subpairs_are_training():
    process = build_geometry_coupled_history(WORLDS[1], retention=1.0)
    observables = build_historical_observables(process, WORLDS[1])
    split = build_observable_split(observables, seed=7)
    train_set = set(split.train_pairs)
    for first, second, third in split.train_triples:
        assert {(first, second), (first, third), (second, third)} <= train_set


def test_fitting_never_raises_a_leakage_error_end_to_end():
    process = build_geometry_coupled_history(WORLDS[1], retention=1.0)
    observables = build_historical_observables(process, WORLDS[1])
    for seed in SMALL_SEEDS:
        compare_historical_baselines(WORLDS[1], observables, seed)  # must not raise


# ---------------------------------------------------------------------------
# Observable construction and baselines
# ---------------------------------------------------------------------------


def test_build_historical_observables_never_exposes_coordinates_or_metric():
    process = build_geometry_coupled_history(WORLDS[0], retention=1.0)
    observables = build_historical_observables(process, WORLDS[0])
    assert observables.pairwise_information.shape == (4, 4)
    assert observables.intervention_tensor.shape[:2] == (4, 4)
    assert len(observables.multipartite_information) == 4
    for field_name in ("coordinates", "metric", "g_munu", "world_id", "t0"):
        assert not hasattr(observables, field_name)


def test_null_baseline_experiment_preserves_existing_behavior():
    report = run_null_baseline_experiment(seed=1234, seeds=SMALL_SEEDS)
    assert report.experiment_id == "EXP-HIST-NULL-001"
    assert report.report.observation_source == "independent_quantum_markov_process"
    assert report.multi_seed.seed_count == 3


def test_three_baselines_share_train_validation_test_split():
    process = build_geometry_coupled_history(WORLDS[1], retention=1.0)
    observables = build_historical_observables(process, WORLDS[1])
    baselines = compare_historical_baselines(WORLDS[1], observables, seed=7)
    assert set(baselines) == {NULL, RESTRICTED, FLEXIBLE}
    assert baselines[NULL].parameter_count == 0
    assert baselines[RESTRICTED].parameter_count == 16
    assert baselines[FLEXIBLE].parameter_count == 17
    for report in baselines.values():
        assert report.train_loss >= 0.0
        assert report.test_loss >= 0.0
        assert 0.0 <= report.causal_f1 <= 1.0


def test_randomized_control_preserves_marginal_statistics_but_breaks_pairing():
    process = build_geometry_coupled_history(WORLDS[1], retention=1.0)
    observables = build_historical_observables(process, WORLDS[1])
    randomized = randomize_historical_observables(observables, seed=3)
    upper_original = np.sort(observables.pairwise_information[np.triu_indices(4, k=1)])
    upper_random = np.sort(randomized.pairwise_information[np.triu_indices(4, k=1)])
    assert np.allclose(upper_original, upper_random)
    assert not np.array_equal(observables.pairwise_information, randomized.pairwise_information)


# ---------------------------------------------------------------------------
# Section 2: process-architecture controls
# ---------------------------------------------------------------------------


def test_zero_coupling_control_uses_zero_angles():
    report = run_process_variant_experiment(WORLDS[1], "ZERO_COUPLING", seeds=SMALL_SEEDS)
    assert report.experiment_id == "EXP-HIST-ZEROCOUPLING-001"


def test_shuffled_geometry_and_randomized_order_controls_run_the_same_pipeline():
    coupled = run_process_variant_experiment(WORLDS[1], "COUPLED", seeds=SMALL_SEEDS)
    shuffled = run_process_variant_experiment(WORLDS[1], "SHUFFLED_GEOMETRY", seeds=SMALL_SEEDS)
    randomized_order = run_process_variant_experiment(WORLDS[1], "RANDOMIZED_ORDER", seeds=SMALL_SEEDS)
    assert set(coupled.baselines) == set(shuffled.baselines) == set(randomized_order.baselines)
    assert shuffled.experiment_id == "EXP-HIST-SHUFFLEDGEOM-001"
    assert randomized_order.experiment_id == "EXP-HIST-RANDORDER-001"


def test_unknown_control_variant_is_rejected():
    with pytest.raises(ValueError):
        run_process_variant_experiment(WORLDS[1], "NOT_A_REAL_CONTROL", seeds=SMALL_SEEDS)


# ---------------------------------------------------------------------------
# Section 3: observable attribution ablation
# ---------------------------------------------------------------------------


def test_observable_ablation_covers_all_five_modes_with_identical_split():
    process = build_geometry_coupled_history(WORLDS[1], retention=1.0)
    observables = build_historical_observables(process, WORLDS[1])
    ablation = run_observable_ablation(WORLDS[1], observables, seed=7)
    assert set(ablation) == {MI_ONLY, CAUSAL_ONLY, MI_PLUS_CAUSAL, MULTIPARTITE_ONLY, ALL_OBSERVABLES}
    for mode, report in ablation.items():
        assert report.observable_mode == mode
        assert report.parameter_count == ablation[MI_ONLY].parameter_count  # same FLEXIBLE capacity throughout


# ---------------------------------------------------------------------------
# Section 4: identifiability terminology
# ---------------------------------------------------------------------------


def test_identifiability_uses_the_weak_observably_distinct_terminology():
    report = run_identifiability_test(WORLDS[0], WORLDS[1])
    assert report.classification in {"OBSERVABLY_DISTINCT", "OBSERVABLY_INDISTINGUISHABLE"}
    assert report.stronger_test_available is False
    assert "does NOT establish" in report.note


def test_identical_world_is_observably_indistinguishable_from_itself():
    report = run_identifiability_test(WORLDS[0], WORLDS[0])
    assert report.classification == "OBSERVABLY_INDISTINGUISHABLE"
    assert report.observable_distance == 0.0


# ---------------------------------------------------------------------------
# Section 5: label/order confound
# ---------------------------------------------------------------------------


def test_relabeled_world_preserves_interval_multiset():
    permutation = (2, 0, 3, 1)
    relabeled = build_relabeled_world(WORLDS[0], permutation)
    assert np.allclose(np.sort(np.abs(relabeled.invariant_intervals).ravel()), np.sort(np.abs(WORLDS[0].invariant_intervals).ravel()))
    assert relabeled.world_id == f"{WORLDS[0].world_id}_RELABELED"


def test_label_permutation_control_reports_a_classification():
    report = run_label_permutation_control(WORLDS[1], seeds=SMALL_SEEDS)
    assert report.classification in {"NO_LABEL_CONFOUND_DETECTED", "LABEL_OR_ORDER_CONFOUND_DETECTED"}


# ---------------------------------------------------------------------------
# Section 7: intervention-family robustness
# ---------------------------------------------------------------------------


def test_intervention_family_robustness_covers_x_and_z():
    reports = run_intervention_family_robustness(WORLDS[1], seeds=SMALL_SEEDS)
    assert set(reports) == {"X", "Z"}
    assert reports["X"].experiment_id == "EXP-HIST-INTERVENTION-X-001"
    assert reports["Z"].experiment_id == "EXP-HIST-INTERVENTION-Z-001"


# ---------------------------------------------------------------------------
# Section 6: N-scaling study
# ---------------------------------------------------------------------------


def test_scaled_worlds_reuse_the_same_three_geometry_families():
    worlds = build_scaled_worlds(6)
    assert [world.world_id for world in worlds] == ["WORLD_1_N6", "WORLD_2_N6", "WORLD_3_N6"]
    assert worlds[0].coordinates.shape == (6, 4)


def test_scaling_study_runs_for_feasible_event_counts():
    points = run_scaling_study(event_counts=(4,), seeds=SMALL_SEEDS)
    assert len(points) == 3  # one per geometry family at N=4
    for point in points:
        assert point.event_count == 4
        assert point.test_loss.seed_count == len(SMALL_SEEDS)


# ---------------------------------------------------------------------------
# Retention curve is a control only
# ---------------------------------------------------------------------------


def test_retention_curve_has_one_point_per_eta_and_includes_zero_coupling_control():
    etas = (0.0, 0.5, 1.0)
    curve = run_information_retention_curve(WORLDS[1], etas=etas, seed=1234)
    assert curve.etas == etas
    assert len(curve.zero_coupling_points) == len(etas)
    for score in curve.scores + curve.zero_coupling_scores:
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Scientific gate and predefined classification vocabulary
# ---------------------------------------------------------------------------


ALLOWED_CLASSIFICATIONS = {
    "NO_GEOMETRY_SIGNAL_DETECTED",
    "OBSERVABLY_DISTINCT_BUT_NOT_IDENTIFIABLE",
    "GEOMETRY_DEPENDENT_SIGNAL_DETECTED",
    "OPTIMIZATION_INSUFFICIENT",
    "REPRESENTATION_INSUFFICIENT",
    "INFORMATION_INSUFFICIENT",
    "EXPERIMENTAL_CONFOUND_PRESENT",
}


def test_scientific_gate_reports_pass_for_a_leakage_free_pipeline():
    label_report = run_label_permutation_control(WORLDS[1], seeds=SMALL_SEEDS)
    identifiability = [run_identifiability_test(WORLDS[0], WORLDS[1])]
    gate = compute_scientific_gate(WORLDS[1], label_report, identifiability, scaling_study_present=True, intervention_robustness_present=True)
    assert gate.train_test_leakage == "PASS"
    assert gate.causal_leakage == "PASS"
    assert gate.critical_checks_pass is True


def test_classification_uses_only_the_predefined_allowed_vocabulary():
    world = WORLDS[1]
    coupled = run_process_variant_experiment(world, "COUPLED", seeds=SMALL_SEEDS)
    zero_coupling = run_process_variant_experiment(world, "ZERO_COUPLING", seeds=SMALL_SEEDS)
    shuffled_geometry = run_process_variant_experiment(world, "SHUFFLED_GEOMETRY", seeds=SMALL_SEEDS)
    randomized_order = run_process_variant_experiment(world, "RANDOMIZED_ORDER", seeds=SMALL_SEEDS)
    randomized_observable = run_randomized_control_experiment(world, seeds=SMALL_SEEDS)
    label_report = run_label_permutation_control(world, seeds=SMALL_SEEDS)
    identifiability = [run_identifiability_test(WORLDS[0], WORLDS[1])]
    gate = compute_scientific_gate(world, label_report, identifiability, scaling_study_present=True, intervention_robustness_present=True)

    classification = classify_historical_reconstruction(gate, coupled, zero_coupling, shuffled_geometry, randomized_order, randomized_observable)
    assert classification in ALLOWED_CLASSIFICATIONS


def test_failed_gate_forces_experimental_confound_present():
    from relational_dynamics.historical_reconstruction import ScientificGate

    failing_gate = ScientificGate("FAIL", "PASS", "PASS", "PASS", "PASS", "n/a", "PASS", "PASS")
    world = WORLDS[1]
    coupled = run_process_variant_experiment(world, "COUPLED", seeds=SMALL_SEEDS)
    zero_coupling = run_process_variant_experiment(world, "ZERO_COUPLING", seeds=SMALL_SEEDS)
    shuffled_geometry = run_process_variant_experiment(world, "SHUFFLED_GEOMETRY", seeds=SMALL_SEEDS)
    randomized_order = run_process_variant_experiment(world, "RANDOMIZED_ORDER", seeds=SMALL_SEEDS)
    randomized_observable = run_randomized_control_experiment(world, seeds=SMALL_SEEDS)

    classification = classify_historical_reconstruction(failing_gate, coupled, zero_coupling, shuffled_geometry, randomized_order, randomized_observable)
    assert classification == "EXPERIMENTAL_CONFOUND_PRESENT"


def test_forbidden_conclusions_are_never_produced_by_the_classifier():
    forbidden = {"NEW_PHYSICS", "EMERGENT_SPACETIME", "TIME_TRAVEL", "CAUSAL_LOOP", "PHYSICAL_LAW_DISCOVERED"}
    world = WORLDS[1]
    coupled = run_process_variant_experiment(world, "COUPLED", seeds=SMALL_SEEDS)
    zero_coupling = run_process_variant_experiment(world, "ZERO_COUPLING", seeds=SMALL_SEEDS)
    shuffled_geometry = run_process_variant_experiment(world, "SHUFFLED_GEOMETRY", seeds=SMALL_SEEDS)
    randomized_order = run_process_variant_experiment(world, "RANDOMIZED_ORDER", seeds=SMALL_SEEDS)
    randomized_observable = run_randomized_control_experiment(world, seeds=SMALL_SEEDS)
    label_report = run_label_permutation_control(world, seeds=SMALL_SEEDS)
    identifiability = [run_identifiability_test(WORLDS[0], WORLDS[1])]
    gate = compute_scientific_gate(world, label_report, identifiability, scaling_study_present=True, intervention_robustness_present=True)

    classification = classify_historical_reconstruction(gate, coupled, zero_coupling, shuffled_geometry, randomized_order, randomized_observable)
    assert classification not in forbidden
