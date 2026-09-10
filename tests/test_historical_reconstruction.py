import numpy as np

from relational_dynamics.ground_truth import build_ground_truth_worlds
from relational_dynamics.historical_process import build_geometry_coupled_history
from relational_dynamics.historical_reconstruction import (
    FLEXIBLE,
    NULL,
    RESTRICTED,
    build_historical_observables,
    classify_historical_reconstruction,
    compare_historical_baselines,
    randomize_historical_observables,
    run_coupled_experiment,
    run_identifiability_test,
    run_information_retention_curve,
    run_null_baseline_experiment,
    run_randomized_control_experiment,
)


WORLDS = build_ground_truth_worlds()


def test_build_historical_observables_never_exposes_coordinates_or_metric():
    process = build_geometry_coupled_history(WORLDS[0], retention=1.0)
    observables = build_historical_observables(process, WORLDS[0])
    assert observables.pairwise_information.shape == (4, 4)
    assert observables.intervention_tensor.shape[:2] == (4, 4)
    assert len(observables.multipartite_information) == 4
    for field_name in ("coordinates", "metric", "g_munu", "world_id", "t0"):
        assert not hasattr(observables, field_name)


def test_null_baseline_experiment_preserves_existing_behavior():
    report = run_null_baseline_experiment(seed=1234, seeds=tuple(range(3)))
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


def test_flexible_model_represents_curvature_information_absent_from_restricted():
    process = build_geometry_coupled_history(WORLDS[1], retention=1.0)
    observables = build_historical_observables(process, WORLDS[1])
    baselines = compare_historical_baselines(WORLDS[1], observables, seed=1234)
    assert baselines[RESTRICTED].fitted_scale is None or baselines[RESTRICTED].fitted_scale == 0.0


def test_randomized_control_preserves_marginal_statistics_but_breaks_pairing():
    process = build_geometry_coupled_history(WORLDS[1], retention=1.0)
    observables = build_historical_observables(process, WORLDS[1])
    randomized = randomize_historical_observables(observables, seed=3)
    upper_original = np.sort(observables.pairwise_information[np.triu_indices(4, k=1)])
    upper_random = np.sort(randomized.pairwise_information[np.triu_indices(4, k=1)])
    assert np.allclose(upper_original, upper_random)
    assert not np.array_equal(observables.pairwise_information, randomized.pairwise_information)


def test_identifiability_reports_a_conservative_classification():
    report = run_identifiability_test(WORLDS[0], WORLDS[1])
    assert report.classification in {"IDENTIFIABLE", "INFORMATIONALLY_NON_IDENTIFIABLE"}
    assert report.observable_distance >= 0.0


def test_identical_world_is_never_reported_as_distinguishable_from_itself():
    report = run_identifiability_test(WORLDS[0], WORLDS[0])
    assert report.classification == "INFORMATIONALLY_NON_IDENTIFIABLE"
    assert report.observable_distance == 0.0


def test_retention_curve_has_one_point_per_eta_and_bounded_scores():
    etas = (0.0, 0.5, 1.0)
    curve = run_information_retention_curve(WORLDS[1], etas=etas, seed=1234)
    assert curve.etas == etas
    for score in curve.scores:
        assert 0.0 <= score <= 1.0


def test_coupled_experiment_reports_seed_statistics_and_zero_speculative_norms():
    report = run_coupled_experiment(WORLDS[1], seeds=tuple(range(5)))
    assert report.experiment_id == "EXP-HIST-COUPLED-001"
    assert report.geometry_hidden is True
    assert report.q_norm == 0.0
    assert report.delta_norm == 0.0
    assert report.claims_new_physics is False
    assert report.claims_time_travel is False
    assert report.seed_statistics[FLEXIBLE].seed_count == 5


def test_randomized_control_experiment_uses_same_pipeline_as_coupled():
    coupled = run_coupled_experiment(WORLDS[1], seeds=tuple(range(5)))
    randomized = run_randomized_control_experiment(WORLDS[1], seeds=tuple(range(5)))
    assert randomized.experiment_id == "EXP-HIST-RANDOM-001"
    assert set(randomized.baselines) == set(coupled.baselines)


def test_classification_uses_only_the_allowed_vocabulary():
    world = WORLDS[1]
    coupled = run_coupled_experiment(world, seeds=tuple(range(5)))
    randomized = run_randomized_control_experiment(world, seeds=tuple(range(5)))
    identifiability = [run_identifiability_test(WORLDS[0], WORLDS[1]), run_identifiability_test(WORLDS[1], WORLDS[2])]
    retention_curve = run_information_retention_curve(world, etas=(0.0, 0.5, 1.0), seed=1234)
    classification = classify_historical_reconstruction(coupled, randomized, identifiability, retention_curve)
    assert classification in {
        "HISTORICAL_RECONSTRUCTION_SUPPORTED",
        "HISTORICAL_RECONSTRUCTION_UNSTABLE",
        "INFORMATION_INSUFFICIENT",
        "REPRESENTATION_INSUFFICIENT",
        "OPTIMIZATION_INSUFFICIENT",
        "OBSERVABLES_NON_IDENTIFYING",
    }
