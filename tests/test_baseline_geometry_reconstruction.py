import numpy as np

from relational_dynamics.geometry_reconstruction import run_baseline_geometry_reconstruction


def test_reconstruction_hides_ground_truth_geometry_from_optimizer():
    report = run_baseline_geometry_reconstruction(seed=1234)

    assert report.geometry_was_hidden is True
    assert report.q_norm == 0.0
    assert report.delta_norm == 0.0
    assert report.parameter_count > 0
    assert report.train_loss >= 0.0
    assert report.test_loss >= 0.0
    assert report.observation_source == "independent_quantum_markov_process"


def test_reconstruction_reports_failure_or_success_without_overclaiming():
    report = run_baseline_geometry_reconstruction(seed=1234)

    assert report.classification in {
        "BASELINE_GEOMETRY_RECONSTRUCTION_VALID",
        "BASELINE_GEOMETRY_RECONSTRUCTION_FAILED",
    }
    assert report.claims_new_physics is False
    assert 0.0 <= report.causal_precision <= 1.0
    assert 0.0 <= report.causal_recall <= 1.0
