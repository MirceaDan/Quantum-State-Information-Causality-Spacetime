import numpy as np

from relational_dynamics.relational_clock import PageWoottersExperiment


def test_page_wootters_clock_has_distinct_conditioned_matter_states():
    experiment = PageWoottersExperiment.two_level_controlled()
    report = experiment.run()

    assert report.constraint_residual < 1e-12
    assert report.clock_probabilities[0] > 0
    assert report.clock_probabilities[1] > 0
    assert report.clock_distinguishability > 0.99
    assert report.conditional_fidelity < 1e-12
    assert report.consistent_with_page_wootters_toy_model


def test_clock_experiment_does_not_use_external_time_or_array_index():
    experiment = PageWoottersExperiment.two_level_controlled()
    report = experiment.run()

    assert report.external_time_used is False
    assert report.computational_index_used_as_time is False
