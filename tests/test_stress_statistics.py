import numpy as np

from relational_dynamics.geometry_reconstruction import run_multi_seed_reconstruction


def test_multi_seed_report_has_required_statistics():
    report = run_multi_seed_reconstruction(seeds=range(10))
    assert report.seed_count == 10
    assert report.mean_test_loss >= 0.0
    assert report.std_test_loss >= 0.0
    assert report.median_test_loss >= 0.0
    assert report.worst_test_loss >= report.best_test_loss
    assert len(report.per_seed) == 10
