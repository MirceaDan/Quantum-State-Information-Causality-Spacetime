import numpy as np

from relational_dynamics.gr import bianchi_convergence, conformal_geometry


def test_bianchi_convergence_reports_resolution_and_rate():
    report = bianchi_convergence(conformal_geometry(scale=0.2), np.array([0.3, 0.1, 0.0, 0.0]), steps=(1e-3, 5e-4, 2.5e-4))
    assert report.steps == (1e-3, 5e-4, 2.5e-4)
    assert len(report.residuals) == 3
    assert report.observed_order > 0.5
