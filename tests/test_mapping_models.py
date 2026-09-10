import numpy as np

from relational_dynamics.mapping_models import compare_mapping_models


def test_mapping_comparison_reports_low_capacity_models_and_splits():
    information = np.array([
        [0.0, 0.8, 0.3, 0.1],
        [0.8, 0.0, 0.7, 0.2],
        [0.3, 0.7, 0.0, 0.6],
        [0.1, 0.2, 0.6, 0.0],
    ])
    causal = np.zeros_like(information)
    truth = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0]])
    report = compare_mapping_models(information, causal, truth, seed=4)

    assert {item.model_id for item in report} == {"NULL", "FIXED_ANALYTICAL", "F_THETA"}
    for item in report:
        assert item.parameter_count >= 0
        assert item.train_loss >= 0
        assert item.validation_loss >= 0
        assert item.test_loss >= 0
        assert item.mapping_is_physical_law is False
