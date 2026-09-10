import numpy as np

from relational_dynamics.geometry import LorentzianEmbedding


def test_causal_metrics_include_f1_and_error_rates():
    embedding = LorentzianEmbedding(np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
    ]))
    truth = embedding.causal_matrix()
    metrics = embedding.compare_causal_relations(truth)
    assert metrics["f1"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0
