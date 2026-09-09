import numpy as np

from relational_dynamics.geometry import LorentzianEmbedding


def test_minkowski_light_cone_classification():
    coordinates = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
    ])
    embedding = LorentzianEmbedding(coordinates)
    assert embedding.causal_relation(0, 1) is True
    assert embedding.causal_relation(0, 2) is True
    assert embedding.causal_relation(0, 3) is False


def test_causal_prediction_metrics_against_known_relations():
    coordinates = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0],
    ])
    embedding = LorentzianEmbedding(coordinates)
    truth = np.array([[False, True, True, False], [False] * 4, [False, True, False, False], [False] * 4])
    metrics = embedding.compare_causal_relations(truth)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_links"] == 0
    assert metrics["false_negative_links"] == 0
