import numpy as np

from relational_dynamics.ground_truth import build_ground_truth_worlds, randomized_observables


def test_three_ground_truth_worlds_have_all_interval_classes():
    worlds = build_ground_truth_worlds()
    assert [world.world_id for world in worlds] == ["WORLD_1", "WORLD_2", "WORLD_3"]
    for world in worlds:
        assert world.coordinates.shape == (4, 4)
        assert set(world.pair_classifications.values()) == {"timelike", "null", "spacelike"}
        assert world.information_matrix.shape == (4, 4)
        assert world.causal_relation.shape == (4, 4)
        assert world.geometry_hidden_from_observables is True


def test_quantum_process_observables_are_not_metric_components():
    worlds = build_ground_truth_worlds()
    assert np.allclose(worlds[0].information_matrix, worlds[1].information_matrix)
    assert np.allclose(worlds[1].information_matrix, worlds[2].information_matrix)
    assert not np.shares_memory(worlds[0].coordinates, worlds[0].information_matrix)


def test_randomized_controls_preserve_marginals_but_break_pair_structure():
    world = build_ground_truth_worlds()[0]
    randomized = randomized_observables(world, seed=7)
    assert np.allclose(np.sort(randomized.information_matrix.ravel()), np.sort(world.information_matrix.ravel()))
    assert randomized.control_type == "pairwise_permutation_control"
    assert not np.array_equal(randomized.information_matrix, world.information_matrix)
