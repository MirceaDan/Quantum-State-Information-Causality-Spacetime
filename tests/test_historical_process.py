import numpy as np

from relational_dynamics.ground_truth import build_ground_truth_worlds
from relational_dynamics.historical_process import (
    _PAULI_X,
    _PAULI_Z,
    build_geometry_coupled_history,
    coupling_angle,
    historical_intervention_response,
)


def test_coupling_angle_decreases_with_interval_magnitude():
    assert coupling_angle(0.0) > coupling_angle(1.0) > coupling_angle(10.0)


def test_process_state_is_a_valid_density_matrix_for_every_world():
    for world in build_ground_truth_worlds():
        process = build_geometry_coupled_history(world, retention=1.0)
        rho = process.system.rho
        assert np.isclose(np.trace(rho), 1.0, atol=1e-8)
        assert np.allclose(rho, rho.conj().T, atol=1e-8)
        assert np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() > -1e-8
        assert process.coupling_mechanism == "geometry_dependent_channel_strength_from_invariant_interval"
        assert len(process.coupling_angles) == process.event_count - 1


def test_retention_zero_destroys_more_information_than_retention_one():
    world = build_ground_truth_worlds()[1]
    high = build_geometry_coupled_history(world, retention=1.0).system
    low = build_geometry_coupled_history(world, retention=0.0).system
    high_mi = high.mutual_information((0,), (1,))
    low_mi = low.mutual_information((0,), (1,))
    assert high_mi >= low_mi


def test_coupling_depends_only_on_invariant_interval_not_on_raw_coordinates():
    import dataclasses

    world = build_ground_truth_worlds()[0]
    translated = dataclasses.replace(world, coordinates=world.coordinates + np.array([5.0, 3.0, 0.0, 0.0]))
    process_a = build_geometry_coupled_history(world, retention=1.0)
    process_b = build_geometry_coupled_history(translated, retention=1.0)
    assert process_a.coupling_angles == process_b.coupling_angles


def test_zero_coupling_control_gives_zero_angles():
    world = build_ground_truth_worlds()[1]
    process = build_geometry_coupled_history(world, retention=1.0, force_zero_coupling=True)
    assert process.control_label == "ZERO_COUPLING"
    assert all(angle == 0.0 for angle in process.coupling_angles)


def test_shuffled_geometry_control_reassigns_intervals_but_keeps_the_multiset():
    world = build_ground_truth_worlds()[1]
    baseline = build_geometry_coupled_history(world, retention=1.0)
    permutation = (2, 0, 1)
    shuffled = build_geometry_coupled_history(world, retention=1.0, interval_permutation=permutation)
    assert shuffled.control_label == "SHUFFLED_GEOMETRY"
    assert shuffled.coupling_angles != baseline.coupling_angles
    assert shuffled.interval_permutation == permutation


def test_randomized_order_control_executes_every_step_exactly_once():
    world = build_ground_truth_worlds()[1]
    schedule = (2, 0, 1)
    process = build_geometry_coupled_history(world, retention=1.0, schedule=schedule)
    assert process.control_label == "RANDOMIZED_ORDER"
    assert process.schedule == schedule
    assert sorted(process.schedule) == list(range(process.event_count - 1))


def test_invalid_permutation_arguments_are_rejected():
    world = build_ground_truth_worlds()[1]
    try:
        build_geometry_coupled_history(world, interval_permutation=(0, 0, 1))
        assert False, "expected a ValueError for a non-permutation interval_permutation"
    except ValueError:
        pass
    try:
        build_geometry_coupled_history(world, schedule=(0, 1))
        assert False, "expected a ValueError for a schedule of the wrong length"
    except ValueError:
        pass


def test_intervention_family_x_and_z_both_produce_a_forward_causal_chain():
    world = build_ground_truth_worlds()[1]
    for operator, label in ((_PAULI_X, "X"), (_PAULI_Z, "Z")):
        response = historical_intervention_response(world, retention=1.0, intervention_operator=operator, intervention_label=label)
        assert response.intervention_family == label
        # no backward influence: event k cannot signal any event before it
        for target in range(response.tensor.shape[0]):
            for source in range(target + 1, response.tensor.shape[0]):
                assert response.tensor[source, target] < 1e-10
