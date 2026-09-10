import numpy as np

from relational_dynamics.ground_truth import build_ground_truth_worlds
from relational_dynamics.historical_process import build_geometry_coupled_history, coupling_angle


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
