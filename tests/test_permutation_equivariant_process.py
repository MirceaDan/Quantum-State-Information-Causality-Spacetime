import itertools
import numpy as np
import pytest

from relational_dynamics.process_topology import line_topology, permute_process
from relational_dynamics.permutation_equivariant_process import simulate_process
from relational_dynamics.process_memory_observables import (
    check_global_scalar_invariants,
    check_intervention_tensor_equivariance,
    check_memory_score_invariance,
    check_multipartite_observable_equivariance,
    check_pairwise_observable_equivariance,
    check_state_equivariance,
    intervention_response_tensor,
    multipartite_information,
    pairwise_mutual_information,
    process_memory_score,
)
from relational_dynamics.broken_process_control import check_broken_state_equivariance

TOLERANCE = 1e-8


def _sample_topology(noise=(0.0, 0.0, 0.0, 0.0)):
    return line_topology(
        (0, 1, 2, 3),
        (0.3, 0.5, 0.2),
        initial_state_angles=(0.1, 1.2, 0.4, 2.0),
        noise_parameters=noise,
    )


def _random_permutations(n, count, seed):
    rng = np.random.default_rng(seed)
    return [tuple(int(index) for index in rng.permutation(n)) for _ in range(count)]


# ---------------------------------------------------------------------------
# Section 7: state / pairwise / multipartite / intervention / memory equivariance
# ---------------------------------------------------------------------------


def test_state_equivariance_holds_for_every_permutation_of_a_small_process():
    topology = _sample_topology()
    for permutation in itertools.permutations(range(4)):
        result = check_state_equivariance(topology, permutation, tolerance=TOLERANCE)
        assert result.passed, (permutation, result)


def test_pairwise_observable_equivariance():
    topology = _sample_topology()
    for permutation in _random_permutations(4, 8, seed=0):
        result = check_pairwise_observable_equivariance(topology, permutation, tolerance=TOLERANCE)
        assert result.passed, (permutation, result)


def test_multipartite_observable_equivariance():
    topology = _sample_topology()
    for permutation in _random_permutations(4, 8, seed=1):
        result = check_multipartite_observable_equivariance(topology, permutation, tolerance=TOLERANCE)
        assert result.passed, (permutation, result)


def test_intervention_tensor_equivariance():
    topology = _sample_topology()
    for permutation in _random_permutations(4, 5, seed=2):
        result = check_intervention_tensor_equivariance(topology, permutation, tolerance=TOLERANCE)
        assert result.passed, (permutation, result)


def test_memory_observable_invariance():
    topology = _sample_topology()
    for permutation in _random_permutations(4, 8, seed=3):
        result = check_memory_score_invariance(topology, permutation, tolerance=TOLERANCE)
        assert result.passed, (permutation, result)


def test_noise_equivariance_holds_with_deterministic_cptp_noise():
    """Deterministic Kraus-sum depolarizing 'noise' is used (no RNG), so equivariance
    still holds to floating precision -- see METHODS.md section 13."""
    topology = _sample_topology(noise=(0.3, 0.4, 0.2, 0.5))
    for permutation in _random_permutations(4, 5, seed=4):
        state = check_state_equivariance(topology, permutation, retention=0.5, tolerance=TOLERANCE)
        pairwise = check_pairwise_observable_equivariance(topology, permutation, retention=0.5, tolerance=TOLERANCE)
        assert state.passed and pairwise.passed


def test_global_scalar_invariants_hold_under_relabeling():
    topology = _sample_topology()
    for permutation in _random_permutations(4, 5, seed=5):
        checks = check_global_scalar_invariants(topology, permutation, tolerance=TOLERANCE)
        assert all(check.passed for check in checks.values()), checks


# ---------------------------------------------------------------------------
# Section 8: metamorphic tests A-D
# ---------------------------------------------------------------------------


def test_metamorphic_A_permutation_preserves_observables_up_to_relabeling():
    topology = _sample_topology()
    permutation = (2, 0, 3, 1)
    original = pairwise_mutual_information(simulate_process(topology))
    permuted = pairwise_mutual_information(simulate_process(permute_process(topology, permutation)))
    np.testing.assert_allclose(permuted[np.ix_(permutation, permutation)], original, atol=TOLERANCE)


def test_metamorphic_B_permutation_then_inverse_recovers_original_topology_observables():
    topology = _sample_topology()
    permutation = (2, 0, 3, 1)
    inverse = tuple(int(index) for index in np.argsort(permutation))
    round_trip = permute_process(permute_process(topology, permutation), inverse)
    original_mi = pairwise_mutual_information(simulate_process(topology))
    round_trip_mi = pairwise_mutual_information(simulate_process(round_trip))
    np.testing.assert_allclose(round_trip_mi, original_mi, atol=TOLERANCE)
    assert round_trip.node_labels == topology.node_labels


def test_metamorphic_C_permutation_composition_matches_direct_composed_permutation():
    """(P^pi)^sigma == P^(sigma o pi), acting LEFT-to-RIGHT: pi is applied first, sigma second."""
    topology = _sample_topology()
    pi = (1, 2, 3, 0)
    sigma = (3, 2, 1, 0)
    composed = tuple(sigma[pi[label]] for label in range(4))
    sequential = permute_process(permute_process(topology, pi), sigma)
    direct = permute_process(topology, composed)
    assert sequential.node_labels == direct.node_labels
    sequential_edges = {(r.first, r.second, r.strength, r.order) for r in sequential.relations}
    direct_edges = {(r.first, r.second, r.strength, r.order) for r in direct.relations}
    assert sequential_edges == direct_edges


def test_metamorphic_D_global_scalars_are_invariant():
    topology = _sample_topology()
    permutation = (3, 1, 0, 2)
    original_score = process_memory_score(topology)
    permuted_score = process_memory_score(permute_process(topology, permutation))
    assert abs(original_score - permuted_score) < TOLERANCE


# ---------------------------------------------------------------------------
# Section 9: negative control
# ---------------------------------------------------------------------------


def test_broken_model_fails_equivariance():
    topology = _sample_topology()
    failures = 0
    for permutation in itertools.permutations(range(4)):
        if permutation == tuple(range(4)):
            continue
        result = check_broken_state_equivariance(topology, permutation, tolerance=TOLERANCE)
        if not result.passed:
            failures += 1
    assert failures > 0, "the negative control must fail equivariance for at least one non-identity permutation"


def test_correct_model_does_not_share_the_broken_models_failure_mode():
    topology = _sample_topology()
    permutation = (2, 0, 3, 1)
    correct = check_state_equivariance(topology, permutation, tolerance=TOLERANCE)
    broken = check_broken_state_equivariance(topology, permutation, tolerance=TOLERANCE)
    assert correct.passed is True
    assert broken.passed is False
    assert broken.max_error > correct.max_error


# ---------------------------------------------------------------------------
# Section 14: reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility_of_simulation_and_observables():
    """simulate_process has no RNG at all; identical inputs must give bit-identical output."""
    topology = _sample_topology(noise=(0.2, 0.1, 0.0, 0.3))
    first = simulate_process(topology, retention=0.7)
    second = simulate_process(topology, retention=0.7)
    np.testing.assert_array_equal(first.final_system.rho, second.final_system.rho)
    assert intervention_response_tensor(topology, retention=0.7).tensor == intervention_response_tensor(topology, retention=0.7).tensor
