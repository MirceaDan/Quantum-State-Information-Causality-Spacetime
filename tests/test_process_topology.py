import itertools
import numpy as np
import pytest

from relational_dynamics.permutation_equivariant_process import permutation_operator, simulate_process
from relational_dynamics.process_topology import (
    NodeAttributes,
    ProcessTopology,
    Relation,
    canonicalize_process,
    complete_topology,
    line_topology,
    permute_process,
)


def test_permutation_operator_is_unitary():
    for num_qubits in (1, 2, 3, 4):
        for permutation in itertools.permutations(range(num_qubits)):
            operator = permutation_operator(permutation, num_qubits)
            identity = np.eye(2**num_qubits)
            np.testing.assert_allclose(operator.conj().T @ operator, identity, atol=1e-12)


def test_permutation_operator_rejects_non_bijections():
    with pytest.raises(ValueError):
        permutation_operator((0, 0, 1), 3)


def test_permutation_inverse():
    permutation = (2, 0, 1)
    forward = permutation_operator(permutation, 3)
    inverse_permutation = tuple(int(index) for index in np.argsort(permutation))
    backward = permutation_operator(inverse_permutation, 3)
    np.testing.assert_allclose(forward @ backward, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(backward @ forward, np.eye(8), atol=1e-12)


def test_permutation_composition():
    """(P^pi)^sigma == P^(sigma o pi): applying pi's operator then sigma's operator on kets
    equals applying the composed permutation's operator directly."""
    pi = (1, 2, 0)
    sigma = (2, 0, 1)
    composed = tuple(sigma[pi[index]] for index in range(3))
    left = permutation_operator(sigma, 3) @ permutation_operator(pi, 3)
    right = permutation_operator(composed, 3)
    np.testing.assert_allclose(left, right, atol=1e-12)


def test_process_relabeling_moves_attributes_and_relations_with_the_labels():
    topology = line_topology((0, 1, 2), (0.3, 0.6), initial_state_angles=(0.1, 0.2, 0.3), noise_parameters=(0.0, 0.1, 0.2))
    permutation = (2, 0, 1)
    relabeled = permute_process(topology, permutation)
    assert set(relabeled.node_labels) == {0, 1, 2}
    # what used to be node 0's attributes now live at label permutation[0] == 2
    assert relabeled.node_attributes[2].initial_state_angle == topology.node_attributes[0].initial_state_angle
    assert relabeled.node_attributes[2].noise_parameter == topology.node_attributes[0].noise_parameter
    # the physical edge {0,1} becomes edge {2,0}, and {1,2} becomes {0,1}
    original_edges = {frozenset((relation.first, relation.second)) for relation in topology.relations}
    relabeled_edges = {frozenset((relation.first, relation.second)) for relation in relabeled.relations}
    expected_edges = {frozenset({permutation[a], permutation[b]}) for a, b in original_edges}
    assert relabeled_edges == expected_edges
    assert frozenset({2, 0}) in relabeled_edges


def test_permute_process_rejects_non_bijection():
    topology = line_topology((0, 1, 2), (0.1, 0.2))
    with pytest.raises(ValueError):
        permute_process(topology, (0, 0, 1))


def test_line_topology_declares_relations_explicitly_not_from_index_arithmetic():
    topology = line_topology((5, 7, 9), (0.4, 0.7))
    edges = {frozenset((relation.first, relation.second)) for relation in topology.relations}
    assert edges == {frozenset({5, 7}), frozenset({7, 9})}
    orders = sorted(relation.order for relation in topology.relations)
    assert orders == [0, 1]


def test_complete_topology_requires_every_pair_and_rejects_index_derived_order():
    labels = (0, 1, 2)
    specs = ((0, 1, 0.1, 5), (0, 2, 0.2, 1), (1, 2, 0.3, 9))
    topology = complete_topology(labels, specs)
    edges = {frozenset((relation.first, relation.second)) for relation in topology.relations}
    assert edges == {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}
    orders = {(relation.first, relation.second): relation.order for relation in topology.relations}
    assert orders[(0, 1)] == 5 and orders[(0, 2)] == 1 and orders[(1, 2)] == 9


def test_complete_topology_rejects_missing_pairs():
    labels = (0, 1, 2)
    incomplete_specs = ((0, 1, 0.1, 0),)
    with pytest.raises(ValueError):
        complete_topology(labels, incomplete_specs)


def test_canonicalize_process_is_permutation_invariant():
    """canonicalize_process(P) and canonicalize_process(P^pi) must yield the SAME canonical topology."""
    topology = line_topology((0, 1, 2), (0.4, 0.9), initial_state_angles=(0.3, 1.1, 0.7), noise_parameters=(0.1, 0.0, 0.2))
    permutation = (2, 0, 1)
    canonical_a, _ = canonicalize_process(topology)
    canonical_b, _ = canonicalize_process(permute_process(topology, permutation))
    assert canonical_a.node_attributes == canonical_b.node_attributes
    edges_a = {(r.first, r.second, r.strength, r.order) for r in canonical_a.relations}
    edges_b = {(r.first, r.second, r.strength, r.order) for r in canonical_b.relations}
    assert edges_a == edges_b


def test_topology_validates_relation_endpoints_are_declared_labels():
    with pytest.raises(ValueError):
        ProcessTopology((0, 1), {0: NodeAttributes(), 1: NodeAttributes()}, (Relation(0, 5, 0.1, 0),))
