import pytest

from relational_dynamics.causal_topology import (
    CausalTopology,
    canonical_topology_key,
    has_cycle,
    is_acyclic,
    is_isomorphic,
    relabel_topology,
    valid_topological_node_orders,
)
from relational_dynamics.topology_generation import branch_topology, chain_topology, cyclic_topology, merge_topology


def test_cycle_detection():
    assert has_cycle(cyclic_topology()) is True
    assert has_cycle(chain_topology()) is False


def test_dag_detection():
    assert is_acyclic(chain_topology()) is True
    assert is_acyclic(branch_topology()) is True
    assert is_acyclic(merge_topology()) is True
    assert is_acyclic(cyclic_topology()) is False


def test_topology_permutation():
    chain = chain_topology()
    permutation = (2, 0, 3, 1)
    relabeled = relabel_topology(chain, permutation)
    assert relabeled.nodes == tuple(sorted(permutation))
    expected_edges = {(permutation[a], permutation[b]) for a, b in chain.directed_relations}
    assert set(relabeled.directed_relations) == expected_edges


def test_topology_permutation_rejects_non_bijection():
    chain = chain_topology()
    with pytest.raises(ValueError):
        relabel_topology(chain, (0, 0, 1, 2))


def test_topology_isomorphism():
    chain = chain_topology()
    reversed_chain = CausalTopology((0, 1, 2, 3), ((3, 2), (2, 1), (1, 0)))
    assert is_isomorphic(chain, reversed_chain)
    assert not is_isomorphic(chain, branch_topology())
    assert not is_isomorphic(chain, merge_topology())
    assert not is_isomorphic(chain, cyclic_topology())


def test_topology_isomorphism_under_relabeling():
    for topology in (chain_topology(), branch_topology(), merge_topology(), cyclic_topology()):
        permutation = (3, 2, 0, 1)
        relabeled = relabel_topology(topology, permutation)
        assert is_isomorphic(topology, relabeled)


def test_topology_canonicalization():
    chain = chain_topology()
    permutation = (3, 1, 0, 2)
    relabeled = relabel_topology(chain, permutation)
    assert canonical_topology_key(chain) == canonical_topology_key(relabeled)
    assert canonical_topology_key(chain) != canonical_topology_key(branch_topology())


def test_schedule_validity():
    chain = chain_topology()
    orders = valid_topological_node_orders(chain)
    assert orders == [(0, 1, 2, 3)]  # a chain has exactly one linear extension

    branch = branch_topology()
    branch_orders = valid_topological_node_orders(branch)
    assert (0, 1, 2, 3) in branch_orders
    assert (0, 2, 1, 3) in branch_orders  # B and C are unordered relative to each other
    for order in branch_orders:
        position = {node: index for index, node in enumerate(order)}
        assert all(position[source] < position[target] for source, target in branch.directed_relations)


def test_schedule_validity_rejects_cyclic_topologies():
    with pytest.raises(ValueError):
        valid_topological_node_orders(cyclic_topology())
