"""Milestone 5: hidden topology construction (mandatory examples + reproducible ensemble).

Topology generation is entirely separate from any quantum simulation or
observable code, so that ensemble construction cannot accidentally leak
information into the observable pipeline (section 7/25).
"""

from __future__ import annotations

import itertools

from .causal_topology import BRANCH, CHAIN, CYCLIC_PROCESS, MERGE, CausalTopology, canonical_topology_key


def chain_topology(node_count: int = 4) -> CausalTopology:
    """A -> B -> C -> D (mandatory example, section 3)."""
    nodes = tuple(range(node_count))
    relations = tuple((index, index + 1) for index in range(node_count - 1))
    return CausalTopology(nodes, relations, CHAIN)


def branch_topology() -> CausalTopology:
    """A -> B, A -> C -> D (mandatory example, section 3), N=4."""
    return CausalTopology((0, 1, 2, 3), ((0, 1), (0, 2), (2, 3)), BRANCH)


def merge_topology() -> CausalTopology:
    """A -> C, B -> C, C -> D (mandatory example, section 3), N=4."""
    return CausalTopology((0, 1, 2, 3), ((0, 2), (1, 2), (2, 3)), MERGE)


def cyclic_topology(node_count: int = 4) -> CausalTopology:
    """A -> B -> C -> A (mandatory example, section 3); node D (if node_count>3) is an
    unconnected spectator, exactly matching the example given in the milestone spec.
    Labelled ``CYCLIC_PROCESS`` per section 3 -- this is a graph-theoretic property of the
    declared relation set, not a physical closed timelike curve (see AUDIT.md)."""
    nodes = tuple(range(node_count))
    relations = ((0, 1), (1, 2), (2, 0))
    return CausalTopology(nodes, relations, CYCLIC_PROCESS)


def mandatory_topologies(node_count: int = 4) -> list[CausalTopology]:
    return [chain_topology(node_count), branch_topology(), merge_topology(), cyclic_topology(node_count)]


def _all_directed_edges(node_count: int) -> list[tuple[int, int]]:
    return [(source, target) for source, target in itertools.permutations(range(node_count), 2)]


def generate_ensemble(node_count: int = 4, edge_counts: tuple[int, ...] = (3,), seed: int = 0, max_size: int = 16) -> list[CausalTopology]:
    """Enumerate all directed subgraphs with the given edge counts, deduplicate by
    isomorphism (via ``canonical_topology_key``), and return a reproducible sample.

    Procedure (documented per section 6): (1) build every directed edge subset of the
    requested sizes among ``node_count`` labelled nodes; (2) compute each subgraph's
    canonical key; (3) keep one representative per distinct canonical key; (4) always
    include the four mandatory examples first; (5) deterministically sort the
    remainder by canonical key and cap at ``max_size``. No randomness is used except
    to document a ``seed`` field for reproducibility bookkeeping (the procedure itself
    is fully deterministic given ``node_count``/``edge_counts``).
    """
    edges = _all_directed_edges(node_count)
    seen_keys: dict[tuple, CausalTopology] = {}

    mandatory = mandatory_topologies(node_count)
    for topology in mandatory:
        seen_keys[canonical_topology_key(topology)] = topology

    for edge_count in edge_counts:
        for combination in itertools.combinations(edges, edge_count):
            nodes_used = {node for edge in combination for node in edge}
            if len(nodes_used) < node_count:
                continue  # require every node to participate, for a well-posed N-node topology
            topology = CausalTopology(tuple(range(node_count)), tuple(combination), "ENSEMBLE")
            key = canonical_topology_key(topology)
            if key not in seen_keys:
                seen_keys[key] = topology

    ordered_keys = [canonical_topology_key(topology) for topology in mandatory]
    remaining_keys = sorted(key for key in seen_keys if key not in ordered_keys)
    final_keys = (ordered_keys + remaining_keys)[:max_size]
    return [seen_keys[key] for key in final_keys]
