"""Milestone 5: hidden causal/process topology abstraction.

Separates causal TOPOLOGY (which nodes are causally related, and in which
direction) from LABELS (arbitrary node names) and from SCHEDULE (the order in
which a quantum simulator actually applies the corresponding relations -- see
``topology_observables.py``). This module is deliberately independent of any
quantum simulation code; it is pure graph mathematics.

Q = 0 and Delta_munu = 0: no speculative sector is used anywhere in M5.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools

MATHEMATICAL_DEFINITION = "MATHEMATICAL_DEFINITION"

CHAIN = "CHAIN"
BRANCH = "BRANCH"
MERGE = "MERGE"
CYCLIC_PROCESS = "CYCLIC_PROCESS"
UNSPECIFIED = "UNSPECIFIED"


@dataclass(frozen=True)
class CausalTopology:
    """A directed graph of causal/process relations. May contain a cycle
    (``CYCLIC_PROCESS``); this is an operational label for a graph property,
    never a claim of a physical closed timelike curve (see AUDIT.md)."""

    nodes: tuple[int, ...]
    directed_relations: tuple[tuple[int, int], ...]
    topology_class: str = UNSPECIFIED
    component_label: str = MATHEMATICAL_DEFINITION

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes)))
        for source, target in self.directed_relations:
            if source not in self.nodes or target not in self.nodes:
                raise ValueError("relation endpoints must be declared nodes")
            if source == target:
                raise ValueError("self-loops are not supported")


def has_cycle(topology: CausalTopology) -> bool:
    """Standard DFS-based directed-cycle detection (established graph theory)."""
    successors: dict[int, list[int]] = {node: [] for node in topology.nodes}
    for source, target in topology.directed_relations:
        successors[source].append(target)
    state = {node: 0 for node in topology.nodes}  # 0 unvisited, 1 in-progress, 2 done

    def visit(node: int) -> bool:
        state[node] = 1
        for neighbor in successors[node]:
            if state[neighbor] == 1:
                return True
            if state[neighbor] == 0 and visit(neighbor):
                return True
        state[node] = 2
        return False

    return any(state[node] == 0 and visit(node) for node in topology.nodes)


def is_acyclic(topology: CausalTopology) -> bool:
    return not has_cycle(topology)


def relabel_topology(topology: CausalTopology, permutation: tuple[int, ...]) -> CausalTopology:
    """Rename every node by ``permutation[old_label] = new_label``; pure relabeling."""
    if sorted(permutation[node] for node in topology.nodes) != sorted(topology.nodes):
        raise ValueError("permutation must be a bijection on the topology's nodes")
    new_relations = tuple((permutation[source], permutation[target]) for source, target in topology.directed_relations)
    return replace(topology, nodes=tuple(permutation[node] for node in topology.nodes), directed_relations=new_relations)


def _edge_set(topology: CausalTopology) -> frozenset[tuple[int, int]]:
    return frozenset(topology.directed_relations)


def is_isomorphic(topology_a: CausalTopology, topology_b: CausalTopology) -> bool:
    """Exhaustive permutation search (established graph theory); only feasible for small N,
    which is exactly the M5 regime (N<=5-6)."""
    if len(topology_a.nodes) != len(topology_b.nodes) or len(topology_a.directed_relations) != len(topology_b.directed_relations):
        return False
    target_edges = _edge_set(topology_b)
    for permutation in itertools.permutations(topology_b.nodes):
        mapping = dict(zip(topology_a.nodes, permutation))
        mapped_edges = frozenset((mapping[source], mapping[target]) for source, target in topology_a.directed_relations)
        if mapped_edges == target_edges:
            return True
    return False


def canonical_topology_key(topology: CausalTopology) -> tuple[tuple[int, int], ...]:
    """Lexicographically smallest relabeled edge set over all node permutations.

    Convenience for deduplication/evaluation ONLY (section 5): never used as the
    reconstruction mechanism itself, only to compare two ALREADY-RECOVERED
    topologies for isomorphism-equivalence, or to deduplicate an ensemble.
    """
    node_count = len(topology.nodes)
    best: tuple[tuple[int, int], ...] | None = None
    for permutation in itertools.permutations(range(node_count)):
        mapping = dict(zip(topology.nodes, permutation))
        relabeled = tuple(sorted((mapping[source], mapping[target]) for source, target in topology.directed_relations))
        if best is None or relabeled < best:
            best = relabeled
    return best


def valid_topological_node_orders(topology: CausalTopology) -> list[tuple[int, ...]]:
    """All linear extensions of an acyclic topology's partial order (brute force; small N only)."""
    if has_cycle(topology):
        raise ValueError("a cyclic topology has no valid topological node order")
    node_count = len(topology.nodes)
    valid_orders = []
    for permutation in itertools.permutations(topology.nodes):
        position = {node: index for index, node in enumerate(permutation)}
        if all(position[source] < position[target] for source, target in topology.directed_relations):
            valid_orders.append(permutation)
    return valid_orders
