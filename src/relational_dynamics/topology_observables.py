"""Milestone 5: convert a hidden CausalTopology into an M4 quantum process and
extract its observable bundle. Reuses M4's substrate (process_topology.py,
permutation_equivariant_process.py, process_memory_observables.py) directly;
no second quantum simulator is created (section 2).

SCHEDULE vs TOPOLOGY (section 8): ``directed_relations`` in a CausalTopology
declares WHICH nodes are causally related and in which direction; it does not
by itself say in what ORDER the corresponding two-qubit gates are applied.
``schedule_from_node_order`` makes this explicit: for an acyclic topology, ANY
valid topological node order induces a valid relation execution schedule; for
a cyclic topology there is no topological order, so the declared relation
tuple order is used as the single documented operational execution pass (not
a physical closed-timelike-curve simulation -- see AUDIT.md).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .causal_topology import CausalTopology, has_cycle
from .permutation_equivariant_process import simulate_process
from .process_memory_observables import intervention_response_tensor, multipartite_information, pairwise_mutual_information, process_memory_score
from .process_topology import NodeAttributes, ProcessTopology, Relation

MODELING_ASSUMPTION = "MODELING_ASSUMPTION"


def schedule_from_node_order(topology: CausalTopology, node_order: tuple[int, ...] | None) -> tuple[int, ...]:
    """Return the relation-index execution order induced by ``node_order``.

    For an acyclic topology, ``node_order`` MUST be one of
    ``valid_topological_node_orders(topology)``. For a cyclic topology,
    ``node_order`` must be ``None`` and the declared relation order is used
    (the single documented operational pass for a graph containing a cycle).
    """
    if has_cycle(topology):
        if node_order is not None:
            raise ValueError("a cyclic topology has no valid node order; pass node_order=None")
        return tuple(range(len(topology.directed_relations)))
    if node_order is None:
        raise ValueError("an acyclic topology requires an explicit valid node order")
    position = {node: index for index, node in enumerate(node_order)}
    keys = [(position[source], position[target]) for source, target in topology.directed_relations]
    return tuple(sorted(range(len(keys)), key=lambda index: keys[index]))


@dataclass(frozen=True)
class ProcessParameters:
    """Physical parameters NOT determined by the topology itself (section 16)."""

    strengths: tuple[float, ...]
    initial_state_angles: tuple[float, ...]
    noise_parameters: tuple[float, ...]
    retention: float = 1.0


def default_process_parameters(topology: CausalTopology, seed: int = 0, strength: float = 0.6) -> ProcessParameters:
    rng = np.random.default_rng(seed)
    relation_count = len(topology.directed_relations)
    node_count = len(topology.nodes)
    strengths = tuple(float(strength) for _ in range(relation_count))
    angles = tuple(float(value) for value in rng.uniform(0.0, 2 * np.pi, size=node_count))
    noise = tuple(float(value) for value in rng.uniform(0.0, 0.1, size=node_count))
    return ProcessParameters(strengths, angles, noise)


def build_m4_process_topology(topology: CausalTopology, parameters: ProcessParameters, node_order: tuple[int, ...] | None = None) -> ProcessTopology:
    """Convert a hidden CausalTopology + declared parameters into an M4 ProcessTopology.

    The relation ``order`` field is assigned from ``schedule_from_node_order``
    (an explicit, declared schedule choice), never from ``range(N-1))`` or raw
    list position.
    """
    schedule = schedule_from_node_order(topology, node_order)
    node_attributes = {
        node: NodeAttributes(2, parameters.initial_state_angles[node], parameters.noise_parameters[node]) for node in topology.nodes
    }
    relations = tuple(
        Relation(source, target, parameters.strengths[index], order=schedule[index])
        for index, (source, target) in enumerate(topology.directed_relations)
    )
    return ProcessTopology(topology.nodes, node_attributes, relations)


@dataclass(frozen=True)
class TopologyObservableBundle:
    """O(G): everything the reconstruction procedure is allowed to see. Never
    contains the topology, schedule, coupling strengths, or coordinates."""

    pairwise_information: np.ndarray
    multipartite_information: dict[tuple[int, int, int], float]
    intervention_tensor: dict[tuple[int, int], float]
    memory_score: float
    node_count: int


def build_observable_bundle(topology: CausalTopology, parameters: ProcessParameters, node_order: tuple[int, ...] | None = None) -> TopologyObservableBundle:
    process_topology = build_m4_process_topology(topology, parameters, node_order)
    result = simulate_process(process_topology, retention=parameters.retention)
    mi = pairwise_mutual_information(result)
    multi = multipartite_information(result)
    response = intervention_response_tensor(process_topology, retention=parameters.retention)
    memory = process_memory_score(process_topology, retention=parameters.retention)
    return TopologyObservableBundle(mi, multi, response.tensor, memory, len(topology.nodes))


def build_observable_bundle_broken(topology: CausalTopology, parameters: ProcessParameters, node_order: tuple[int, ...] | None = None) -> TopologyObservableBundle:
    """Negative control C (section 18): identical pipeline, but built on M4's
    deliberately label-dependent broken simulator instead of ``simulate_process``."""
    from .broken_process_control import simulate_process_broken_label_dependent

    process_topology = build_m4_process_topology(topology, parameters, node_order)
    result = simulate_process_broken_label_dependent(process_topology, retention=parameters.retention)
    mi = pairwise_mutual_information(result)
    multi = multipartite_information(result)
    memory = process_memory_score(process_topology, retention=parameters.retention)
    return TopologyObservableBundle(mi, multi, {}, memory, len(topology.nodes))
