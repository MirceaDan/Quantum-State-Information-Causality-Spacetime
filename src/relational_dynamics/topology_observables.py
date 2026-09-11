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
from dataclasses import replace
import itertools
import numpy as np

from .causal_topology import CausalTopology, has_cycle, relabel_topology
from .permutation_equivariant_process import permutation_operator, simulate_process
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


@dataclass(frozen=True)
class PhysicalProcessInstance:
    """A complete reproducible process; all node/edge data move under relabeling."""

    topology: CausalTopology
    node_initial_state_angles: dict[int, float]
    node_noise_parameters: dict[int, float]
    relation_strengths: dict[tuple[int, int], float]
    relation_schedule: tuple[tuple[int, int], ...]
    retention: float
    seed: int

    def __post_init__(self) -> None:
        nodes = set(self.topology.nodes)
        edges = set(self.topology.directed_relations)
        if set(self.node_initial_state_angles) != nodes or set(self.node_noise_parameters) != nodes:
            raise ValueError("node parameters must cover exactly the topology nodes")
        if set(self.relation_strengths) != edges or set(self.relation_schedule) != edges:
            raise ValueError("relation strengths and schedule must cover exactly the topology relations")
        if len(self.relation_schedule) != len(edges):
            raise ValueError("relation schedule must contain every relation exactly once")


def create_physical_process_instance(
    topology: CausalTopology,
    seed: int,
    node_order: tuple[int, ...] | None = None,
    strength_range: tuple[float, float] = (0.35, 0.75),
    noise_range: tuple[float, float] = (0.0, 0.1),
    retention: float = 1.0,
) -> PhysicalProcessInstance:
    """Sample one nuisance-parameter realization and freeze it as physical data."""
    rng = np.random.default_rng(seed)
    nodes = topology.nodes
    edges = topology.directed_relations
    schedule_indices = schedule_from_node_order(topology, node_order)
    schedule = tuple(edges[index] for index in schedule_indices)
    return PhysicalProcessInstance(
        topology,
        {node: float(rng.uniform(0.0, 2 * np.pi)) for node in nodes},
        {node: float(rng.uniform(*noise_range)) for node in nodes},
        {edge: float(rng.uniform(*strength_range)) for edge in edges},
        schedule,
        retention,
        seed,
    )


def relabel_physical_process(instance: PhysicalProcessInstance, permutation: tuple[int, ...]) -> PhysicalProcessInstance:
    """Relabel the entire physical process without resampling or changing its schedule."""
    relabeled_topology = relabel_topology(instance.topology, permutation)
    return PhysicalProcessInstance(
        relabeled_topology,
        {permutation[node]: value for node, value in instance.node_initial_state_angles.items()},
        {permutation[node]: value for node, value in instance.node_noise_parameters.items()},
        {(permutation[source], permutation[target]): value for (source, target), value in instance.relation_strengths.items()},
        tuple((permutation[source], permutation[target]) for source, target in instance.relation_schedule),
        instance.retention,
        instance.seed,
    )


def reschedule_physical_process(instance: PhysicalProcessInstance, node_order: tuple[int, ...]) -> PhysicalProcessInstance:
    """Change only the operational schedule; all topology and nuisance data stay fixed."""
    schedule_indices = schedule_from_node_order(instance.topology, node_order)
    schedule = tuple(instance.topology.directed_relations[index] for index in schedule_indices)
    return replace(instance, relation_schedule=schedule)


def build_m4_process_from_instance(instance: PhysicalProcessInstance) -> ProcessTopology:
    order_by_edge = {edge: order for order, edge in enumerate(instance.relation_schedule)}
    node_attributes = {
        node: NodeAttributes(2, instance.node_initial_state_angles[node], instance.node_noise_parameters[node])
        for node in instance.topology.nodes
    }
    relations = tuple(
        Relation(source, target, instance.relation_strengths[(source, target)], order_by_edge[(source, target)])
        for source, target in instance.topology.directed_relations
    )
    return ProcessTopology(instance.topology.nodes, node_attributes, relations)


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
    memory = float(np.mean(list(response.tensor.values()))) if response.tensor else 0.0
    return TopologyObservableBundle(mi, multi, response.tensor, memory, len(topology.nodes))


def build_observable_bundle_from_instance(instance: PhysicalProcessInstance) -> TopologyObservableBundle:
    """Generate observables from one frozen physical instance; exposes no hidden metadata."""
    process_topology = build_m4_process_from_instance(instance)
    result = simulate_process(process_topology, retention=instance.retention)
    response = intervention_response_tensor(process_topology, retention=instance.retention)
    return TopologyObservableBundle(
        pairwise_mutual_information(result),
        multipartite_information(result),
        response.tensor,
        float(np.mean(list(response.tensor.values()))) if response.tensor else 0.0,
        len(instance.topology.nodes),
    )


@dataclass(frozen=True)
class ProcessRelabelingReport:
    permutations_tested: int
    state_max_error: float
    pairwise_max_error: float
    multipartite_max_error: float
    intervention_max_error: float
    memory_max_error: float
    global_scalar_max_error: float
    passed: bool


def same_process_relabeling_report(instance: PhysicalProcessInstance, tolerance: float = 1e-8) -> ProcessRelabelingReport:
    """Exhaustively test one frozen process under every node relabeling."""
    original_process = build_m4_process_from_instance(instance)
    original_result = simulate_process(original_process, retention=instance.retention)
    original_bundle = build_observable_bundle_from_instance(instance)
    original_spectrum = np.sort(np.linalg.eigvalsh(original_result.final_system.rho))
    errors = {name: [] for name in ("state", "pairwise", "multipartite", "intervention", "memory", "global")}
    for permutation in itertools.permutations(instance.topology.nodes):
        relabeled = relabel_physical_process(instance, permutation)
        relabeled_process = build_m4_process_from_instance(relabeled)
        relabeled_result = simulate_process(relabeled_process, retention=relabeled.retention)
        relabeled_bundle = build_observable_bundle_from_instance(relabeled)
        operator = permutation_operator(permutation, len(permutation))
        recovered_state = operator.conj().T @ relabeled_result.final_system.rho @ operator
        errors["state"].append(float(np.max(np.abs(recovered_state - original_result.final_system.rho))))
        errors["pairwise"].append(float(np.max(np.abs(relabeled_bundle.pairwise_information[np.ix_(permutation, permutation)] - original_bundle.pairwise_information))))
        errors["multipartite"].extend(
            abs(relabeled_bundle.multipartite_information[tuple(sorted(permutation[node] for node in triple))] - value)
            for triple, value in original_bundle.multipartite_information.items()
        )
        errors["intervention"].extend(
            abs(relabeled_bundle.intervention_tensor[(permutation[source], permutation[target])] - value)
            for (source, target), value in original_bundle.intervention_tensor.items()
        )
        errors["memory"].append(abs(relabeled_bundle.memory_score - original_bundle.memory_score))
        relabeled_spectrum = np.sort(np.linalg.eigvalsh(relabeled_result.final_system.rho))
        errors["global"].append(float(np.max(np.abs(relabeled_spectrum - original_spectrum))))

    maxima = {name: max(values, default=0.0) for name, values in errors.items()}
    return ProcessRelabelingReport(
        len(tuple(itertools.permutations(instance.topology.nodes))),
        maxima["state"],
        maxima["pairwise"],
        maxima["multipartite"],
        maxima["intervention"],
        maxima["memory"],
        maxima["global"],
        max(maxima.values()) <= tolerance,
    )


@dataclass(frozen=True)
class DirectionalityAudit:
    gate_is_symmetric: bool
    relation_direction_enters_gate: bool
    direction_enters_schedule: bool
    recovered_object_description: str


def audit_directionality() -> DirectionalityAudit:
    """Declare what the current partial-SWAP process can and cannot encode."""
    return DirectionalityAudit(
        gate_is_symmetric=True,
        relation_direction_enters_gate=False,
        direction_enters_schedule=True,
        recovered_object_description="directed topology plus schedule-mediated execution structure",
    )


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
