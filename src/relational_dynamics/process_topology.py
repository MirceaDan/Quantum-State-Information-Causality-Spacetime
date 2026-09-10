"""Process topology abstraction for Milestone 4 (permutation-equivariant process memory).

INSPECTION NOTE (required by the M4 spec, section 1): the prior Reverse-Theseus
implementation in ``historical_process.py`` hard-codes raw-index-dependent
physics in exactly the places the M4 root-cause analysis flagged:

- ``state_vector = zero if 0 % 2 == 0 else one`` / ``zero if index % 2 == 0
  else one`` -- index-PARITY initial state (label_confund).
- ``step_indices = list(range(event_count - 1))`` and the coupling loop that
  always builds relations ``(0,1),(1,2),...`` -- FIXED INDEX ADJACENCY
  regardless of any declared topology (process_structure_confund).
- ``build_observable_split`` (in ``historical_reconstruction.py``) enumerates
  pairs via ``[(first, second) for first in range(event_count) for second in
  range(first + 1, event_count)]`` and shuffles that LIST -- the split position
  is tied to raw pair-enumeration order (split_design_confund).

Milestone 4 does not modify any of that code. It is a clearly separated new
module/path. The topology objects here store relations as EXPLICIT DATA
(labelled pairs plus a declared execution-order attribute), never regenerated
from ``range(N-1)``/``i % 2`` during simulation or permutation, so that
``permute_process`` is a pure relabeling and never a re-derivation of physics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools

MATHEMATICAL_DEFINITION = "MATHEMATICAL_DEFINITION"
MODELING_ASSUMPTION = "MODELING_ASSUMPTION"


@dataclass(frozen=True)
class NodeAttributes:
    """Per-node physical parameters. Attached to a LABEL, never to a raw tensor position."""

    dimension: int = 2
    initial_state_angle: float = 0.0  # Bloch polar angle; a declared per-node parameter, not derived from the label
    noise_parameter: float = 0.0  # local depolarizing probability contribution, in [0, 1]
    component_label: str = MODELING_ASSUMPTION


@dataclass(frozen=True)
class Relation:
    """An explicit, declared physical interaction between two labelled nodes.

    ``order`` is an EXPLICIT temporal-schedule tag supplied at construction
    time (part of the physical process description); simulation sorts by this
    field, never by Python list position, so relabeling never reorders the
    schedule.
    """

    first: int
    second: int
    strength: float
    order: int
    component_label: str = MODELING_ASSUMPTION

    def relabel(self, permutation: tuple[int, ...]) -> "Relation":
        return replace(self, first=permutation[self.first], second=permutation[self.second])


@dataclass(frozen=True)
class ProcessTopology:
    """node identity != node label: this object separates WHICH nodes exist and
    HOW they relate (the physical content) from the integer names attached to
    them (bookkeeping only). ``node_labels`` is stored as a ``frozenset``-like
    sorted tuple purely for reproducible serialization; iteration order over it
    must never be treated as physically meaningful."""

    node_labels: tuple[int, ...]
    node_attributes: dict[int, NodeAttributes]
    relations: tuple[Relation, ...]
    temporal_relations: tuple[tuple[int, int], ...] = ()
    component_label: str = MODELING_ASSUMPTION

    def __post_init__(self) -> None:
        if set(self.node_attributes) != set(self.node_labels):
            raise ValueError("node_attributes must be keyed by exactly the node_labels")
        for relation in self.relations:
            if relation.first not in self.node_labels or relation.second not in self.node_labels:
                raise ValueError("relation endpoints must be declared node labels")
        object.__setattr__(self, "node_labels", tuple(sorted(self.node_labels)))


def permute_process(topology: ProcessTopology, permutation: tuple[int, ...]) -> ProcessTopology:
    """Relabel every field of ``topology`` by ``permutation`` (old label -> new label).

    This RENAMES the physical content; it does not move, reorder, resimulate,
    or reinterpret it. ``permutation`` must be a bijection on ``topology.node_labels``.
    """
    labels = topology.node_labels
    if sorted(permutation[label] for label in labels) != sorted(labels):
        raise ValueError("permutation must be a bijection on the topology's node labels")
    new_labels = tuple(permutation[label] for label in labels)
    new_attributes = {permutation[label]: attributes for label, attributes in topology.node_attributes.items()}
    new_relations = tuple(relation.relabel(permutation) for relation in topology.relations)
    new_temporal = tuple((permutation[first], permutation[second]) for first, second in topology.temporal_relations)
    return ProcessTopology(new_labels, new_attributes, new_relations, new_temporal)


def canonicalize_process(topology: ProcessTopology) -> tuple[ProcessTopology, tuple[int, ...]]:
    """OPTIONAL canonical relabeling by a permutation-INVARIANT sort key (never by raw label value).

    Provided only as a convenience for deduplicating physically-identical
    topologies; it is NOT used as a substitute for the equivariance tests,
    which must compare simulated observables directly.
    """
    labels = topology.node_labels

    def _signature(label: int) -> tuple:
        attributes = topology.node_attributes[label]
        neighbor_strengths = sorted(
            relation.strength for relation in topology.relations if label in (relation.first, relation.second)
        )
        degree = len(neighbor_strengths)
        return (attributes.dimension, attributes.initial_state_angle, attributes.noise_parameter, degree, tuple(neighbor_strengths))

    ordered_labels = sorted(labels, key=_signature)
    permutation = [0] * len(labels)
    for new_label, old_label in enumerate(ordered_labels):
        permutation[old_label] = new_label
    return permute_process(topology, tuple(permutation)), tuple(permutation)


def line_topology(
    labels: tuple[int, ...],
    strengths: tuple[float, ...],
    initial_state_angles: tuple[float, ...] | None = None,
    noise_parameters: tuple[float, ...] | None = None,
) -> ProcessTopology:
    """A declared path topology: ``labels[0]--labels[1]--...``.

    The ORDER of ``labels`` here is a one-time DECLARATION of which physical
    subsystems are connected to which (the topology itself); it is stored as
    explicit ``Relation`` data and is never regenerated from index arithmetic
    afterwards. Relabeling this topology (``permute_process``) changes the
    LABELS attached to each physical node; it never changes which physical
    nodes are connected.
    """
    if len(strengths) != len(labels) - 1:
        raise ValueError("a line topology needs exactly len(labels)-1 strengths")
    angles = initial_state_angles if initial_state_angles is not None else tuple(0.0 for _ in labels)
    noise = noise_parameters if noise_parameters is not None else tuple(0.0 for _ in labels)
    node_attributes = {label: NodeAttributes(2, angle, noise_value) for label, angle, noise_value in zip(labels, angles, noise)}
    relations = tuple(
        Relation(labels[index], labels[index + 1], strengths[index], order=index) for index in range(len(labels) - 1)
    )
    return ProcessTopology(tuple(labels), node_attributes, relations)


def complete_topology(
    labels: tuple[int, ...],
    relation_specs: tuple[tuple[int, int, float, int], ...],
    initial_state_angles: tuple[float, ...] | None = None,
    noise_parameters: tuple[float, ...] | None = None,
) -> ProcessTopology:
    """Every pair of declared labels is related.

    ``relation_specs`` is a fully explicit ``(first, second, strength, order)``
    tuple supplied by the caller for every pair. Unlike an earlier draft of
    this function, ``order`` is never derived from ``sorted(labels)`` or from
    ``itertools.combinations`` enumeration order -- deriving a physical
    schedule from sorted integer label values is exactly the kind of hidden
    permutation-breaking dependency this milestone forbids (see the "hidden
    permutation breaking" note in AUDIT.md).
    """
    angles = initial_state_angles if initial_state_angles is not None else tuple(0.0 for _ in labels)
    noise = noise_parameters if noise_parameters is not None else tuple(0.0 for _ in labels)
    node_attributes = {label: NodeAttributes(2, angle, noise_value) for label, angle, noise_value in zip(labels, angles, noise)}
    expected_pairs = {frozenset(pair) for pair in itertools.combinations(labels, 2)}
    supplied_pairs = {frozenset({first, second}) for first, second, _, _ in relation_specs}
    if supplied_pairs != expected_pairs:
        raise ValueError("relation_specs must supply exactly one relation per unordered label pair")
    relations = tuple(Relation(first, second, strength, order) for first, second, strength, order in relation_specs)
    return ProcessTopology(tuple(labels), node_attributes, relations)
