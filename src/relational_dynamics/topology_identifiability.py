"""Milestone 5, section 19: the strong identifiability experiment explicitly missing
from Reverse-Theseus. For two NON-ISOMORPHIC topologies, search a finite bounded
parameter family for whether their observable bundles can be made to coincide.

Terminology (do not overpromise): a result of ``OBSERVABLY_INDISTINGUISHABLE`` is
only a statement about the explicitly tested parameter family and tolerance below,
not a general identifiability theorem. ``IDENTIFIABLE`` is never used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .causal_topology import CausalTopology, is_isomorphic
from .topology_observables import ProcessParameters, build_observable_bundle
from .topology_reconstruction import ALL_OBSERVABLES, _reference_node_order, isomorphism_aware_distance

OBSERVABLY_DISTINCT = "OBSERVABLY_DISTINCT"
OBSERVABLY_INDISTINGUISHABLE = "OBSERVABLY_INDISTINGUISHABLE"


def _random_parameters(topology: CausalTopology, rng: np.random.Generator) -> ProcessParameters:
    relation_count = len(topology.directed_relations)
    node_count = len(topology.nodes)
    strengths = tuple(float(value) for value in rng.uniform(0.05, np.pi / 4, size=relation_count))
    angles = tuple(float(value) for value in rng.uniform(0.0, 2 * np.pi, size=node_count))
    noise = tuple(float(value) for value in rng.uniform(0.0, 0.2, size=node_count))
    retention = float(rng.uniform(0.5, 1.0))
    return ProcessParameters(strengths, angles, noise, retention)


@dataclass(frozen=True)
class IdentifiabilityReport:
    topology_a_class: str
    topology_b_class: str
    best_distance: float
    tolerance: float
    trials: int
    mode: str
    classification: str
    note: str = (
        "OBSERVABLY_INDISTINGUISHABLE is relative to the explicitly tested bounded "
        "parameter family (strengths in [0.05, pi/4], retention in [0.5, 1.0], random "
        "local initial-state/noise parameters) and the stated tolerance ONLY; it is not "
        "a general inverse-identifiability theorem."
    )


def run_identifiability_test(
    topology_a: CausalTopology,
    topology_b: CausalTopology,
    tolerance: float = 0.05,
    trials: int = 30,
    seed: int = 0,
    mode: str = ALL_OBSERVABLES,
) -> IdentifiabilityReport:
    if is_isomorphic(topology_a, topology_b):
        raise ValueError("the identifiability test requires two NON-isomorphic topologies")
    node_order_a = _reference_node_order(topology_a)
    node_order_b = _reference_node_order(topology_b)
    rng = np.random.default_rng(seed)
    best_distance = float("inf")
    for _ in range(trials):
        params_a = _random_parameters(topology_a, rng)
        params_b = _random_parameters(topology_b, rng)
        bundle_a = build_observable_bundle(topology_a, params_a, node_order_a)
        bundle_b = build_observable_bundle(topology_b, params_b, node_order_b)
        best_distance = min(best_distance, isomorphism_aware_distance(bundle_a, bundle_b, mode))
    classification = OBSERVABLY_INDISTINGUISHABLE if best_distance <= tolerance else OBSERVABLY_DISTINCT
    return IdentifiabilityReport(topology_a.topology_class, topology_b.topology_class, best_distance, tolerance, trials, mode, classification)
