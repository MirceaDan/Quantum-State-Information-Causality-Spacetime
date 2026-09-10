"""Analytical synthetic worlds and independent quantum/process observables."""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import numpy as np

from .gr import DifferentialGeometry, conformal_geometry, spatial_conformal_geometry, minkowski_geometry
from .quantum import QuantumSystem


@dataclass(frozen=True)
class GroundTruthWorld:
    world_id: str
    geometry: DifferentialGeometry
    coordinates: np.ndarray
    invariant_intervals: np.ndarray
    pair_classifications: dict[tuple[int, int], str]
    causal_relation: np.ndarray
    information_matrix: np.ndarray
    process_causal_relation: np.ndarray
    geometry_hidden_from_observables: bool = True

    def observables(self) -> dict[str, np.ndarray]:
        return {
            "information_matrix": self.information_matrix.copy(),
            "causal_matrix": self.process_causal_relation.copy(),
        }


@dataclass(frozen=True)
class RandomizedObservables:
    information_matrix: np.ndarray
    causal_matrix: np.ndarray
    control_type: str


def _process_information(event_count: int, persistence: float = 0.85) -> np.ndarray:
    probabilities = []
    for bits in itertools.product((0, 1), repeat=event_count):
        probability = 0.5
        for first, second in zip(bits, bits[1:]):
            probability *= persistence if first == second else 1.0 - persistence
        probabilities.append(probability)
    system = QuantumSystem.from_density_matrix(np.diag(probabilities).astype(complex), (2,) * event_count)
    information = np.zeros((event_count, event_count), dtype=float)
    for first in range(event_count):
        for second in range(first + 1, event_count):
            information[first, second] = information[second, first] = system.mutual_information((first,), (second,))
    return information


def _process_causal_chain(event_count: int) -> np.ndarray:
    causal = np.zeros((event_count, event_count), dtype=bool)
    for source in range(event_count - 1):
        causal[source, source + 1] = True
    return causal


def _world(
    world_id: str,
    geometry: DifferentialGeometry,
    coordinates: np.ndarray,
    information: np.ndarray,
    process_causal: np.ndarray,
) -> GroundTruthWorld:
    event_count = len(coordinates)
    intervals = np.zeros((event_count, event_count), dtype=float)
    classifications: dict[tuple[int, int], str] = {}
    causal = np.zeros((event_count, event_count), dtype=bool)
    for first in range(event_count):
        for second in range(first + 1, event_count):
            midpoint = (coordinates[first] + coordinates[second]) / 2
            difference = coordinates[first] - coordinates[second]
            metric = geometry.metric(midpoint)
            interval = float(difference @ metric @ difference)
            intervals[first, second] = intervals[second, first] = interval
            if interval < -1e-10:
                label = "timelike"
            elif abs(interval) <= 1e-10:
                label = "null"
            else:
                label = "spacelike"
            classifications[(first, second)] = label
            if coordinates[second, 0] > coordinates[first, 0] and interval <= 1e-10:
                causal[first, second] = True
    return GroundTruthWorld(world_id, geometry, coordinates, intervals, classifications, causal, information.copy(), process_causal.copy())


def build_ground_truth_worlds() -> list[GroundTruthWorld]:
    coordinates = np.array(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]],
        dtype=float,
    )
    information = _process_information(len(coordinates))
    process_causal = _process_causal_chain(len(coordinates))
    return [
        _world("WORLD_1", minkowski_geometry(), coordinates, information, process_causal),
        _world("WORLD_2", conformal_geometry(scale=0.2), coordinates, information, process_causal),
        _world("WORLD_3", spatial_conformal_geometry(scale=0.15), coordinates, information, process_causal),
    ]


def randomized_observables(world: GroundTruthWorld, seed: int = 0) -> RandomizedObservables:
    rng = np.random.default_rng(seed)
    information = world.information_matrix.copy()
    upper = np.triu_indices_from(information, k=1)
    values = information[upper].copy()
    rng.shuffle(values)
    randomized_information = np.zeros_like(information)
    randomized_information[upper] = values
    randomized_information[(upper[1], upper[0])] = values
    causal = world.process_causal_relation.copy()
    source_order = rng.permutation(len(causal))
    randomized_causal = causal[source_order][:, source_order]
    return RandomizedObservables(randomized_information, randomized_causal, "pairwise_permutation_control")
