"""Baseline geometry reconstruction from independent quantum/process observables."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np
from scipy.optimize import least_squares

from .geometry import LorentzianEmbedding
from .quantum import QuantumSystem
from .speculative_gate import BaselineGate


@dataclass(frozen=True)
class GeometryReconstructionReport:
    classification: str
    geometry_was_hidden: bool
    observation_source: str
    parameter_count: int
    train_loss: float
    validation_loss: float
    test_loss: float
    causal_precision: float
    causal_recall: float
    coordinate_relative_error: float
    q_norm: float
    delta_norm: float
    claims_new_physics: bool = False


@dataclass(frozen=True)
class MultiSeedReconstructionReport:
    seed_count: int
    mean_test_loss: float
    std_test_loss: float
    median_test_loss: float
    best_test_loss: float
    worst_test_loss: float
    per_seed: tuple[float, ...]


def _markov_density_matrix(event_count: int, persistence: float = 0.85) -> np.ndarray:
    probabilities = []
    for bits in itertools.product((0, 1), repeat=event_count):
        probability = 0.5
        for first, second in zip(bits, bits[1:]):
            probability *= persistence if first == second else 1.0 - persistence
        probabilities.append(probability)
    return np.diag(np.asarray(probabilities, dtype=float)).astype(complex)


def _independent_observables(event_count: int) -> tuple[np.ndarray, np.ndarray]:
    system = QuantumSystem.from_density_matrix(_markov_density_matrix(event_count), (2,) * event_count)
    information = np.zeros((event_count, event_count), dtype=float)
    for first in range(event_count):
        for second in range(first + 1, event_count):
            information[first, second] = information[second, first] = system.mutual_information((first,), (second,))
    causal = np.zeros((event_count, event_count), dtype=bool)
    for source in range(event_count - 1):
        causal[source, source + 1] = True
    return information, causal


def _fit_from_observables(information: np.ndarray, causal: np.ndarray, seed: int) -> tuple[np.ndarray, float, float, float, float]:
    event_count = len(information)
    rng = np.random.default_rng(seed)
    max_information = max(float(np.max(information)), 1e-12)
    dissimilarity = -np.log((information + 1e-12) / (max_information + 1e-12))
    pairs = [(first, second) for first in range(event_count) for second in range(first + 1, event_count)]
    rng.shuffle(pairs)
    train_end = max(1, int(0.6 * len(pairs)))
    validation_end = max(train_end + 1, int(0.8 * len(pairs)))
    train_pairs = pairs[:train_end]
    validation_pairs = pairs[train_end:validation_end]
    test_pairs = pairs[validation_end:]
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])

    def residuals(flat: np.ndarray, selected_pairs: list[tuple[int, int]]) -> np.ndarray:
        coordinates = flat.reshape(event_count, 4)
        values = []
        for first, second in selected_pairs:
            difference = coordinates[first] - coordinates[second]
            separation = float(difference @ metric @ difference)
            values.append(np.sqrt(abs(separation) + 1e-12) - dissimilarity[first, second])
        for source, target in zip(*np.where(causal)):
            difference = coordinates[source] - coordinates[target]
            separation = float(difference @ metric @ difference)
            values.append(max(0.0, separation))
            values.append(max(0.0, coordinates[source, 0] - coordinates[target, 0]))
        values.extend(coordinates[0].tolist())
        return np.asarray(values, dtype=float)

    initial = rng.normal(scale=0.2, size=(event_count, 4))
    initial[:, 0] = np.arange(event_count, dtype=float) + rng.normal(scale=0.05, size=event_count)
    initial[0] = 0.0
    fitted = least_squares(lambda values: residuals(values, train_pairs), initial.ravel(), max_nfev=500).x.reshape(event_count, 4)

    def pair_loss(selected_pairs: list[tuple[int, int]]) -> float:
        values = residuals(fitted.ravel(), selected_pairs)
        return float(np.mean(values[: len(selected_pairs)] ** 2)) if selected_pairs else 0.0

    return fitted, pair_loss(train_pairs), pair_loss(validation_pairs), pair_loss(test_pairs), max_information


def run_baseline_geometry_reconstruction(seed: int = 1234) -> GeometryReconstructionReport:
    """Run the baseline only; ground-truth coordinates are evaluation-only data."""
    BaselineGate().validate()
    event_count = 6
    true_coordinates = np.column_stack((np.arange(event_count, dtype=float), np.zeros((event_count, 3))))
    information, causal = _independent_observables(event_count)
    return evaluate_observable_reconstruction(information, causal, true_coordinates, seed, "independent_quantum_markov_process")


def evaluate_observable_reconstruction(
    information: np.ndarray,
    causal: np.ndarray,
    true_coordinates: np.ndarray,
    seed: int,
    observation_source: str,
) -> GeometryReconstructionReport:
    """Fit only observable matrices; coordinates are evaluation-only inputs."""
    event_count = len(information)
    fitted, train_loss, validation_loss, test_loss, _ = _fit_from_observables(information, causal, seed)
    embedding = LorentzianEmbedding(fitted)
    metrics = embedding.compare_causal_relations(causal)
    scale = max(float(np.linalg.norm(true_coordinates)), 1e-12)
    coordinate_error = float(np.linalg.norm(fitted - true_coordinates) / scale)
    valid = test_loss < 0.25 and coordinate_error < 0.25 and metrics["precision"] >= 0.8 and metrics["recall"] >= 0.8
    return GeometryReconstructionReport(
        "BASELINE_GEOMETRY_RECONSTRUCTION_VALID" if valid else "BASELINE_GEOMETRY_RECONSTRUCTION_FAILED",
        True,
        observation_source,
        event_count * 4,
        train_loss,
        validation_loss,
        test_loss,
        float(metrics["precision"]),
        float(metrics["recall"]),
        coordinate_error,
        0.0,
        0.0,
    )


def run_multi_seed_reconstruction(seeds: object = range(10)) -> MultiSeedReconstructionReport:
    losses = tuple(run_baseline_geometry_reconstruction(int(seed)).test_loss for seed in seeds)
    values = np.asarray(losses, dtype=float)
    return MultiSeedReconstructionReport(
        len(losses),
        float(np.mean(values)),
        float(np.std(values)),
        float(np.median(values)),
        float(np.min(values)),
        float(np.max(values)),
        losses,
    )
