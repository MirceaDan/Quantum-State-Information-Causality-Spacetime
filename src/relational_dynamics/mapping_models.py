"""Low-capacity information-to-distance model comparison."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class MappingModelReport:
    model_id: str
    parameter_count: int
    train_loss: float
    validation_loss: float
    test_loss: float
    mapping_is_physical_law: bool = False


def compare_mapping_models(information: np.ndarray, causal: np.ndarray, target_distance: np.ndarray, seed: int = 0) -> list[MappingModelReport]:
    information = np.asarray(information, dtype=float)
    causal = np.asarray(causal, dtype=float)
    target_distance = np.asarray(target_distance, dtype=float)
    count = len(information)
    pairs = [(first, second) for first in range(count) for second in range(first + 1, count)]
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    train_end = max(1, int(0.6 * len(pairs)))
    validation_end = max(train_end + 1, int(0.8 * len(pairs)))
    splits = (pairs[:train_end], pairs[train_end:validation_end], pairs[validation_end:])
    epsilon = 1e-12
    max_information = max(float(np.max(information)), epsilon)

    def features(first: int, second: int) -> np.ndarray:
        mutual = information[first, second]
        influence = causal[first, second]
        return np.array([1.0, -np.log(mutual + epsilon), influence, influence**2, mutual * influence])

    def loss(prediction: np.ndarray, selected: list[tuple[int, int]]) -> float:
        if not selected:
            return 0.0
        return float(np.mean([(prediction[first, second] - target_distance[first, second]) ** 2 for first, second in selected]))

    analytical = -np.log((information + epsilon) / (max_information + epsilon))
    reports = [MappingModelReport("NULL", 0, loss(np.zeros_like(target_distance), splits[0]), loss(np.zeros_like(target_distance), splits[1]), loss(np.zeros_like(target_distance), splits[2]))]
    reports.append(MappingModelReport("FIXED_ANALYTICAL", 0, loss(analytical, splits[0]), loss(analytical, splits[1]), loss(analytical, splits[2])))
    design = np.array([features(first, second) for first, second in splits[0]])
    targets = np.array([target_distance[first, second] for first, second in splits[0]])
    result = least_squares(lambda parameters: design @ parameters - targets, np.zeros(5), loss="soft_l1")
    prediction = np.zeros_like(target_distance)
    for first, second in pairs:
        value = float(features(first, second) @ result.x)
        prediction[first, second] = prediction[second, first] = value
    regularization = 1e-3 * float(np.sum(result.x**2))
    reports.append(MappingModelReport("F_THETA", 5, loss(prediction, splits[0]) + regularization, loss(prediction, splits[1]) + regularization, loss(prediction, splits[2]) + regularization))
    return reports
