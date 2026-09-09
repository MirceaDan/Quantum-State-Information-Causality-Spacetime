"""State reconstruction and reversibility baselines, explicitly not time travel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .quantum import QuantumSystem

STATE_RECONSTRUCTION = "STATE_RECONSTRUCTION"


@dataclass(frozen=True)
class QuantumChannel:
    apply: Callable[[np.ndarray], np.ndarray]
    name: str
    component_label: str = "ESTABLISHED_PHYSICS"


def unitary_channel(unitary: np.ndarray) -> QuantumChannel:
    unitary = np.asarray(unitary, dtype=complex)
    return QuantumChannel(lambda rho: unitary @ rho @ unitary.conj().T, "unitary channel")


def depolarizing_channel(probability: float) -> QuantumChannel:
    if not 0 <= probability <= 1:
        raise ValueError("depolarizing probability must be in [0, 1]")

    def apply(rho: np.ndarray) -> np.ndarray:
        dimension = rho.shape[0]
        return (1 - probability) * rho + probability * np.eye(dimension) / dimension

    return QuantumChannel(apply, f"depolarizing channel p={probability}")


@dataclass(frozen=True)
class ReconstructionReport:
    forward_fidelity: float
    reverse_fidelity: float
    round_trip_loss: float
    label: str = STATE_RECONSTRUCTION
    temporal_reversal_claim: bool = False


def reconstruct_state(
    rho_a: np.ndarray,
    forward: QuantumChannel,
    reverse: QuantumChannel,
) -> ReconstructionReport:
    rho_b = forward.apply(rho_a)
    reconstructed_a = reverse.apply(rho_b)
    round_trip_b = forward.apply(reconstructed_a)
    forward_fidelity = QuantumSystem.fidelity(rho_b, forward.apply(rho_a))
    reverse_fidelity = QuantumSystem.fidelity(rho_a, reconstructed_a)
    round_trip_loss = max(0.0, QuantumSystem.relative_entropy(rho_b, round_trip_b))
    return ReconstructionReport(forward_fidelity, reverse_fidelity, round_trip_loss)
