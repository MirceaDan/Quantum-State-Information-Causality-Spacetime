"""Hard scientific gate for the baseline phase.

Q and Delta_munu are deliberately absent from the baseline dynamics. Any future
extension must be introduced in a separate phase after the baseline experiment
has passed its residual and leakage checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BaselineGate:
    q_norm: float = 0.0
    delta_norm: float = 0.0
    speculative_enabled: bool = False

    def validate(self) -> None:
        if self.q_norm != 0.0:
            raise ValueError("Q must be exactly zero in the baseline phase")
        if self.delta_norm != 0.0:
            raise ValueError("Delta_munu must be exactly zero in the baseline phase")
        if self.speculative_enabled:
            raise ValueError("speculative sectors are frozen in the baseline phase")


def assert_baseline_only(q: np.ndarray, delta: np.ndarray) -> BaselineGate:
    q_norm = float(np.linalg.norm(np.asarray(q)))
    delta_norm = float(np.linalg.norm(np.asarray(delta)))
    if q_norm != 0.0:
        raise ValueError("Q must be exactly zero in the baseline phase")
    if delta_norm != 0.0:
        raise ValueError("Delta_munu must be exactly zero in the baseline phase")
    return BaselineGate()


def assert_baseline_config(config: dict[str, object]) -> BaselineGate:
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("baseline config must contain a model mapping")
    if model.get("Q") != 0:
        raise ValueError("config model.Q must be exactly zero in the baseline phase")
    if model.get("Delta_munu") != 0:
        raise ValueError("config model.Delta_munu must be exactly zero in the baseline phase")
    if model.get("speculative_Q_enabled") is not False or model.get("speculative_Delta_enabled") is not False:
        raise ValueError("speculative sectors must be disabled in the baseline phase")
    return BaselineGate()
