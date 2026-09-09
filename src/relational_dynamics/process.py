"""Fixed-order process abstraction; no causal reversal is inferred."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

MATHEMATICAL_DEFINITION = "MATHEMATICAL_DEFINITION"


@dataclass(frozen=True)
class FixedOrderProcess:
    operations: tuple[Callable[[np.ndarray], np.ndarray], ...]
    component_label: str = MATHEMATICAL_DEFINITION

    def apply(self, state: np.ndarray) -> np.ndarray:
        result = np.asarray(state)
        for operation in self.operations:
            result = operation(result)
        return result
