"""N-event scaling-study worlds for the Reverse-Theseus integrity audit (section 6).

Reuses the EXACT same construction recipe as ``ground_truth.build_ground_truth_worlds``
(same three geometry families, same information/process-causal generators) at
different relational-event counts. The canonical WORLD_1/2/3 (fixed at N=4) are
untouched; this module only adds new same-family worlds for a scaling study.

Computational note: the dense qubit simulation used by ``historical_process.py``
scales as 4**event_count in memory and worse in time (kron/matrix-multiply
costs), so only a feasible subset of event counts is actually executed by the
orchestration scripts; see docs/LIMITATIONS.md.
"""

from __future__ import annotations

import numpy as np

from .gr import conformal_geometry, minkowski_geometry, spatial_conformal_geometry
from .ground_truth import GroundTruthWorld, _process_causal_chain, _process_information, _world

SCALING_EVENT_COUNTS = (4, 6, 8, 12, 20)
FEASIBLE_SCALING_EVENT_COUNTS = (4, 6, 8)


def _scaled_coordinates(event_count: int) -> np.ndarray:
    """Same style as the canonical worlds: event 0 at t=0, the rest at t=1 fanning out in x."""
    coordinates = np.zeros((event_count, 4), dtype=float)
    coordinates[1:, 0] = 1.0
    coordinates[1:, 1] = np.arange(event_count - 1, dtype=float)
    return coordinates


def build_scaled_worlds(event_count: int) -> list[GroundTruthWorld]:
    """Same three geometry families and coupling law as the canonical worlds, at ``event_count`` events."""
    if event_count < 2:
        raise ValueError("scaling worlds need at least two events")
    coordinates = _scaled_coordinates(event_count)
    information = _process_information(event_count)
    process_causal = _process_causal_chain(event_count)
    return [
        _world(f"WORLD_1_N{event_count}", minkowski_geometry(), coordinates, information, process_causal),
        _world(f"WORLD_2_N{event_count}", conformal_geometry(scale=0.2), coordinates, information, process_causal),
        _world(f"WORLD_3_N{event_count}", spatial_conformal_geometry(scale=0.15), coordinates, information, process_causal),
    ]
