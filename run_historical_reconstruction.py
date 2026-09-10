"""Run the Reverse-Theseus historical-geometry-reconstruction experiment suite.

Writes machine-readable results distinguishing EXP-HIST-NULL-001 (preserved
independent-observable null baseline), EXP-HIST-COUPLED-001 (geometry-coupled
positive control), EXP-HIST-RETENTION-001 (information-retention sweep),
EXP-HIST-IDENT-001 (identifiability probe), and EXP-HIST-RANDOM-001
(randomized control on the coupled observables). Ground truth is used only to
evaluate reconstructions after fitting; it is never given to the fit.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from relational_dynamics.ground_truth import build_ground_truth_worlds
from relational_dynamics.historical_reconstruction import (
    classify_historical_reconstruction,
    run_coupled_experiment,
    run_identifiability_test,
    run_information_retention_curve,
    run_null_baseline_experiment,
    run_randomized_control_experiment,
)

SEEDS = tuple(range(10))
RETENTION_ETAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): value for key, value in value.items()}
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _report_to_dict(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _report_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _report_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_to_dict(item) for item in value]
    return value


def main() -> None:
    worlds = build_ground_truth_worlds()
    world_by_id = {world.world_id: world for world in worlds}

    null_report = run_null_baseline_experiment(seed=1234, seeds=SEEDS)

    coupled_reports = {world.world_id: run_coupled_experiment(world, seeds=SEEDS) for world in worlds}
    randomized_reports = {world.world_id: run_randomized_control_experiment(world, seeds=SEEDS) for world in worlds}
    retention_curves = {world.world_id: run_information_retention_curve(world, etas=RETENTION_ETAS, seed=1234) for world in worlds}

    identifiability_reports = []
    world_ids = [world.world_id for world in worlds]
    for first in range(len(world_ids)):
        for second in range(first + 1, len(world_ids)):
            identifiability_reports.append(run_identifiability_test(world_by_id[world_ids[first]], world_by_id[world_ids[second]]))

    classifications = {
        world.world_id: classify_historical_reconstruction(
            coupled_reports[world.world_id],
            randomized_reports[world.world_id],
            identifiability_reports,
            retention_curves[world.world_id],
        )
        for world in worlds
    }

    output = {
        "EXP-HIST-NULL-001": _report_to_dict(null_report),
        "EXP-HIST-COUPLED-001": {world_id: _report_to_dict(report) for world_id, report in coupled_reports.items()},
        "EXP-HIST-RANDOM-001": {world_id: _report_to_dict(report) for world_id, report in randomized_reports.items()},
        "EXP-HIST-RETENTION-001": {world_id: _report_to_dict(curve) for world_id, curve in retention_curves.items()},
        "EXP-HIST-IDENT-001": [_report_to_dict(report) for report in identifiability_reports],
        "classification": classifications,
        "q_norm": 0.0,
        "delta_norm": 0.0,
        "claims_new_physics": False,
        "claims_time_travel": False,
    }

    result_root = ROOT / "results"
    historical_root = result_root / "historical"
    historical_root.mkdir(parents=True, exist_ok=True)
    (historical_root / "historical_reconstruction_results.json").write_text(json.dumps(output, indent=2, default=json_default), encoding="utf-8")

    print(json.dumps(classifications, indent=2))
    for world_id, report in coupled_reports.items():
        flexible = report.seed_statistics["FLEXIBLE"]
        restricted = report.seed_statistics["RESTRICTED"]
        randomized_flexible = randomized_reports[world_id].seed_statistics["FLEXIBLE"]
        print(
            f"{world_id}: FLEXIBLE test_loss mean={flexible.mean:.4f} std={flexible.std:.4f} | "
            f"RESTRICTED mean={restricted.mean:.4f} | RANDOMIZED(FLEXIBLE) mean={randomized_flexible.mean:.4f} | "
            f"causal_f1={report.baselines['FLEXIBLE'].causal_f1:.3f} curvature_error={report.baselines['FLEXIBLE'].curvature_invariant_error:.4f}"
        )
    for report in identifiability_reports:
        print(f"IDENT {report.world_a} vs {report.world_b}: distance={report.observable_distance:.6f} -> {report.classification}")


if __name__ == "__main__":
    main()
