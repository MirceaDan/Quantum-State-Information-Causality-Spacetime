"""Milestone 4 runner: permutation-equivariant quantum process-memory validation.

Executes: baseline process -> relabelings -> equivariance tests -> memory
observables -> broken negative control -> scale tests -> gate -> results.json.

This module does NOT attempt geometry, Lorentzian intervals, GR, or CTCs.
Q = 0 and Delta_munu = 0 throughout.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from relational_dynamics.broken_process_control import check_broken_state_equivariance
from relational_dynamics.permutation_equivariant_process import simulate_process
from relational_dynamics.process_memory_observables import (
    check_global_scalar_invariants,
    check_intervention_tensor_equivariance,
    check_memory_score_invariance,
    check_multipartite_observable_equivariance,
    check_pairwise_observable_equivariance,
    check_state_equivariance,
    process_memory_score,
)
from relational_dynamics.process_topology import line_topology, permute_process

SEED = 1234
IMPLEMENTED_AND_VALIDATED = "IMPLEMENTED_AND_VALIDATED"
IMPLEMENTED_BUT_NOT_FULLY_VALIDATED = "IMPLEMENTED_BUT_NOT_FULLY_VALIDATED"
TOLERANCE = 1e-8


def _to_jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_topology(event_count: int) -> "ProcessTopology":
    labels = tuple(range(event_count))
    rng = np.random.default_rng(SEED)
    strengths = tuple(float(value) for value in rng.uniform(0.1, np.pi / 4, size=event_count - 1))
    angles = tuple(float(value) for value in rng.uniform(0.0, 2 * np.pi, size=event_count))
    noise = tuple(float(value) for value in rng.uniform(0.0, 0.3, size=event_count))
    return line_topology(labels, strengths, angles, noise)


def _permutations_for(event_count: int, permutation_seed: int, max_random: int = 20) -> list[tuple[int, ...]]:
    """Exhaustive for small N; random sample otherwise (section 12)."""
    if event_count <= 5:
        return [permutation for permutation in itertools.permutations(range(event_count))]
    rng = np.random.default_rng(permutation_seed)
    return [tuple(int(index) for index in rng.permutation(event_count)) for _ in range(max_random)]


def run_equivariance_suite(topology, permutations: list[tuple[int, ...]]) -> dict:
    checks = {"state": [], "pairwise": [], "multipartite": [], "intervention": [], "memory": [], "globals": []}
    for permutation in permutations:
        checks["state"].append(check_state_equivariance(topology, permutation, tolerance=TOLERANCE))
        checks["pairwise"].append(check_pairwise_observable_equivariance(topology, permutation, tolerance=TOLERANCE))
        checks["multipartite"].append(check_multipartite_observable_equivariance(topology, permutation, tolerance=TOLERANCE))
        checks["intervention"].append(check_intervention_tensor_equivariance(topology, permutation, tolerance=TOLERANCE))
        checks["memory"].append(check_memory_score_invariance(topology, permutation, tolerance=TOLERANCE))
        for label, check in check_global_scalar_invariants(topology, permutation, tolerance=TOLERANCE).items():
            checks["globals"].append(check)

    summary = {}
    all_errors = []
    all_passed = True
    for category, results in checks.items():
        errors = [result.max_error for result in results]
        all_errors.extend(errors)
        category_passed = all(result.passed for result in results)
        all_passed = all_passed and category_passed
        summary[category] = {
            "max_error": float(max(errors)) if errors else 0.0,
            "mean_error": float(np.mean(errors)) if errors else 0.0,
            "passed": category_passed,
        }
    summary["overall_pass"] = all_passed
    summary["overall_max_error"] = float(max(all_errors)) if all_errors else 0.0
    summary["overall_mean_error"] = float(np.mean(all_errors)) if all_errors else 0.0
    summary["num_permutations"] = len(permutations)
    return summary


def run_negative_control(topology, permutations: list[tuple[int, ...]]) -> dict:
    non_identity = [permutation for permutation in permutations if permutation != tuple(range(len(topology.node_labels)))]
    failures = [check_broken_state_equivariance(topology, permutation, tolerance=TOLERANCE) for permutation in non_identity]
    detected_failure = any(not result.passed for result in failures)
    return {
        "pass_expected_failure": detected_failure,
        "num_permutations_tested": len(non_identity),
        "max_error_observed": float(max(result.max_error for result in failures)) if failures else 0.0,
    }


def run_scaling(event_counts: tuple[int, ...] = (3, 4, 5)) -> list[dict]:
    points = []
    for event_count in event_counts:
        topology = _build_topology(event_count)
        permutations = _permutations_for(event_count, permutation_seed=SEED)
        start = time.perf_counter()
        equivariance = run_equivariance_suite(topology, permutations)
        elapsed = time.perf_counter() - start
        memory_value = process_memory_score(topology)
        points.append(
            {
                "event_count": event_count,
                "hilbert_dimension": 2**event_count,
                "num_permutations_tested": equivariance["num_permutations"],
                "max_equivariance_error": equivariance["overall_max_error"],
                "mean_equivariance_error": equivariance["overall_mean_error"],
                "process_memory_score": memory_value,
                "runtime_seconds": elapsed,
            }
        )
    return points


def main() -> None:
    existing_suite_pass = "--existing-suite-passed" in sys.argv

    topology = _build_topology(4)
    permutations = _permutations_for(4, permutation_seed=SEED)

    equivariance = run_equivariance_suite(topology, permutations)
    negative_control = run_negative_control(topology, permutations)
    memory_score = process_memory_score(topology)
    scaling = run_scaling()

    reproducibility_pass = simulate_process(topology).final_system.rho.tobytes() == simulate_process(topology).final_system.rho.tobytes()

    m4_equivariance_pass = equivariance["overall_pass"]
    m4_negative_control_pass = negative_control["pass_expected_failure"]
    m4_reproducibility_pass = bool(reproducibility_pass)

    gate = {
        "M4_EQUIVARIANCE_PASS": m4_equivariance_pass,
        "M4_NEGATIVE_CONTROL_PASS": m4_negative_control_pass,
        "M4_REPRODUCIBILITY_PASS": m4_reproducibility_pass,
        "EXISTING_TEST_SUITE_PASS": existing_suite_pass,
    }
    gate_pass_known = m4_equivariance_pass and m4_negative_control_pass and m4_reproducibility_pass and existing_suite_pass

    classification = IMPLEMENTED_AND_VALIDATED if gate_pass_known else IMPLEMENTED_BUT_NOT_FULLY_VALIDATED

    results = {
        "milestone": "M4",
        "classification": classification,
        "claims_new_physics": False,
        "claims_emergent_spacetime": False,
        "claims_time_travel": False,
        "q_norm": 0.0,
        "delta_norm": 0.0,
        "equivariance": equivariance,
        "memory": {
            "process_memory_score": memory_score,
            "definition": "mean finite-replay intervention response R(X->Y); NOT a claim of a rigorous non-Markovianity measure",
        },
        "negative_control": negative_control,
        "scaling": scaling,
        "reproducibility": {
            "pass": m4_reproducibility_pass,
            "seed": SEED,
            "event_count": 4,
            "permutation_seed": SEED,
            "num_permutations": len(permutations),
        },
        "gate": gate,
    }

    print(json.dumps({key: results[key] for key in ("milestone", "classification", "gate")}, indent=2, default=str))
    print(f"equivariance overall_max_error={equivariance['overall_max_error']:.3e}  overall_pass={equivariance['overall_pass']}")
    print(f"negative_control pass_expected_failure={negative_control['pass_expected_failure']}  max_error_observed={negative_control['max_error_observed']:.3e}")
    for point in scaling:
        print(f"  N={point['event_count']} dim={point['hilbert_dimension']} max_err={point['max_equivariance_error']:.3e} runtime={point['runtime_seconds']:.3f}s")

    result_root = ROOT / "results" / "historical"
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "m4_permutation_equivariant_process_memory_results.json").write_text(
        json.dumps(_to_jsonable(results), indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
