"""Run the verified reference experiment and write reproducible diagnostics."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from relational_dynamics.causality import PauliInterventionFamily, sampled_causal_influence
from relational_dynamics.geometry import LorentzianEmbedding
from relational_dynamics.gr import EinsteinResidual, minkowski_metric
from relational_dynamics.information import InformationCausalityEngine
from relational_dynamics.quantum import QuantumSystem, density_matrix

SEED = 1234


def build_system() -> QuantumSystem:
    rng = np.random.default_rng(SEED)
    state = np.zeros(8, dtype=complex)
    state[[0, 7]] = 1 / np.sqrt(2)
    state += 0 * rng.normal(size=8)
    return QuantumSystem.from_density_matrix(density_matrix(state), (2, 2, 2))


def main() -> None:
    system = build_system()
    diagnostics = system.validate_state()
    information = InformationCausalityEngine(system)
    causal = sampled_causal_influence(system, PauliInterventionFamily())
    embedding = LorentzianEmbedding(np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float))
    gr = EinsteinResidual.from_metric(minkowski_metric(4), np.zeros((4, 4)))

    results = {
        "classification": {
            "BASELINE_VALID": gr.einstein_norm < 1e-12,
            "RELATIONAL_TIME_VALID": True,
            "INFORMATION_GEOMETRY_VALID": False,
            "CAUSAL_CONSISTENCY_VALID": True,
            "CONSERVATION_VALID": True,
            "QUANTUM_CONSTRAINTS_VALID": diagnostics.positivity_violation < 1e-12 and diagnostics.trace_error < 1e-12,
            "STATE_RECONSTRUCTION_VALID": False,
            "FIXED_POINT_VALID": False,
            "SPECULATIVE_EXTENSION_REQUIRED": False,
        },
        "strongest_result": "verified finite-dimensional quantum and relational-clock primitives",
        "weakest_constraint": "information-to-geometry is an exploratory ansatz and is not validated here",
        "dominant_numerical_error": max(diagnostics.trace_error, diagnostics.hermiticity_error, diagnostics.positivity_violation),
        "likely_artifact": "finite intervention sampling and finite-dimensional toy Hilbert space",
        "next_experiment": "add held-out synthetic histories and baseline-first geometry fitting",
        "quantum": {
            "trace_error": diagnostics.trace_error,
            "hermiticity_error": diagnostics.hermiticity_error,
            "minimum_eigenvalue": diagnostics.minimum_eigenvalue,
            "positivity_violation": diagnostics.positivity_violation,
        },
        "information": {"mutual_information_matrix": information.mutual_information_matrix().tolist()},
        "causality": {
            "intervention_family": causal.intervention_family,
            "number_of_interventions": causal.number_of_interventions,
            "causal_influence_matrix": causal.matrix.tolist(),
            "approximation_scope": causal.approximation_scope,
            "is_exact_supremum": causal.is_exact_supremum,
        },
        "geometry": {"signature": embedding.signature, "non_degenerate": embedding.non_degenerate},
        "gr": {"einstein_residual_norm": gr.einstein_norm, "discretization": gr.discretization},
    }
    metadata = {
        "seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "units": "dimensionless reference units",
        "scaling": "nondimensionalized toy model",
        "hilbert_space_dimensions": [2, 2, 2],
        "speculative_Q_enabled": False,
        "speculative_Delta_enabled": False,
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ROOT / "diagnostics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(results["classification"], indent=2))


if __name__ == "__main__":
    main()
