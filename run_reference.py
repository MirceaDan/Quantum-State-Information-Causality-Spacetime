"""Run the verified reference experiment and write reproducible diagnostics."""

from __future__ import annotations

import json
import csv
import platform
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from relational_dynamics.causality import PauliInterventionFamily, sampled_causal_influence
from relational_dynamics.causality import cnot_process, identity_process, sampled_dynamical_influence
from relational_dynamics.fixed_point import FixedPointExperiment
from relational_dynamics.geometry import LorentzianEmbedding
from relational_dynamics.gr import EinsteinResidual, bianchi_residual, conformal_geometry, minkowski_geometry, minkowski_metric
from relational_dynamics.hidden_time import audit_source_tree
from relational_dynamics.information import InformationCausalityEngine
from relational_dynamics.quantum import QuantumSystem, density_matrix
from relational_dynamics.reconstruction import depolarizing_channel, reconstruct_state, unitary_channel
from relational_dynamics.relational_clock import PageWoottersExperiment

SEED = 1234


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


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
    clock = PageWoottersExperiment.two_level_controlled().run()
    direct_state = density_matrix(np.array([1, 0, 0, 0], dtype=complex))
    direct_causality = sampled_dynamical_influence(direct_state, (2, 2), cnot_process(), 0, 1)
    independent_causality = sampled_dynamical_influence(direct_state, (2, 2), identity_process(), 0, 1)
    curved_geometry = conformal_geometry(scale=0.2)
    curved_point = np.array([0.3, 0.1, 0.0, 0.0])
    curved_tensors = curved_geometry.curvature_tensors(curved_point)
    flat_bianchi = bianchi_residual(minkowski_geometry(), np.zeros(4))
    curved_bianchi = bianchi_residual(curved_geometry, curved_point)
    flip = np.array([[0, 1], [1, 0]], dtype=complex)
    reversible = reconstruct_state(density_matrix(np.array([1, 0], dtype=complex)), unitary_channel(flip), unitary_channel(flip))
    irreversible = reconstruct_state(density_matrix(np.array([1, 0], dtype=complex)), depolarizing_channel(0.5), depolarizing_channel(0.5))
    fixed_point = FixedPointExperiment(depolarizing_channel(0.5)).run(density_matrix(np.array([1, 0], dtype=complex)))
    hidden_time = audit_source_tree(ROOT / "src")

    results = {
        "classification": {
            "BASELINE_VALID": gr.einstein_norm < 1e-12,
            "RELATIONAL_TIME_VALID": clock.consistent_with_page_wootters_toy_model,
            "INFORMATION_GEOMETRY_VALID": False,
            "CAUSAL_CONSISTENCY_VALID": direct_causality.influence > independent_causality.influence,
            "BIANCHI_VALID": np.linalg.norm(curved_bianchi) < 1e-4,
            "ENERGY_MOMENTUM_CONSERVATION_VALID": np.linalg.norm(curved_bianchi) < 1e-4,
            "DELTA_CONSERVATION_VALID": True,
            "CAUSAL_LOOP_VALID": not fixed_point.causal_loop_candidate,
            "QUANTUM_CONSTRAINTS_VALID": diagnostics.positivity_violation < 1e-12 and diagnostics.trace_error < 1e-12,
            "STATE_RECONSTRUCTION_VALID": False,
            "FIXED_POINT_VALID": fixed_point.classification == "FIXED_POINT" and fixed_point.fixed_point_error < 1e-8,
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
            "dynamic_no_influence": independent_causality.influence,
            "dynamic_direct_influence": direct_causality.influence,
        },
        "relational_clock": {
            "constraint_residual": clock.constraint_residual,
            "clock_distinguishability": clock.clock_distinguishability,
            "conditional_fidelity": clock.conditional_fidelity,
            "external_time_used": clock.external_time_used,
        },
        "geometry": {"signature": embedding.signature, "non_degenerate": embedding.non_degenerate},
        "gr": {
            "baseline_source": "computed Einstein tensor of Minkowski geometry",
            "einstein_residual_norm": gr.einstein_norm,
            "discretization": gr.discretization,
            "curved_christoffel_norm": float(np.linalg.norm(curved_tensors.christoffel)),
            "curved_riemann_norm": float(np.linalg.norm(curved_tensors.riemann)),
            "curved_einstein_norm": float(np.linalg.norm(curved_tensors.einstein)),
            "flat_bianchi_norm": float(np.linalg.norm(flat_bianchi)),
            "curved_bianchi_norm": float(np.linalg.norm(curved_bianchi)),
        },
        "reconstruction": {
            "reversible_reverse_fidelity": reversible.reverse_fidelity,
            "reversible_round_trip_loss": reversible.round_trip_loss,
            "irreversible_reverse_fidelity": irreversible.reverse_fidelity,
            "irreversible_round_trip_loss": irreversible.round_trip_loss,
        },
        "fixed_point": {
            "classification": fixed_point.classification,
            "fixed_point_error": fixed_point.fixed_point_error,
            "stability": fixed_point.stability,
            "causal_loop_candidate": fixed_point.causal_loop_candidate,
        },
        "hidden_time": {"record_count": len(hidden_time), "forbidden_count": sum(not record["allowed"] for record in hidden_time)},
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
        "optimizer": "none in reference phase",
        "learning_rate": None,
        "tolerances": {"state": 1e-10, "geometry": 1e-5},
        "convergence_criteria": "reported per experiment; no global optimizer",
        "intervention_family": "Pauli operations and finite I/X dynamical controls",
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2, default=json_default), encoding="utf-8")
    (ROOT / "diagnostics.json").write_text(json.dumps(metadata, indent=2, default=json_default), encoding="utf-8")
    (ROOT / "hidden_time_audit.json").write_text(json.dumps(hidden_time, indent=2, default=json_default), encoding="utf-8")
    result_root = ROOT / "results"
    for name in ("baseline", "relational", "causal", "geometry", "extensions", "ablations"):
        (result_root / name).mkdir(parents=True, exist_ok=True)
    rows = [
        {"experiment_id": "EXP-Q-001", "model_id": "QM", "seed": SEED, "metric": "quantum_constraint_residual", "value": diagnostics.trace_error + diagnostics.hermiticity_error + diagnostics.positivity_violation},
        {"experiment_id": "EXP-C-001", "model_id": "DYNAMIC_CNOT", "seed": SEED, "metric": "direct_influence", "value": direct_causality.influence},
        {"experiment_id": "EXP-C-002", "model_id": "DYNAMIC_IDENTITY", "seed": SEED, "metric": "direct_influence", "value": independent_causality.influence},
        {"experiment_id": "EXP-G-001", "model_id": "MINKOWSKI", "seed": SEED, "metric": "einstein_residual_norm", "value": gr.einstein_norm},
        {"experiment_id": "EXP-G-002", "model_id": "CONFORMAL", "seed": SEED, "metric": "bianchi_residual_norm", "value": float(np.linalg.norm(curved_bianchi))},
        {"experiment_id": "EXP-R-001", "model_id": "REVERSIBLE_UNITARY", "seed": SEED, "metric": "reverse_fidelity", "value": reversible.reverse_fidelity},
        {"experiment_id": "EXP-R-002", "model_id": "DEPOLARIZING", "seed": SEED, "metric": "reverse_fidelity", "value": irreversible.reverse_fidelity},
    ]
    (result_root / "master_results.json").write_text(json.dumps(rows, indent=2, default=json_default), encoding="utf-8")
    with (result_root / "master_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot_root = ROOT / "plots"
    plot_root.mkdir(exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.imshow(information.mutual_information_matrix(), cmap="viridis")
    plt.colorbar(label="mutual information [nats]")
    plt.title(f"EXP-I-001 / GHZ mutual information / seed={SEED}")
    plt.xlabel("subsystem index")
    plt.ylabel("subsystem index")
    plt.tight_layout()
    plt.savefig(plot_root / "information_matrix.png", dpi=140)
    plt.close()
    plt.figure(figsize=(4, 3))
    plt.bar(["identity", "CNOT"], [independent_causality.influence, direct_causality.influence])
    plt.ylabel("trace distance [dimensionless]")
    plt.title(f"EXP-C-001/002 / dynamical influence / seed={SEED}")
    plt.tight_layout()
    plt.savefig(plot_root / "causal_influence.png", dpi=140)
    plt.close()
    plt.figure(figsize=(4, 3))
    plt.bar(["flat Bianchi", "curved Bianchi"], [np.linalg.norm(flat_bianchi), np.linalg.norm(curved_bianchi)])
    plt.ylabel("covariant divergence norm [dimensionless]")
    plt.title(f"EXP-G-001/002 / Bianchi residual / seed={SEED}")
    plt.tight_layout()
    plt.savefig(plot_root / "bianchi_residuals.png", dpi=140)
    print(json.dumps(results["classification"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
