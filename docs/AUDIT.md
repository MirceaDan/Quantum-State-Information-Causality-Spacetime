# Repository Audit

Date: 2026-09-09

This audit is based on the current source and tests. A passing test is not treated as physical validation by itself.

## Classification scheme

- `IMPLEMENTED_AND_VALIDATED`: implementation is present and tests compare against an independent analytic or structural result.
- `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED`: implementation exists, but tests cover only execution, shape, self-consistency, or a trivial identity.
- `PLACEHOLDER`: interface exists but does not implement the stated mathematical object.
- `MATHEMATICAL_APPROXIMATION`: finite-dimensional, sampled, discretized, or otherwise explicitly limited implementation.
- `SPECULATIVE`: exploratory ansatz or extension, not established physics.
- `INCORRECT_OR_MISLEADING`: implementation or output claims more than the code establishes.

## Source components

| Component | Classification | Evidence and limitation |
|---|---|---|
| `quantum.py` | `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED` plus `MATHEMATICAL_APPROXIMATION` | Implements finite-dimensional density matrices, partial trace, entropy, mutual information, fidelity, relative entropy, and a spectral Hamiltonian-constraint solver. Analytic basis/product/GHZ/mixed values are now tested. It remains a finite-dimensional approximation and assumes a Hermitian constraint matrix. |
| `information.py` | `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED` plus `SPECULATIVE` mapping | Entropy and mutual information delegate to the quantum implementation. The information-distance formula now reports metric axioms but remains an exploratory dissimilarity, not a physical law. `F_theta` has no train/validation/test experiment. |
| `causality.py` | `MATHEMATICAL_APPROXIMATION` | Retains the finite Pauli static estimate and adds identity/CNOT dynamical channels measured by bounded trace distance. Controlled tests distinguish no influence, direct one-way influence, and common-cause correlation. It remains finite sampling, not the exact operational supremum. |
| `geometry.py` | `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED` plus `MATHEMATICAL_APPROXIMATION` | Computes separations, light-cone relations, and precision/recall against an independent causal matrix. Information-to-geometry reconstruction is not implemented. |
| `gr.py` | `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED` plus `MATHEMATICAL_APPROXIMATION` | The legacy residual API remains limited, but the module now calculates Christoffel, Riemann, Ricci, scalar curvature, and Einstein tensors for analytic metric fields, plus connection-aware covariant divergence and a numerical Bianchi check. Central differences are used for tensor-field derivatives. |
| `history.py` | `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED` | Stores computational index, clock reading, and optional external time and rejects a populated external-time field. It does not infer ordering from clock/causal correlations and has no hidden-time source audit. |
| `process.py` | `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED` | Adds explicit fixed-order Kraus channels with CP-by-construction and TP completeness diagnostics. Process matrices/indefinite causal order are explicitly unimplemented. |
| `relational_clock.py` | `IMPLEMENTED_AND_VALIDATED` plus `MATHEMATICAL_APPROXIMATION` | Controlled two-level Page-Wootters experiment with exact finite-dimensional zero-energy constraint and distinct Fourier-clock-conditioned states. This is not evidence for physical emergent time. |
| `reconstruction.py` | `IMPLEMENTED_AND_VALIDATED` plus `MATHEMATICAL_APPROXIMATION` | Unitary and depolarizing channel baselines report reverse fidelity and round-trip loss, explicitly labeled state reconstruction. |
| `fixed_point.py` | `IMPLEMENTED_AND_VALIDATED` plus `MATHEMATICAL_APPROXIMATION` | Detects convergence and short periods for finite linear channels; does not infer causal loops. |
| `hidden_time.py` | `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED` | AST audit classifies metadata/numerical controls and forbidden physical-time names. It is a static audit, not semantic proof. |
| `run_reference.py` | `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED` | Classifications now derive from clock, dynamical causality, tensor/Bianchi, reconstruction, fixed-point, and hidden-time diagnostics. Global optimization and full ablations remain absent. |

## Existing tests

| Test file | Classification | What it actually establishes |
|---|---|---|
| `test_quantum_constraints.py` | `analytic_validation` for Bell reductions; `identity_test` for fidelity/relative entropy | Checks trace/Hermiticity/positivity diagnostics, Bell reduced state, `S=log(2)`, and `I=2log(2)`. Fidelity and relative entropy are self-identities only. |
| `test_relational_clock.py` | `primitive_test` and `identity_test` | Covers the original conditioning and manually selected kernel state. `test_relational_clock_experiment.py` adds a controlled Page-Wootters experiment. |
| `test_information.py` | `primitive_test` and `MATHEMATICAL_APPROXIMATION` smoke test | Checks matrix symmetry/diagonal and positivity of one exploratory dissimilarity. It does not test metric axioms or held-out reconstruction. |
| `test_causality.py` | `primitive_test` / metadata smoke test | Checks finite-family metadata; `test_dynamical_causality.py` adds independent identity/CNOT/common-cause controls. |
| `test_geometry.py` | `primitive_test` | Checks Minkowski separation and a diagonal coordinate transformation. It does not validate causal classification or recovered geometry. |
| `test_gr_baseline.py` | `identity_test` | Covers the legacy flat residual API; tensor calculus and Bianchi tests cover actual curvature calculations. |
| `test_conservation.py` | `identity_test` | Covers the legacy zero-array monitor; covariant divergence tests now include conserved and deliberately non-conserved tensors. |
| `test_no_hidden_time.py` | `primitive_test` | Rejects an explicitly populated `external_time` field in one object. It does not audit source code or distinguish optimizer iteration from physical time globally. |

## Confirmed preliminary conclusions

1. The repository contains finite-dimensional quantum primitives, an explicit constraint solver, a controlled relational-clock experiment, and finite causal intervention estimates.
2. Identity/CNOT/common-cause controls now demonstrate correlation versus direct dynamical influence in toy channels.
3. Tensor curvature, covariant divergence, and a numerical Bianchi check are implemented for controlled analytic metric fields, not general numerical relativity.
4. State reconstruction and fixed-point behavior are measured without being labeled temporal reversal or causal loops.
5. Information-to-geometry inference, held-out reconstruction, global optimization, ablations, and speculative sectors remain unvalidated.

## Required repair order

1. Add held-out synthetic geometry reconstruction with fixed, low-dimensional, and null mappings.
2. Add refinement, coordinate, noise, precision, seed, and intervention-family stress tests.
3. Add complete model ablations and train/validation/test reporting before any optimizer.
4. Implement speculative Q and Delta sectors one at a time only after baseline failures are demonstrated.
5. Keep process-matrix/indefinite-order work separate from fixed-order channel validation.

## Reverse-Theseus historical reconstruction phase (added 2026-09-10)

Conceptual change: OLD = independent quantum/process observables -> geometry (retained as `EXP-HIST-NULL-001`, still a valid negative null result). NEW = historical geometry -> geometry-coupled relational quantum/process evolution (`historical_process.py`) -> later surviving observables (`historical_reconstruction.HistoricalObservables`) -> hidden historical geometry reconstruction attempt (`historical_reconstruction.compare_historical_baselines`).

| Component | Classification | Evidence and limitation |
|---|---|---|
| `historical_process.py` | `MODELING_ASSUMPTION` | Declared, non-established coupling: partial-swap angle from invariant-interval magnitude; local depolarizing loss from `eta`. Tested for valid density matrices, angle monotonicity, translation invariance of the coupling, and retention-ordered information loss. |
| `historical_reconstruction.py` | `MATHEMATICAL_APPROXIMATION` plus `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED` for the FLEXIBLE metric family | NULL/RESTRICTED/FLEXIBLE baselines share one train/validation/test split; FLEXIBLE adds a single global conformal-scale parameter, evaluated against ground-truth scalar curvature. Current numeric result across 10 seeds and 3 worlds is `OPTIMIZATION_INSUFFICIENT` (see RESULTS.md); this is reported as-is, not adjusted to force a positive result. |
| `run_historical_reconstruction.py` | `IMPLEMENTED_BUT_ONLY_SMOKE_TESTED` | Orchestrates all five experiment IDs and writes `results/historical/historical_reconstruction_results.json`; does not itself validate physics. |

Pre-existing breakage found and fixed before this phase began (unrelated to the new experiment but blocking `ground_truth.py` imports and several tests): missing `lorentz_boost` in `geometry.py`, missing `spatial_conformal_geometry`/`bianchi_convergence` in `gr.py`, missing `f1`/`false_positive_rate`/`false_negative_rate` keys in `compare_causal_relations`, and missing `BIANCHI_VALID`/`ENERGY_MOMENTUM_CONSERVATION_VALID`/`DELTA_CONSERVATION_VALID`/`CAUSAL_LOOP_VALID`/`gr.baseline_source` in `run_reference.py`'s classification output. The full suite (54 tests) was confirmed passing before any new-phase code was added.
