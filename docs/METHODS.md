# Methods

## Scope

This is a finite-dimensional reference experiment, not a quantum-gravity simulation. The attached ansatz is preserved as the mathematical specification; implementation labels distinguish established definitions, numerical approximations, modeling assumptions, and speculative sectors.

## Quantum state

The reference Hilbert space is a tensor product of qubits. Density matrices are checked for unit trace, Hermiticity, and positive semidefiniteness. Partial traces use an explicit tensor reshape. Entropy and mutual information use the von Neumann definitions in the specification. No optimizer modifies a physical state in the current phase.

## Constraint and relational clock

`HamiltonianConstraintSolver` diagonalizes a finite Hermitian constraint, identifies the zero-eigenvalue subspace within a stated tolerance, projects a preferred vector into that subspace, and reports `NO_PHYSICAL_STATE` when the kernel is empty. The Page-Wootters experiment uses `H_M=diag(0,1)` and `H_C=diag(0,-1)` with the entangled zero-energy state. Clock projectors are Fourier-basis projectors; no external simulation-time variable enters the state or conditioning.

## Causality

The static causal estimate samples a finite Pauli intervention family and is explicitly not the exact supremum. The dynamical experiment uses a finite I/X intervention family and compares target marginals with trace distance, a bounded operational measure. Identity, CNOT, and common-cause GHZ cases are independent ground-truth controls.

## Geometry and GR

Minkowski and conformal metrics are represented by metric functions with analytic first and second derivatives. Christoffel symbols, Riemann, Ricci, scalar curvature, and Einstein tensors are calculated by tensor contractions. Covariant divergence uses the mixed tensor, finite central differences for its derivative, and explicit connection terms. The Bianchi residual is independently reported. The conformal example is a controlled curved metric, not a claim about a physical source solution.

## Reversibility and fixed points

Kraus operators define fixed-order quantum channels. Reversible unitary and depolarizing channels are compared by forward/reverse fidelity and round-trip relative entropy. Fixed-point experiments distinguish convergence from periodic or unresolved processes. No fixed point is classified as a causal loop.

## Reproducibility

`run_reference.py` records the seed, software metadata, dimensions, tolerances, intervention families, and disabled speculative flags. It writes JSON/CSV results and labeled plots. The global optimizer, Q sector, Delta sector, model selection, and large ablation matrix are intentionally not enabled in this reference phase.

## Reverse-Theseus historical geometry reconstruction phase

This phase changes the scientific question. The earlier `geometry_reconstruction.py` experiment fits coordinates from observables generated **independently** of geometry (`ground_truth.py`'s `information_matrix` is identical across all three worlds by construction) and is retained unchanged as the null baseline, `EXP-HIST-NULL-001`.

The new construction in `historical_process.py` is a **geometry-coupled** relational process: `O_t1 -> [G_t0]`.

- **Coupling mechanism (declared MODEL ASSUMPTION, not established physics):** adjacent relational events (qubits) are entangled with a partial-swap unitary `U(theta) = cos(theta) I + i sin(theta) SWAP`, where `theta = (pi/4) exp(-|s_ij| / length_scale)` and `s_ij` is the world's Lorentzian invariant interval between the two events. Smaller |interval| gives stronger coupling. This is the *only* place geometric information enters; no coordinate or metric component is ever written into a quantum state or an observable vector.
- **Information retention (`eta`):** after every coupling step, a standard local depolarizing channel is applied to both coupled qubits with probability `(1-eta) * 0.5`. `eta=1` applies no extra loss; `eta=0` applies the maximal loss.
- **Observables (`historical_reconstruction.HistoricalObservables`):** pairwise mutual information, multipartite (triple) co-information, and an *operational* causal-response tensor. The causal probe (`historical_process.historical_intervention_response`) replays the full process with an X-flip intervention applied to each event **at the moment it enters the causal chain** (not on the frozen final state, which cannot signal across disjoint subsystems by the no-communication theorem), and measures the trace-distance response of every later event's marginal. This correctly recovers a forward-only causal structure.
- **Reconstruction baselines (`compare_historical_baselines`):** NULL (zero parameters), RESTRICTED (coordinates only, flat Minkowski assumed), and FLEXIBLE (coordinates plus one global conformal-scale parameter, letting the fit represent `g_hat_munu(x) = exp(2 * scale_hat * x^0) * eta_munu`, evaluated against ground-truth scalar curvature at a reference point). All three share the same train/validation/test pair split for a given seed.
- **Identifiability (`run_identifiability_test`):** compares the flattened observable vectors of two worlds under the same construction; below a tolerance the pair is reported `INFORMATIONALLY_NON_IDENTIFIABLE` rather than forcing a unique reconstruction.
- **Randomized control (`run_randomized_control_experiment`):** pairwise-permutes the coupled observables (preserving marginals, destroying pairing) and runs the identical pipeline.
- **Classification (`classify_historical_reconstruction`):** returns the weakest defensible label from `HISTORICAL_RECONSTRUCTION_SUPPORTED`, `HISTORICAL_RECONSTRUCTION_UNSTABLE`, `INFORMATION_INSUFFICIENT`, `REPRESENTATION_INSUFFICIENT`, `OPTIMIZATION_INSUFFICIENT`, or `OBSERVABLES_NON_IDENTIFYING`. `Q = 0` and `Delta_munu = 0` are enforced by `BaselineGate().validate()` in every entry point.

## Experimental-integrity / identifiability audit (follow-up phase)

This phase (`run_integrity_audit.py`) does not change the coupling law, the synthetic worlds, or model capacity. It audits the previous phase's positive-looking signal for leakage and confounds.

- **Leakage-free split (`ObservableSplit`, `build_observable_split`):** pairwise, causal, and multipartite observables share one deterministic seed-derived train/validation/test split. A causal entry is masked into the training set ONLY if its underlying event pair is a training pair (`_mask_causal_to_pairs`); a multipartite triple is training-only if all three of its sub-pairs are training pairs. `assert_causal_subset` is a structural runtime guard invoked inside the `least_squares` residual closure itself (not just at call sites) that raises `ValueError` if any non-training causal entry would reach the optimizer. Held-out (validation/test) loss is always evaluated with a fixed mutual-information-based interval metric, identical across every observable-attribution mode, so ablations remain comparable.
- **Process-architecture controls (`historical_process.build_geometry_coupled_history`):** `FIXED_ORDER_CONTROL` (the original default), `ZERO_COUPLING` (`theta=0` for every step), `SHUFFLED_GEOMETRY` (the true invariant intervals are reassigned to coupling steps by a fixed permutation, geometry itself untouched), and `RANDOMIZED_ORDER` (the same fixed set of adjacent-pair gates executed in a permuted sequence). All four run through the identical `compare_historical_baselines` pipeline via `run_process_variant_experiment`.
- **Observable attribution (`run_observable_ablation`):** the same FLEXIBLE model capacity and the same train/validation/test split are fit five times, varying only which observable types feed the training residual: `MI_ONLY`, `CAUSAL_ONLY`, `MI_PLUS_CAUSAL`, `MULTIPARTITE_ONLY`, `ALL_OBSERVABLES`.
- **Identifiability terminology (section 4):** the observable-vector-distance test now reports `OBSERVABLY_DISTINCT` / `OBSERVABLY_INDISTINGUISHABLE` rather than `IDENTIFIABLE`/`NON_IDENTIFIABLE`. A genuine inverse-model identifiability search (does some reconstruction-model parameterization make two geometrically inequivalent worlds observably equivalent?) is explicitly NOT implemented; `IdentifiabilityReport.stronger_test_available` is `False` and `.note` states this.
- **Label/order confound (`build_relabeled_world`, `run_label_permutation_control`):** every event-indexed field of a world (coordinates, invariant intervals, causal relation, information matrix, process-causal relation, pair classifications) is consistently relabeled by a fixed permutation, and the identical `COUPLED` pipeline is re-run. A large relative gap between the original and relabeled mean test loss is reported as `LABEL_OR_ORDER_CONFOUND_DETECTED` rather than silently accepted.
- **N-scaling study (`historical_scaling.py`, `run_scaling_study`):** the same three geometry families and the same coupling law are re-instantiated at `N in {4, 6, 8, 12, 20}` relational events, reusing `ground_truth._world`/`_process_information`/`_process_causal_chain` unchanged. Because the dense qubit simulation scales as `4**N` in memory, only `N in {4, 6, 8}` are actually executed; `N in {12, 20}` are structurally supported but not run (documented in LIMITATIONS.md, not silently skipped).
- **Intervention-family robustness (`run_intervention_family_robustness`):** the causal probe is rebuilt with a Pauli-X and a Pauli-Z single-qubit intervention; both remain a finite operational approximation, not the exact operational supremum.
- **Retention curve as a control only:** `run_information_retention_curve` now also runs the identical eta sweep on a `ZERO_COUPLING` process, so "does signal vanish as retention drops" can be compared against "does a process with no geometric coupling at all behave differently" (section 8).
- **Scientific gate (`ScientificGate`, `compute_scientific_gate`) and predefined classification (`classify_historical_reconstruction`):** the classification vocabulary is fixed in code before any experiment matrix is run: `NO_GEOMETRY_SIGNAL_DETECTED`, `OBSERVABLY_DISTINCT_BUT_NOT_IDENTIFIABLE`, `GEOMETRY_DEPENDENT_SIGNAL_DETECTED`, `OPTIMIZATION_INSUFFICIENT`, `REPRESENTATION_INSUFFICIENT`, `INFORMATION_INSUFFICIENT`, `EXPERIMENTAL_CONFOUND_PRESENT`. If `train_test_leakage`, `causal_leakage`, `order_confound`, or `label_confound` is not `PASS`, the function returns `EXPERIMENTAL_CONFOUND_PRESENT` unconditionally -- no positive conclusion is reachable when the gate fails.
