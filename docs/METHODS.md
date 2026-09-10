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
