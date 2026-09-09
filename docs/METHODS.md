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
