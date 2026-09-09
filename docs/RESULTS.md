# Results

## Computed reference experiment

The current reference run uses seed `1234`, three qubits for the GHZ information example, and dimensionless units.

- Quantum state diagnostics: trace, Hermiticity, and positivity residuals are at floating-point scale.
- Hamiltonian constraint: exact finite-dimensional kernel solving is validated, including degenerate and no-kernel cases.
- Relational clock: the controlled Page-Wootters state has near-zero constraint residual and distinct clock-conditioned matter states. The conditional fidelity is near zero for the two selected clock readings.
- Static Pauli intervention: the GHZ example has mutual information but zero sampled local influence. This is a finite intervention result, not an exact causal theorem.
- Dynamical causality: identity gives zero influence; CNOT gives one-way influence from control to target; the GHZ common-cause control has correlation without direct process influence.
- Geometry: Minkowski light-cone classification recovers the independent timelike/null ground truth with precision and recall one.
- GR tensor baseline: Minkowski curvature and Einstein tensor vanish. The conformal metric has nonzero Christoffel, Riemann, and Einstein tensors; the numerical Bianchi residual is reported separately.
- State reconstruction: unitary evolution is reversible at numerical precision; depolarization is not perfectly reversible.
- Fixed points: a contracting depolarizing process converges to a fixed point; a bit flip is classified as periodic, not as a causal loop.

## Scientific classification

`results.json` is the machine-readable source of the current classification. Information-to-geometry reconstruction and a full state-reconstruction target experiment are not claimed as validated. Speculative Q and Delta sectors remain disabled.

The strongest defensible result is that the finite reference framework can distinguish correlation from direct dynamical influence in controlled toy channels while preserving explicit quantum and geometric diagnostics.
