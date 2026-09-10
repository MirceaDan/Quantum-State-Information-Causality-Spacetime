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
- Hidden-geometry baseline reconstruction: the optimizer saw only observables from an independent quantum Markov process and a separate causal chain. It did not receive ground-truth coordinates. For seed `1234`, train loss was `5.39e-6`, validation loss `2.818`, test loss `3.926`, causal precision `0.167`, causal recall `0.4`, and relative coordinate error `0.368`. The experiment is therefore classified `BASELINE_GEOMETRY_RECONSTRUCTION_FAILED`.
- Ground-truth dataset: `WORLD_1` is Minkowski, `WORLD_2` is the temporal conformal metric, and `WORLD_3` is an independent spatial conformal metric. Their quantum/process observables are generated independently and reused across worlds only as a controlled independence test. Randomized controls preserve pairwise information values while destroying their event pairing.
- Mapping comparison: null, fixed analytical, and five-parameter regularized `F_THETA` models report separate train/validation/test losses. None is promoted to a physical law.
- Stability: ten reconstruction seeds, a proper Lorentz boost, and Bianchi step refinement are included in machine-readable outputs. These controls do not convert the failed reconstruction into a positive result.

## Scientific classification

`results.json` is the machine-readable source of the current classification. `BASELINE_VALID` now comes from the computed Minkowski Einstein tensor. `BIANCHI_VALID`, `ENERGY_MOMENTUM_CONSERVATION_VALID`, and `DELTA_CONSERVATION_VALID` are separate. The generated scientific conclusion is `RECONSTRUCTION_UNSTABLE`: the baseline geometry reconstruction failed its held-out and causal criteria and varies across seeds. This is a failure of the current observable-to-geometry pipeline, not evidence for new physics. The speculative sectors are frozen exactly at `Q = 0` and `Delta_munu = 0` and were not available to absorb reconstruction error.

The strongest defensible result is that the finite reference framework can distinguish correlation from direct dynamical influence in controlled toy channels, while the present independent-observable baseline cannot reconstruct the chosen geometry under the stated split and loss. No conclusion about emergent spacetime, time travel, or new physics follows.
