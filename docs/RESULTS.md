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

## Reverse-Theseus historical geometry reconstruction (new phase)

Machine-readable output: `results/historical/historical_reconstruction_results.json` (from `run_historical_reconstruction.py`), covering `EXP-HIST-NULL-001`, `EXP-HIST-COUPLED-001`, `EXP-HIST-RETENTION-001`, `EXP-HIST-IDENT-001`, and `EXP-HIST-RANDOM-001` for all three ground-truth worlds (`WORLD_1` Minkowski, `WORLD_2` temporal-conformal, `WORLD_3` spatial-conformal), each over 10 seeds.

- **EXP-HIST-NULL-001** (preserved, unchanged): independent observables do not reconstruct geometry; this negative result still holds and is not reinterpreted.
- **EXP-HIST-COUPLED-001 / EXP-HIST-RANDOM-001:** the geometry-coupled FLEXIBLE reconstruction clearly outperforms its own randomized-control counterpart (mean test loss roughly 1-2 orders of magnitude lower for the coupled case across all three worlds), which supports that the chosen coupling mechanism does carry reconstructible relational information. However, the FLEXIBLE model's mean test loss (seeds 0-9) remains well above the RESTRICTED model's on all three worlds, and both remain far above a benchmark success threshold, with high seed-to-seed variance (std comparable to or larger than the mean). With only four relational events per synthetic world (six pairwise constraints), the nonlinear joint fit of coordinates and a global conformal scale is data-starved and prone to poor local minima.
- **EXP-HIST-IDENT-001:** all three pairwise world comparisons are `IDENTIFIABLE` (observable-vector distances 0.17-0.29, well above the identity tolerance); a world is correctly reported `INFORMATIONALLY_NON_IDENTIFIABLE` from itself.
- **EXP-HIST-RETENTION-001:** `F_geom(eta)` is computed at six retention levels per world; it is a benchmark quantity, not a physical law.

**Overall classification: `OPTIMIZATION_INSUFFICIENT`** for all three worlds under `classify_historical_reconstruction`. The coupling mechanism demonstrably carries geometry-dependent signal (beats its own randomized control, and worlds are mutually identifiable), but the current small-N nonlinear least-squares fit does not reliably converge to a low-loss reconstruction. This is reported as a genuine negative/inconclusive result rather than adjusted to force a positive one.
