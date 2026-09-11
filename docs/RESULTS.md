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

## Experimental-integrity / identifiability audit (follow-up phase)

Machine-readable output: `results/historical/integrity_audit_results.json` (from `run_integrity_audit.py`), run on `WORLD_2` (temporal-conformal) at 10 seeds.

**Control matrix (FLEXIBLE, mean test loss across 10 seeds):**

| Control | mean | std | median | min | max |
|---|---|---|---|---|---|
| COUPLED | 15.53 | 17.90 | 11.28 | 0.018 | 66.65 |
| ZERO_COUPLING | 0.65 | 0.98 | 0.12 | 0.024 | 3.00 |
| SHUFFLED_GEOMETRY | 5.92 | 4.56 | 5.78 | 0.076 | 12.93 |
| RANDOMIZED_ORDER | 195.77 | 191.79 | 210.77 | 4.29 | 626.98 |
| RANDOMIZED_OBSERVABLE (permutation) | 977.41 | 2630.35 | 16.78 | 2.23 | 8838.20 |

COUPLED is *worse* on average than `ZERO_COUPLING`: with no coupling at all, mutual information collapses to a near-uniform (degenerate) target that a nonlinear fit satisfies almost trivially, producing a spuriously low loss that is not evidence of better reconstruction. This is exactly the kind of artifact the integrity audit is designed to catch.

**Label/order confound (section 5):** relabeling `WORLD_2`'s four events with a fixed permutation and re-running the identical `COUPLED` pipeline changed the mean test loss from 15.53 to 5.41 (a factor of ~3, `relative_gap` far above the 0.5 tolerance) -> **`LABEL_OR_ORDER_CONFOUND_DETECTED`**. The current small-N pipeline is not invariant to a pure relabeling of the same physical world, which it should be if only geometry mattered.

**Scientific gate:** `train_test_leakage=PASS`, `causal_leakage=PASS`, `geometry_leakage=PASS`, `scaling_study_present=PASS`, `intervention_robustness_present=PASS`, but **`order_confound=FAIL`** and **`label_confound=FAIL`**.

**FINAL CLASSIFICATION: `EXPERIMENTAL_CONFOUND_PRESENT`.** Per the predefined classification logic, a failing order/label-confound gate forces this outcome unconditionally; no positive conclusion is reachable. This supersedes the earlier phase's `OPTIMIZATION_INSUFFICIENT` reading -- the earlier positive-looking signal (COUPLED beating the pairwise-permutation randomized control) does not survive a leakage-free, confound-checked protocol as currently implemented.

**Observable attribution (section 3):** with matched splits, `MI_ONLY` (test_loss 7.06), `CAUSAL_ONLY` (8.33), `MI_PLUS_CAUSAL` (7.89), `MULTIPARTITE_ONLY` (8.33), and `ALL_OBSERVABLES` (7.89) are all similar in magnitude; causal_f1 is identical (0.667) across every mode. This does not show a clear attribution advantage for any single observable type in the current setup.

**Identifiability (weak sense only):** all three world pairs remain `OBSERVABLY_DISTINCT` (distances 0.17-0.28). The stronger inverse-model identifiability test described in section 4 is not implemented (`stronger_test_available=False`).

**N-scaling study (feasible sizes N=4,6,8 only; see LIMITATIONS.md for N=12,20):** test-loss mean does not improve monotonically with N (e.g. WORLD_2: 15.53 at N=4, 15.08 at N=6, 23.82 at N=8); causal_f1 mean decreases with N (0.55 -> 0.31 -> 0.24). There is no evidence in this range that reconstruction improves systematically with relational sample size.

**Intervention-family robustness:** X and Z single-qubit interventions give different FLEXIBLE mean test losses (15.53 vs 6.27), indicating the causal-observable signature is not stable across intervention families in the current construction.

**Retention curve as control only (section 8):** `F_geom(eta)` for COUPLED is non-monotonic (0, 0, 0.167, 0.167, 0, 0.333) and the ZERO_COUPLING control is flat at 0 for every eta -- COUPLED does not cleanly dominate its own zero-coupling control across the sweep, consistent with the confound finding above.

## Milestone 4 -- Permutation-Equivariant Quantum Process Memory

Machine-readable output: `results/historical/m4_permutation_equivariant_process_memory_results.json` (from `run_permutation_equivariant_process_memory.py --existing-suite-passed`).

**Scientific gate (section 19 of the M4 spec):**

| Gate condition | Result |
|---|---|
| `M4_EQUIVARIANCE_PASS` | **PASS** (overall max error 4.6e-15 across all N=4 permutations, state/pairwise/multipartite/intervention/memory/global-scalar checks) |
| `M4_NEGATIVE_CONTROL_PASS` | **PASS** (the deliberately broken index-parity model fails equivariance for every non-identity permutation tested, max error up to 0.88) |
| `M4_REPRODUCIBILITY_PASS` | **PASS** (no RNG anywhere in the simulator; identical inputs give bit-identical output) |
| `EXISTING_TEST_SUITE_PASS` | **PASS** (125 tests: 124 passed, 1 expected `xfail` for the still-open Reverse-Theseus confound, 0 unexpected failures) |

All four gate conditions pass -> **`IMPLEMENTED_AND_VALIDATED`** for the M4 permutation-equivariant process-memory substrate specifically (state, pairwise-information, multipartite-information, intervention-response, and process-memory-score equivariance; NOT a claim about geometry reconstruction, which this milestone does not attempt).

**Scaling (N=3,4,5; exhaustive permutation enumeration):**

| N | Hilbert dim | permutations tested | max equivariance error | runtime |
|---|---|---|---|---|
| 3 | 8 | 6 | 6.7e-16 | 0.06s |
| 4 | 16 | 24 | 4.6e-15 | 0.48s |
| 5 | 32 | 120 | 5.1e-15 | 6.4s |

Error does not grow with N in this range; runtime grows roughly with `N! x 4^N` from the exhaustive-permutation x dense-simulation combination, consistent with the known `4^N` scaling limitation already documented for the Reverse-Theseus phase.

**Process-memory score** (mean multi-step intervention response) for the N=4 sample topology: a nonzero value, confirming information from an earlier declared relation step is still detectable in a later marginal after passing through the whole schedule -- a literal, narrowly-defined notion of finite memory, not a general non-Markovianity claim.

No geometry, Lorentzian interval, conformal factor, GR, or CTC content is present anywhere in M4. `Q = 0` and `Delta_munu = 0` throughout.

## Milestone 5 -- Hidden Causal Topology Recovery

> **Superseded protocol notice (2026-09-11):** the original label-invariance result below compared a relabeled graph after nuisance parameters and schedule had been regenerated or left attached to numeric labels. It was not a same-physical-process relabeling test and its `M5_LABEL_INVARIANCE_PASS=FAIL` conclusion is invalid. The corrected clean-validation results are reported in the next section.

### M5.1/M5.2 clean validation (final hardening)

Machine-readable results: `results/historical/m5_clean_validation_results.json`. Scope: all 34 non-isomorphic N=4 directed topologies with 2--4 edges and all nodes participating; reference seeds 100--111; held-out seeds 1000--1009; strengths in `[0.35,0.75]`, local state angles in `[0,2*pi]`, local noise in `[0,0.1]`, and retention in `{1.0,0.9,0.75,0.5}`. Runtime: 4033.8 seconds.

The corrected same-physical-process relabeling protocol **passes** for all tested topologies and exhaustive N=4 permutations. Maximum observable discrepancy is `4.88e-15`; reconstruction predictions are invariant up to graph isomorphism. The old label-invariance failure was therefore a protocol error (resampling/misattachment), not a property of M5.

Final gates: process relabeling PASS; label invariance PASS; train/test leakage PASS; causal leakage PASS; identifiability audit execution PASS; existing suite PASS; **schedule robustness FAIL; nuisance robustness FAIL; topology recovery FAIL**. Classification: **`IMPLEMENTED_BUT_NOT_FULLY_VALIDATED`**.

M5.1 operational cyclic-vs-DAG classification (`CAUSAL_ONLY`): accuracy `0.803`, FPR `0.150`, FNR `0.250`, confusion matrix TP=120, TN=153, FP=27, FN=40. Random-label control is identical (`0.803`), as required. Random schedule gives `0.765`; randomized strengths `0.785`; zero coupling gives `0.529` with FNR `1.0`, showing cyclic separation requires nontrivial dynamics.

M5.2 exact recovery over the 34-class family: `MI_ONLY=0.053`, `MULTIPARTITE_ONLY=0.035`, `CAUSAL_ONLY=0.324`, `MEMORY_ONLY=0.056`, `CAUSAL_PLUS_MEMORY=0.191`, `ALL=0.156`. Causal response is the strongest component; the scalar memory diagnostic does not add useful recovery information and reduces accuracy when combined with causal response. Clean/control accuracies under the primary mode are `0.265` (clean), `0.118` (shuffled observables), and `0.029` (zero coupling).

Schedule robustness fails: prediction-invariance rates are `0.357` for `CAUSAL_ONLY` and `0.286` for `CAUSAL_PLUS_MEMORY`. Independently resampled nuisance-parameter recovery is `0.179` overall; by retention it is `0.188` (1.0), `0.182` (0.9), `0.182` (0.75), and `0.165` (0.5). The primary limitation is not a sharp noise threshold but poor parameter/schedule robustness already in the ideal case.

Finite-domain ambiguity audit: 529 of 561 non-isomorphic topology pairs are `OBSERVABLY_INDISTINGUISHABLE` within tolerance `0.05`; 32 are separated. This is not a theorem of global non-identifiability, but it shows the present finite observable family does not uniquely encode most tested topologies.

Machine-readable output: `results/historical/m5_topology_reconstruction_results.json` (from `run_topology_reconstruction.py --existing-suite-passed`). Mandatory topologies: CHAIN, BRANCH, MERGE, CYCLIC_PROCESS (N=4); ensemble: 10 non-isomorphic topologies (deduplicated via `canonical_topology_key`).

**Scientific gate:**

| Gate condition | Result |
|---|---|
| `M4_EQUIVARIANCE_PASS` | PASS (established separately in Milestone 4) |
| `M5_LABEL_INVARIANCE_PASS` | **FAIL** (see below) |
| `M5_NEGATIVE_CONTROL_PASS` | PASS |
| `M5_REPRODUCIBILITY_PASS` | PASS (no RNG in the simulator) |
| `M5_NO_LEAKAGE_PASS` | PASS (`assert_no_seed_leakage`; TRAIN_SEEDS and TEST_SEEDS are disjoint by construction) |
| `EXISTING_TEST_SUITE_PASS` | PASS (151 tests: 149 passed, 2 expected `xfail`, 0 unexpected failures) |

**The overall M5 gate FAILS because `M5_LABEL_INVARIANCE_PASS` fails.** Per the milestone's own rule (section 27), this means: **stop interpreting topology-recovery accuracy as a general result; the finding below is reported honestly rather than acted on as if it were validated.** Classification: **`IMPLEMENTED_BUT_NOT_FULLY_VALIDATED`** -- explicitly NOT `EXPERIMENTAL_CONFOUND_PRESENT`, because M4 already established the underlying process/observables are structurally permutation-equivariant to floating precision; the failure here is statistical instability of finite-sample nearest-candidate distance matching at this small N, not a hidden label/index confound.

**M5.1 -- cycle vs. DAG classification (`RECOMMENDED_MODE=CAUSAL_ONLY`):** `cycle_detection_accuracy=1.000`, `cycle_false_positive_rate=0.000`, `cycle_false_negative_rate=0.000` across the 4 mandatory topologies x 3 test seeds. **The causal intervention-response observable cleanly separates cyclic from acyclic process topologies in this small-N setting.**

**M5.2 -- observable ablation (exact recovery accuracy on the 4 mandatory topologies, isomorphism-aware distance, 12 reference seeds):**

| Mode | Accuracy |
|---|---|
| `MI_ONLY` | 0.25 |
| `MULTIPARTITE_ONLY` | 0.50 |
| `CAUSAL_ONLY` | 0.75 |
| `MEMORY_ONLY` | 0.25 |
| `CAUSAL_PLUS_MEMORY` | **1.00** |
| `ALL` (naive equal weight) | 0.50 |

**This directly answers section 13's question: yes, temporal/causal process information (`CAUSAL_PLUS_MEMORY`) substantially outperforms static pairwise correlations (`MI_ONLY`) for topology recovery in this regime.** Naively combining all four observable types with equal weight (`ALL`) performs WORSE than causal information alone -- the static components (MI, multipartite, memory) can dilute or even mislead the naive combined distance for this small ensemble.

**Label-permutation invariance (section 14, the central M5 integrity test):** FAILS for both `ALL_OBSERVABLES` and `RECOMMENDED_MODE=CAUSAL_ONLY` on the tested relabeling of each mandatory topology, even after using the (structurally necessary) isomorphism-aware distance and 12-20 Monte-Carlo reference samples per candidate. Exploration during development showed this is a genuine statistical-sample-size effect, not a residual structural label-dependency bug: increasing reference samples changed which specific queries succeeded or failed, but did not eliminate the instability at N=4 with a 4-10-topology candidate ensemble.

**Identifiability (CHAIN vs. CYCLIC_PROCESS, 10 trials, bounded parameter search):** `OBSERVABLY_DISTINCT` at best_distance=0.083 -- across the tested parameter family, no sampled parameter pair made the two topologies' observable bundles coincide.

**Negative controls:** shuffled-observable and topology-independent-process controls both failed to recover the true topology, as expected.

## Root-cause analysis of the order/label confound (diagnosis only, no repair attempted)

Machine-readable output: `results/historical/root_cause_analysis_results.json` (from `run_root_cause_analysis.py`). **Root-cause classification: `MULTIPLE_CONFUNDS`.** Full mechanism and evidence are in docs/AUDIT.md; summary:

- The fixed-order construction couples events by raw index adjacency (`(0,1),(1,2),(2,3)`), never by a declared geometric/causal criterion. Relabeling `WORLD_2`'s events with permutation `(2,0,1,3)` changes the coupling graph to `{(0,2),(0,1),(1,3)}` -- only one edge is shared, and every observable component (pairwise MI, multipartite information, causal estimate, intervention-response tensor) changes once mapped back to original-event identity. On the one shared edge, the coupling angle matches exactly, showing the coupling LAW itself is not the problem.
- Exhaustively enumerating all 24 relabelings of `WORLD_2` shows mean test loss is only weakly correlated (0.14) with a simple index-displacement feature, and adjacency-preserving vs adjacency-breaking permutations have similar group means (13.70 vs 12.11) -- the effect is structural/discrete (which edges get coupled), not a smooth function of "how shuffled" the labels are.
- Four explicit non-geometric controls (A: randomized labels, B: same graph/random strengths, C: same strengths/randomized order, D: same geometry/randomized indexing) all move the mean test loss substantially away from the `COUPLED` baseline (18.28), confirming reconstruction difficulty is heavily driven by non-geometric structure.

**The scientific classification remains `EXPERIMENTAL_CONFOUND_PRESENT`.** This phase explains the mechanism; it does not fix it, per instruction.
