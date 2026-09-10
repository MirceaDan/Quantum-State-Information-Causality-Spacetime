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

## Experimental-integrity / identifiability audit (follow-up phase, added 2026-09-10)

This phase does not change the coupling law, the synthetic worlds, or model capacity; it audits the previous phase's positive-looking signal for leakage and confounds, per an explicit user-flagged concern that `_residuals()` consumed `observables.causal_estimate` globally despite receiving `selected_pairs`.

| Component | Classification | Evidence and limitation |
|---|---|---|
| `historical_reconstruction.ObservableSplit` / `build_observable_split` / `assert_causal_subset` | `IMPLEMENTED_AND_VALIDATED` | Pairwise, causal, and multipartite splits are derived from one seed; `assert_causal_subset` is invoked inside the `least_squares` residual closure itself and is exercised by dedicated leakage tests (`tests/test_historical_reconstruction.py`), including a test that a full unmasked causal matrix correctly FAILS the same assertion. |
| Process-architecture controls (`ZERO_COUPLING`, `SHUFFLED_GEOMETRY`, `RANDOMIZED_ORDER`) | `IMPLEMENTED_AND_VALIDATED` | All four controls run through the identical reconstruction pipeline; `RANDOMIZED_ORDER` is tested to execute every adjacent-pair gate exactly once; `SHUFFLED_GEOMETRY` is tested to change the coupling-angle assignment while leaving the ground-truth geometry untouched. |
| `run_label_permutation_control` / `build_relabeled_world` | `IMPLEMENTED_AND_VALIDATED` -- **FOUND A REAL CONFOUND** | Relabeling `WORLD_2`'s events changed the mean test loss by roughly 3x across identical seeds, well outside the declared tolerance. This is the single most important audit finding: the previous phase's apparent positive result is not label/order-invariant and must not be reported as `HISTORICAL_RECONSTRUCTION_SUPPORTED` or similar. |
| `historical_scaling.py` | `IMPLEMENTED_BUT_PARTIALLY_RUN` | Structurally supports arbitrary N; only N in {4,6,8} executed due to `4**N` dense-simulation cost. N in {12,20} are unexecuted, not silently faked. |
| `run_intervention_family_robustness` | `IMPLEMENTED_AND_VALIDATED` | X and Z both produce a strictly forward-only causal-response tensor (tested); their FLEXIBLE test losses differ substantially, indicating the causal signature is not yet stable across intervention families. |
| `compute_scientific_gate` / `classify_historical_reconstruction` | `IMPLEMENTED_AND_VALIDATED` | Classification vocabulary and gate logic are fixed in code (section 9/11/12 of the spec) before the final experiment matrix was run; a failing gate deterministically forces `EXPERIMENTAL_CONFOUND_PRESENT` regardless of any other metric, which is what the final run actually produced. |

## Final audit classification

Running `run_integrity_audit.py` on `WORLD_2` at 10 seeds: `train_test_leakage=PASS`, `causal_leakage=PASS`, `geometry_leakage=PASS`, but `order_confound=FAIL` and `label_confound=FAIL` -> **`EXPERIMENTAL_CONFOUND_PRESENT`**. This supersedes the previous phase's `OPTIMIZATION_INSUFFICIENT` reading. The strongest currently defensible statement is: *the geometry-coupled construction's previously observed advantage over a randomized control does not survive a leakage-free, order/label-confound-checked protocol in the current small-N implementation.* No `HISTORICAL_RECONSTRUCTION_SUPPORTED`, `GEOMETRY_DEPENDENT_SIGNAL_DETECTED`, or similar positive conclusion is reachable while this gate fails.

## Root-cause analysis of the order/label confound (added 2026-09-10, diagnosis only)

This phase changed nothing about the coupling law, the synthetic geometries, `Q`, `Delta_munu`, or the reconstruction optimizer. `historical_diagnostics.py` and `run_root_cause_analysis.py` were added purely to explain *why* `order_confound`/`label_confound` failed and why `ZERO_COUPLING` could outperform `COUPLED`. The scientific classification remains **`EXPERIMENTAL_CONFOUND_PRESENT`** -- this section explains, it does not repair.

**Root-cause classification: `MULTIPLE_CONFUNDS`** (`process_structure_confund=True`, `label_confund=True`, `order_confund=False`, `split_design_confund=True`), from `run_root_cause_analysis(WORLD_2, seeds=range(5), permutation=(2,0,1,3))`.

1. **`PROCESS_STRUCTURE_CONFUND` (dominant, root mechanism).** The fixed-order construction couples physical event pairs strictly by raw index adjacency: step `k` always couples events `k` and `k+1`, i.e. the coupling GRAPH is `{(0,1),(1,2),(2,3)}` by construction, never by any declared geometric or causal criterion. Under the audit permutation `(2,0,1,3)`, the coupling graph in original-event identity becomes `{(0,2),(0,1),(1,3)}` -- only edge `(0,1)` is shared. `compare_observables_under_relabeling` confirms: `coupling_graph_preserved=False`, and pairwise MI (max abs diff 1.23), causal estimate (Hamming distance 5/12), multipartite information, and the intervention-response tensor ALL change after mapping the relabeled observables back into original-event index space. Critically, on the one edge that IS shared between the two constructions, the coupling angle matches exactly (`angles_on_shared_edges_match=True`): the coupling LAW `theta = f(interval)` itself is relabeling-invariant. The confound is entirely about *which pairs get coupled at all*, not about the formula applied to a given pair.
2. **`LABEL_CONFUND`.** The initial computational-basis state is assigned by raw index parity (`zero if index % 2 == 0 else one`), independent of any geometric quantity; combined with (1), this means relabeling changes both which pairs interact AND which physical event starts in `|0>` vs `|1>` for a given coupling step -- detected the same way as (1), since it directly contributes to the observable changes above.
3. **`SPLIT_DESIGN_CONFUND`.** The train/validation/test pair split (`build_observable_split`) is defined on raw integer pair indices (e.g. `(0,1)`, `(1,2)`, ...), not on physical event identity. A relabeling therefore changes which REAL geometric relations land in the training vs held-out set even though the split *procedure* is unchanged and leakage-free in the narrow sense audited previously. `run_label_permutation_control` on `WORLD_2` reports `LABEL_OR_ORDER_CONFOUND_DETECTED` and control D (`same geometry, randomized indexing`) gives a materially different mean test loss (6.09) than the unpermuted `COUPLED` baseline (18.28).
4. **`ORDER_CONFUND` was NOT flagged at the aggregate level.** Exhaustively enumerating all 24 permutations of `WORLD_2`'s four events and correlating mean test loss against a purely combinatorial index-displacement feature gives a weak correlation (0.14) and adjacency-preserving vs adjacency-breaking group means that are actually close (13.70 vs 12.11, ratio 0.88) with high within-group variance. In other words, loss is NOT cleanly predicted by a simple "how shuffled is the labeling" feature once averaged over many permutations -- the effect is structural (which specific edges get coupled; see (1)) rather than a smooth monotonic function of displacement from the identity permutation.

**Non-geometric controls A-D** (mean FLEXIBLE test loss, `WORLD_2`, seeds 0-4; COUPLED baseline = 18.28):

| Control | Description | Mean test loss |
|---|---|---|
| A | same observable values, randomized event labels | 1783.06 |
| B | same coupling graph, randomized (non-geometric) coupling strengths | 9.79 |
| C | same coupling strengths, randomized execution order | 185.30 |
| D | same geometry, randomized event indexing | 6.09 |

Control B (same graph, random strengths) gives a loss *closer* to baseline than A, C, or D -- consistent with (1): once the coupling graph is fixed, the specific strength values matter comparatively less than which pairs are coupled and how the pipeline indexes/splits them.

**What this means for the earlier `ZERO_COUPLING`-beats-`COUPLED` finding:** with `force_zero_coupling=True` the pairwise-MI target degenerates toward a near-uniform value (see LIMITATIONS.md #17); combined with the structural confounds above, the comparison between `COUPLED` and `ZERO_COUPLING` mean test loss is not a clean geometry-vs-no-geometry comparison in the current implementation.

**What must change before re-attempting reconstruction (not done in this phase):** the coupling schedule must be decoupled from raw array index (e.g. coupling determined by an explicitly declared, geometry- or causal-order-derived adjacency rather than literal array position), the initial-state assignment must not depend on index parity, and the train/validation/test split must be defined in a way that is provably invariant under event relabeling. None of these changes were made here, per the instruction to explain the problem before repairing it.
