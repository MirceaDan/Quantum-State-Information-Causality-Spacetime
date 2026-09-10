# Limitations

1. The Hilbert spaces are small qubit systems. Results do not scale automatically to field theory, gravity, or realistic observers.
2. The static causal estimate is a finite intervention approximation. Its relative-entropy form is not bounded to `[0,1]` without an additional normalization; the dynamical control experiment therefore uses trace distance and reports the measure explicitly.
3. The Page-Wootters result is a two-level analytically controlled toy model. It demonstrates conditional-state variation, not emergent physical time in nature.
4. The conformal curved metric is a tensor-calculus validation target. No stress-energy source, boundary-value problem, or Einstein-equation solution family is inferred from it.
5. Covariant divergence uses central differences for the derivative of the supplied tensor field. Step size sensitivity and refinement studies remain required.
6. The information-to-distance mapping is an exploratory modeling assumption. No claim is made that mutual information generates geometry.
7. The process-matrix interface for indefinite causal order is intentionally unimplemented.
8. There is no global optimizer, held-out geometry reconstruction, full ablation matrix, multi-seed statistical study, or speculative extension comparison yet.
9. The fixed-point experiments use linear channels. A fixed point or periodic orbit is not evidence for a closed timelike curve or time travel.
10. The historical Baba Novac/1987 target remains conceptual and is not populated with fabricated microscopic data.

These limitations are part of the result and should be resolved before making a paper-level claim about reconstruction or new physics.

## Reverse-Theseus phase limitations

11. The geometry-process coupling (partial-swap angle set by invariant-interval magnitude, local depolarizing loss set by retention) is a declared synthetic MODEL ASSUMPTION, not derived from any established quantum-gravity or holographic principle.
12. The synthetic worlds have only four relational events (six pairwise constraints); the FLEXIBLE reconstruction (17 parameters) is consequently data-starved, and the reported `OPTIMIZATION_INSUFFICIENT` classification may reflect this small-N regime rather than a fundamental information limit -- the experiments do not yet distinguish these two possibilities with a scaling study.
13. The FLEXIBLE model's metric family (`exp(2 scale x^0) eta`) matches `WORLD_2`'s construction but not `WORLD_3`'s spatially-conformal metric; `WORLD_3`'s curvature-invariant error is therefore expected to reflect representation mismatch as well as fit quality.
14. The causal-response probe replays the full deterministic construction under a bit-flip intervention; it is a finite single-intervention approximation to the operational causal observable in the specification, not the full multi-intervention tensor `C_{i,j,a}` for an arbitrary family `a`.
15. `F_geom(eta)` is an explicit benchmark combination of causal-F1 and coordinate error, chosen for this experiment; it is not a general-purpose or physically motivated reconstruction score.

## Experimental-integrity audit findings (added after the follow-up audit phase)

16. The pipeline is NOT invariant under a pure relabeling of a world's events (section 5 label/order confound): relabeling `WORLD_2` changed the mean test loss by roughly a factor of 3 across the same 10 seeds. This means the earlier phase's apparent positive signal (COUPLED beating a pairwise-permutation randomized control) cannot currently be attributed cleanly to geometry rather than to event index/order; the honest classification is `EXPERIMENTAL_CONFOUND_PRESENT`, not a positive result.
17. `ZERO_COUPLING` (no geometric information at all) can produce a LOWER mean test loss than `COUPLED`, because destroying all coupling makes the mutual-information target degenerate (near-uniform), which is trivially easy for the nonlinear fit to satisfy. A lower loss is therefore not by itself evidence of better reconstruction; this is exactly the kind of artifact the control matrix is designed to expose, and it is reported rather than hidden.
18. `N in {12, 20}` in the required scaling-study specification were NOT executed: the dense qubit density-matrix simulation used throughout this repository scales as `4**event_count` in memory (and worse in time for the kron/matrix-multiply-heavy coupling and causal-probe replay), making `N=12` (4096x4096 complex matrices) slow and `N=20` (over a million x million) computationally infeasible with the current architecture. `historical_scaling.py` supports arbitrary N structurally; only `N in {4, 6, 8}` were actually run and reported.
19. The observable-attribution ablation (`MI_ONLY`/`CAUSAL_ONLY`/`MI_PLUS_CAUSAL`/`MULTIPARTITE_ONLY`/`ALL_OBSERVABLES`) shows very similar test losses and identical causal-F1 across all five modes in the current run; this may reflect genuine lack of differential information content, or may reflect that the small event count (N=4) leaves too few held-out pairs to distinguish the modes statistically. The experiment does not yet disambiguate these two possibilities.
20. The stronger inverse-model identifiability test described in section 4 of the integrity-audit specification (searching reconstruction-model parameterizations for geometrically-inequivalent worlds with equivalent observables) is explicitly NOT implemented; only the weaker `OBSERVABLY_DISTINCT`/`OBSERVABLY_INDISTINGUISHABLE` raw-vector-distance test is available.
