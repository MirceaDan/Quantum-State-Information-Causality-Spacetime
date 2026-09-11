# Relational Dynamics Toy Model

Reference implementation of the attached computational ansatz. Components are explicitly tagged as established physics, mathematical definitions, modeling assumptions, speculative extensions, or numerical approximations.

The project follows the staged development order in the specification. Tests are written before the full implementation and are the executable contract.

## Development

```powershell
python -m pip install -e ".[dev]"
pytest
```

The external simulation clock, when used by future optimizers, is a numerical control variable only. It is not stored as a physical degree of freedom.

## Project status

**STATUS: PAUSED**

This repository is an exploratory computational research project. The work below was deliberately stopped after the M5 validation phase. The project did **not** produce a new physical law, an emergent-spacetime result, a time-travel mechanism, or a validated quantum-gravity model.

The purpose of preserving the repository is reproducibility and future reference: the failed approaches, controls, confounds, and negative results are part of the scientific record and should not be lost.

## Research question

The original ansatz asked whether quantum state, information, and causal structure could reproduce or constrain spacetime-like relational structure without treating an external classical time parameter as fundamental. The working conceptual interface was:

```text
quantum state  <->  information  <->  causal structure  <->  spacetime geometry
```

The specification explicitly required established physics and mathematics, modeling assumptions, speculative extensions, and numerical approximations to remain separated. In particular, the project never treated an information-to-distance mapping as established physics.

## Experimental history and results

### Baseline quantum / relational-physics diagnostics

The initial finite-dimensional reference implementation validated a collection of controlled toy results:

- quantum-state trace, Hermiticity, positivity, and finite-dimensional constraint diagnostics behaved at numerical precision;
- a controlled Page-Wootters-style relational-clock toy model produced a near-zero constraint residual and distinct clock-conditioned states;
- mutual information could be present without sampled direct causal influence in a GHZ/common-cause control;
- controlled dynamical examples distinguished identity (zero influence) from a one-way CNOT influence;
- Minkowski light-cone classification recovered the independently specified timelike/null labels in the toy geometry test;
- the GR tensor baseline correctly gave vanishing curvature/Einstein tensor for Minkowski and nonzero tensors for the conformal metric test;
- unitary state evolution was numerically reversible while depolarizing evolution was not;
- fixed points and periodic dynamics were correctly distinguished from causal loops.

These are validation experiments for the computational framework, not discoveries about nature.

### Reverse-Theseus / hidden-geometry reconstruction

The next phase attempted to reconstruct a hidden synthetic geometry from quantum/process observables. The construction coupled a Lorentzian invariant interval to a partial-SWAP interaction strength and tested whether different synthetic worlds could be distinguished or reconstructed.

The experiments found observable differences between the synthetic worlds, but the inverse reconstruction was unstable and data-starved. More importantly, the integrity audit exposed genuine experimental confounds in the first construction:

- raw-index-dependent process structure could masquerade as a geometry signal;
- label/order dependence changed reconstruction performance under pure relabeling;
- zero-coupling could produce deceptively low loss because the information target became degenerate;
- intervention-family choice materially affected the causal signature;
- increasing the number of relational events did not produce systematic reconstruction improvement in the tested range.

The decisive classification was **`EXPERIMENTAL_CONFOUND_PRESENT`**. The apparent positive geometry signal was therefore not accepted as evidence for information-generated geometry.

The original Reverse-Theseus construction is retained as a historical/negative-control phase, not as a validated reconstruction method.

### M4 -- Permutation-equivariant quantum process memory

M4 rebuilt the process substrate so that physical quantities were attached to declared nodes and relations rather than raw array/index arithmetic. The process supports qubit systems, explicit relation order, deterministic CPTP noise, mutual information, multipartite information, finite intervention-response observables, and a narrowly defined process-memory diagnostic.

The M4 scientific gate passed:

- permutation equivariance: **PASS**, maximum error approximately `4.6e-15` for exhaustive `N=4` permutations;
- deliberately broken label-dependent simulator: **PASS** as a negative control, with failures up to approximately `0.88`;
- reproducibility: **PASS`;
- test suite: **125 tests, 124 passed, 1 expected xfail, 0 unexpected failures**.

The M4 classification was **`IMPLEMENTED_AND_VALIDATED` for the process-memory substrate only**. It makes no claim about geometry, emergent spacetime, time travel, quantum gravity, or new physics.

The process-memory score is intentionally only a finite diagnostic: it is not presented as a rigorous or generally accepted measure of quantum non-Markovianity.

### M5 -- Hidden causal topology recovery

M5 asked a narrower inverse question:

> Given only process observables generated from a hidden causal topology, can the topology be recovered up to graph isomorphism, without geometry, coordinates, node ordering, or machine learning?

The experiment used small directed graph ensembles, including chain, branch, merge, and cyclic operational-process examples. Reconstruction was deterministic nearest-candidate matching with exhaustive permutation-aware comparison.

Important positive results:

- the underlying M4 observables remained permutation-equivariant;
- train/test seed leakage checks passed;
- negative controls passed;
- the causal intervention-response observable was substantially more informative for topology recovery than static mutual information in the small mandatory-topology experiment;
- DAG versus cyclic operational process classification was clean in the tested configuration, including successful collapse of the signal under zero-coupling controls.

However, the overall M5 scientific gate **failed** because label-invariance of the full inverse reconstruction did not pass. Therefore the exact topology-recovery accuracy was **not** promoted to a general scientific conclusion.

The final M5 implementation was classified as **`IMPLEMENTED_BUT_NOT_FULLY_VALIDATED`**. The failure was preserved rather than modifying the model until the gate passed.

An additional audit identified two important limitations of the current benchmark that must be remembered if this work is ever revisited:

1. reference processes used fixed coupling strength while clean query processes sampled coupling strengths over a range, creating a nuisance-parameter/domain mismatch in the inverse benchmark;
2. the bounded identifiability audit used a different unscaled distance from the main reconstruction path, so its `OBSERVABLY_INDISTINGUISHABLE` counts must not be interpreted as a theorem about the observable representation.

There is also a semantic limitation: the current two-qubit gate is symmetric. The direction of a declared relation enters through the execution schedule, not through an intrinsically directed quantum channel. The recovered object is therefore more accurately described as **directed topology plus schedule-mediated execution structure**.

## What was learned

The most valuable output of the project is methodological rather than physical:

1. **Correlation is not the same thing as causal influence.** Controlled quantum examples can separate the two.
2. **Permutation equivariance is a necessary scientific guard** whenever labels are supposed to be physically meaningless.
3. **Inverse reconstruction can produce convincing but spurious signals** when process structure, indexing, labels, or nuisance parameters leak into the observable construction.
4. **Negative controls matter.** Zero-coupling, shuffled-observable, broken-label, relabeling, and schedule controls exposed failure modes that a raw reconstruction loss would have hidden.
5. **A finite computational toy model can validate an implementation without validating the physical hypothesis it was designed to explore.**
6. The project did not establish that information generates geometry, that process memory generates spacetime, that causal cycles correspond to physical CTCs, or that time travel is physically possible.

## Why the project is paused

The project reached the point where further progress would require substantially more work in quantum-process theory, causal-order formalisms, identifiability, and/or much larger numerical experiments. The current experiments did not provide a sufficiently strong result to justify that additional cost.

This is therefore a **stopping point, not a failed software project**: the repository contains a tested substrate, explicit hypotheses, negative results, discovered confounds, and reproducible experimental machinery. If the project is ever resumed, it should begin by re-reading the audits and reproducing the existing gates rather than extending the milestone chain automatically.

## Scientific bottom line

```text
Tried:             YES
Implemented:       YES
Validated pieces:  YES
Interesting signal: YES, but limited to controlled toy observables
New physics:       NO
Emergent spacetime: NO
Time travel:       NO
General topology recovery: NO
Major confounds found: YES
Further work justified now: NO
Repository worth preserving: YES
```

What would have remained for research:
M6 : Test inverse identifiability

M7 : Topology → causal geometry

M8 : Lorentzian geometry reconstruction

M9 : Geometric dynamics / gravity

M10 : Causal cycles / CTC-like structures

The appropriate status for future reference is therefore:

**PAUSED — exploratory computational study completed to the current validation boundary.**