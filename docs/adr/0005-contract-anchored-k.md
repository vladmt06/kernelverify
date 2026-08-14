# ADR 0005: Anchoring K to a written contract, and the ensemble premise it falsified

Date: 2026-08-14
Status: Accepted

## Context

ADR 0004 ships `tolerance(case) = max(base_tol, K * ensemble_floor(case))` with `K = 1.5`.
The floor is the worst deviation of three hand-written correct implementations, and K was the smallest value on a grid that produced no false positive over a handful of held-out variants.

That procedure never states what K would have to be to be safe.
It states only that 1.5 was not observed to be unsafe against the few implementations we happened to write, which is the same weakness this project already refuses to accept for the fault catalogue: a number fitted to a small self-authored population.

The fix is to write down what the verifier actually promises, and then measure what that promise demands.

## Decision 1: the admissible-implementation contract is written down

`kernelverify/tolerance/contract.py` states the class of implementations the verifier undertakes never to flag.

| Clause | The freedom granted | Why it is in |
|---|---|---|
| C1 | Every intermediate in binary32 or wider; the result rounded to the storage dtype exactly once | The working-precision convention the corpus Triton kernels use |
| C2 | Any association order and any permutation of any reduction, and any reassociation of the specified expression | The verifier is black box over the candidate and cannot observe the order it used |
| C3 | Any tile width, including online-softmax and padded-block formulations | Tile width is a free parameter of the shipped algorithms |
| C4 | Exponential sums must be shifted by a maximum over their own summands | Stated as a requirement, not a freedom; see below |
| C5 | The formula is the one the corpus fp64 reference computes | A different but defensible function is a different operator |
| C6 | Epsilons, scale exponents and coefficients belong to the specification | They are not the implementer's to choose |

The three exclusions carry the weight, and each has a catalogue fault standing on it.
C1 excludes narrow intermediates, which is what keeps `attention[scores_dtype=float16]` a fault rather than a legal variant.
C4 excludes unshifted exponential sums even though they are exact in real arithmetic, because they overflow: admitting them would leave the class with no finite worst case to take a maximum over, and the tolerance derived from it would be vacuous.
C5 excludes the exact-erf GELU, which is a correct function and an inadmissible implementation of a tanh-approximation GELU.

A contract wide enough to admit everything forces a tolerance that absolves everything, so each exclusion is a promise the verifier is deliberately allowed to break.

## Method

`bench/calibrate_k.py` samples the class and measures it against the same fp64 references the verifier uses.

For every case in the battery it computes the worst deviation over a population of admissible implementations, and the smallest K for which `max(base_tol, K * floor)` covers that worst deviation.
The population is drawn twice under independent seeds: the floor is built from the first draw and scored against the second, so no ensemble is ever credited for covering the samples it was made of.
Errors are stored rather than verdicts, so sweeping K afterwards is arithmetic and costs no further reference calls.

The battery is 4,060 cases, and 322,200 admissible implementations were evaluated across it.
ADR 0004 records that battery as 4,100 cases; the exact figure is 4,060 and the earlier number was a rounding in prose, not a table cell.

## Result 1: K = 1.5 is sound, and the anchor rests on one case

Zero false positives, at every ensemble membership tested, over the whole population.

The class binds on exactly 2 of 4,060 cases: attention and flash attention at D=64, M=8, N=256, float32, constant rows.
Everywhere else the base tolerance already exceeds anything the contract class can produce, so the value of K decides nothing.
Under the ADR 0004 ensemble those cases demanded `K >= 1.147`; under the ensemble adopted below they demand exactly `K >= 1.000`.

## Result 2: K is nearly inert, which is the more useful finding

Detection across the 45-fault catalogue, sweeping K with the ensemble fixed:

| K | Viable faults | (case, fault) detections | fp16-score canary | Control failures |
|---|---|---|---|---|
| 1.000 | 40 | 9,678 | 103/440 | 0 |
| 1.500 (shipped) | 40 | 9,678 | 103/440 | 0 |
| 3.000 | 40 | 9,678 | 103/440 | 0 |
| 6.000 | 40 | 9,678 | 103/440 | 0 |
| 10.000 | 40 | 9,677 | 102/440 | 0 |

Nothing moves between K = 1.0 and K = 6.0.
K is not a delicately tuned parameter sitting on a cliff; it is a wide plateau, and the risk in this oracle does not live there.

This is also the reason K stays at 1.5 rather than being raised into the middle of the plateau.
Detection flatness is measured over the current 45 faults only, and buying margin on the strength of a population that cannot price it is the exact move this project's rules exist to prevent.

## Result 3: the ensemble premise was false, and that is where the risk lives

ADR 0004 built the floor from "worst-legitimate-order variants: sequential (non-pairwise) accumulation", on the premise that bounding the sequential order bounds the class.

Measured, that premise does not hold.
Seeded random permutations of the same reductions exceed the ADR 0004 ensemble floor by up to 8.45x on the attention family, and the member that sets the worst case is a permutation, never the magnitude-sorted order that theory nominates as the worst summation order.
The worst instance is flash attention at D=64, M=2, N=64, float32, corpus inputs: ensemble floor 1.76e-5 against a class floor of 1.49e-4.

So K = 1.5 was not covering a class with 1.31x of margin.
It was partly compensating for an ensemble that did not represent the class it stood for, and it stayed sound only because the base tolerance dominates on every case where the gap is large.

## Decision 2: the ensemble is a prefix of the contract, at a budget of eight

Floors built from one draw of the population, scored against the independent draw:

| Ensemble | Members | Max K demanded | Worst under-coverage | False positives at K=1.5 | Verdicts moved |
|---|---|---|---|---|---|
| ADR 0004, hand-written | up to 3 | 1.147 | 8.453x | 0 | 0 |
| Contract prefix | 1 | 2.119 | 9.974x | 1 | 0 |
| Contract prefix | 2 | 1.000 | 8.453x | 0 | 0 |
| Contract prefix | 6 | 1.000 | 5.365x | 0 | 0 |
| Contract prefix | 8 (adopted) | 1.000 | 3.203x | 0 | 0 |
| Contract prefix | 12 | 1.000 | 2.327x | 0 | 0 |
| Contract prefix | 24 | 1.000 | 2.000x | 0 | 0 |
| Contract prefix | 32 | 1.000 | 2.000x | 0 | 0 |

Under-coverage is measured only on cases where the class floor reaches a tenth of the base tolerance, because a ratio between two quantities four orders below the tolerance says nothing about the oracle.

Three things decide the budget.
A single-member floor is genuinely insufficient: it demands K = 2.119 and ships a false positive at the shipped K, which is the measurement that justifies having an ensemble at all.
From two members on, the held-out class demands exactly 1.000, so K stops absorbing ensemble error and becomes margin over it.
Under-coverage falls fastest up to eight members and then flattens, reaching 2.000x only at twenty-four, at three times the per-case cost.

The ordering inside the prefix is itself a decision, and it is stated in the module: the reference implementation first so a one-member floor is the status quo ante, then a structurally different formulation of the same operator, then tile-width extremes, then the deterministic adversarial orders, and only then the random permutations.
The remaining deterministic members come last because none of them can bound the class: Kahan and binary64 are strictly more accurate than the reference, ascending-magnitude is the best order rather than a bad one, and the leftover tile widths sit between pairwise and sequential, both already carried.

Not one (case, fault) verdict out of 9,678 changes at any budget in that table.
The full policy rerun confirms it end to end: 95.0% / 97.5% / 100.0% / 100.0% at B = 4/8/16/32 for the boundary-pairs policy, 100% on the corpus subset from B=4, the same 5 equivalent mutants, and the same systematic small-budget survivors.
Every table in ADR 0002 through ADR 0004 stands unchanged.

The run also completed without a single control assertion, which is the integrated proof that the eight-member floor clears every correct implementation on all 4,060 cases.

## Decision 3: the verdict-cache fingerprint carries the ensemble, not just K

ADR 0004 stated that the cache fingerprint carries an oracle tag so an oracle change can never silently reuse stale verdicts.
The tag encoded K alone.
Swapping who is in the ensemble changes every floor while leaving the tag identical, so the larger half of the tolerance was unfingerprinted, and this ADR's own change would have been the first to exercise the hole.
The tag now carries the contract version and every ensemble member's label.

## Consequences

K = 1.5 is anchored rather than fitted, and the sentence behind it is now checkable: over 322,200 admissible implementations across the battery, the class demands at most 1.000 under the shipped ensemble, with zero false positives.
The calibration is reproducible from the repository, which the original K = 1.5 was not, because ADR 0004's grid search left no committed script.

The known hole, stated rather than papered over: the shipped floor still under-covers its own contract by up to 3.203x on cases where the floor is within an order of magnitude of binding, and 3.203 exceeds K.
No case in this battery is both binding and order-sensitive, which is the only reason that gap has no consequence today.
The pre-registered trigger for revisiting is the first measured case with a class floor above its base tolerance and an under-coverage ratio above K; the response is a larger prefix, not a larger K, because the budget sweep shows more members buy coverage while more K buys nothing measurable.

Two limits are structural rather than budgetary.
On 54 cases the shipped ensemble is bit-exact against the fp64 reference while some legal implementation is not, so no multiple of the floor could ever cover the class there; those are fp16 output-rounding cases and the worst sits at 0.078x the base tolerance, so the absolute base tolerance carries them.
And a sampled population gives a lower bound on the class supremum, never the supremum itself, so every number here is a floor on what the contract demands.

Triple-checking against a second surface was not needed to reach these conclusions, but a parallel lane independently reproduced the ordering mechanism on a quantized matvec, reporting that 19% of admissible permutations exceed a hand-picked worst-order floor there, by up to 1.94x on a single reduction against this lane's 8.45x on chained ones.
That measurement has not been re-run here and is recorded as corroboration, not as evidence.

Tightening C2 to the orderings a device can actually emit was considered and rejected.
It would buy detection margin in principle, but the measured detection sensitivity of this floor is zero across every ensemble budget from 1 to 32 and every K from 1.0 to 6.0, so there is no margin here to reclaim, and a verifier's promise should be stated over what is admissible rather than over what current hardware happens to produce.
