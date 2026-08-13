# ADR 0002: Scoring oracles by mutation, and the battery policy that won

Date: 2026-08-13
Status: Accepted; the battery policy and result tables are superseded by ADR 0003

## Context

ADR 0001 established that escape is a property of the pair (bug, test case), not of a bug alone.
That reframes the product question: given a budget of kernel evaluations, which test cases should an oracle spend it on?
Answering that requires scoring test policies against a fault population, which is component 6 of the plan and the part no competitor ships.

## Method

The fault population is synthesised, not curated.
Every operator lives once in `kernelverify/reference/kernels.py`, correct by default, with keyword seams where a realistic transcription fault can be introduced.
`kernelverify/mutation/catalogue.py` walks those seams to produce 35 faults: dropped factors, misremembered constants, missing square roots, wrong padding values, assigned accumulators, short loop bounds.
The 10 hand-written faults from the gpuemu corpus fall out as a labelled subset, because `bench/cpu_ports.py` pins the same keywords.
This guarantees the synthetic population contains the published faults instead of merely resembling them.

Verdicts for every (fault, case) pair are computed once against the corpus's own fp64 references and per-operator tolerances.
Every policy is then charged the same budget of kernel evaluations per operator, and scored on the fraction of faults it detects.
Stochastic policies are averaged over 40 seeded runs.

## Equivalent mutants are excluded, with reasons

3 of 35 faults are undetectable at every case in the whole space, and scoring against them would deflate every policy equally.

| Excluded fault | Why no oracle can see it |
|---|---|
| GELU coefficient truncated to 0.0447 | The change is below fp32 resolution at every tested input |
| Softmax padding filled with -1e9 instead of -inf | exp(-1e9) and exp(-inf) are both exactly zero |
| Softmax without max-subtraction | Mathematically identical until inputs are large enough to overflow, which uniform[-10,10] never is |

The attention variant of the max-subtraction fault stays in, because attention scores grow with D and do overflow at corpus scale, which is also why it is the hardest fault in the set.

## The first run falsified our own policy

The first coverage policy was deterministic: greedy feature cover with ties broken toward small shapes.
It plateaued at 88% detection while plain random sampling reached 98% at B=32.
Its miss list showed why: faults that express only at rare combinations, such as the flash-attention rescale needing several tiles of sequence, reward exploration and punish any fixed list, and the small-shape tie-break pointed away from exactly those cases.

## Decision: a boundary core plus random exploration

The replacement policy has two parts, and nothing in it refers to any known fault.

The deterministic core covers the boundaries where the mechanisms from ADR 0001 live, on both ends at once.
Each dimension's smallest candidate is covered, because dilution gives small reductions the strongest absolute signal and degenerate sizes are where accumulator faults hide.
Each dimension's largest candidate is covered, because tiling faults need a sequence spanning several tiles before they express.
A non-power-of-two value is covered per dimension, because block padding does not exist without one.
Both dtypes and both input scales are covered, because tolerance width and saturation each flipped verdicts in ADR 0001.
The core is a greedy set cover over those features, finishes in 4 to 6 cases, and never repeats itself.
The rest of the budget is uniform random over the whole declared space.

## Result

Detection rate across the 32 viable faults, at equal budget B per operator:

| Policy | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| Single shape, fp32 (what benchmarks do) | 78% | 78% | 78% | 78% |
| Random schema (the corpus's own fuzzer) | 89% | 93% | 94% | 95% |
| Random schema, both input scales | 91% | 95% | 97% | 98% |
| Boundary core + random (ours) | 98% | 99% | 100% | 100% |

On the 10 published corpus faults alone, the boundary hybrid is at 100% from B=4.
The last fault to resist it is attention without max-subtraction, missed in 13 of 40 runs at B=8 and never at B=16, because its overflow needs the largest dimensions and the aggressive input scale simultaneously and the core covers features independently rather than in pairs.

## Consequences

The sales claim now has measured numbers behind it: a standard single-shape harness catches 78% of realistic faults and more budget buys nothing, while the boundary hybrid reaches 100% of this population at 16 evaluations per operator.
The honest caveat attached to that claim is that the fault population, while larger than anything published, is still written by us; extending the catalogue faster than the policy adapts is the ongoing discipline that keeps the number honest.
Pairwise feature coverage is the known next improvement if future faults keep hiding at feature intersections, and it should be adopted only when a measured miss demands it.
