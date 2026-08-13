# ADR 0001: The published escape rate is asserted, not measured

Date: 2026-08-13
Status: Accepted

## Context

The plan for this project rests on a claim from "The Correctness Illusion in LLM-Generated GPU Kernels" (arXiv 2606.20128).
The claim is that its seeded buggy GPU kernels are all certified correct by a standard single-shape `allclose` oracle.
That claim is the reason to build a verification layer at all, so it had to be checked before any of the harness was written.

The paper's authors released their corpus and validator under MIT OR Apache-2.0.
The corpus is vendored here at commit `2f15310368118ddd8c76887675f8c6c0f76a0989`.

## What we found in the released artifact

Every kernel's `meta.json` carries a field `benchmark_verdict`.
All 26 corpus entries are hardcoded to `"pass"`, correct controls and seeded bugs alike.
No code in the released repository ever computes that field, only reads it.

Their headline metric is defined in `drivers/p1_llm_kernels.py` as
`illusion = (benchmark_verdict == "pass") and (failed > 0)`.
With the left operand hardcoded true for every entry, the metric reduces to "our validator failed this kernel".
Since the same authors wrote both the seeded bugs and the validator, the reported result is close to a tautology.

The corpus's own source comments contradict the metadata directly.
The GELU kernel is documented as "uniformly ~2x too large" and "not shape-dependent, so gpuemu should catch it on essentially every test case", while its `benchmark_verdict` is still `"pass"`.

## What we measured

`bench/measure_escape.py` measures the escape rate directly.
It uses the corpus's own fp64 reference scripts over the corpus's own protocol, its own input distribution, and its own per-operator tolerances.
The Triton kernels require CUDA, so each was ported to numpy in `bench/cpu_ports.py` with the block-padding semantics preserved, because that padding is what several bugs depend on.
The correct variant of every operator is run through the same sweep as a control, and the run aborts if any control fails.
All controls pass, and the one corpus kernel that runs on CPU unmodified matches our port bit for bit.

Single-shape benchmark oracle, at the shape a benchmark would pick, float32:

| Oracle | Measured escape rate |
|---|---|
| `allclose` atol=rtol=1e-3 | 2 of 10 |
| `allclose` atol=rtol=1e-2 | 2 of 10 |
| The corpus's own per-operator tolerances | 2 of 10 |
| Asserted by the corpus metadata | 10 of 10 |

The two that escape are both softmax tail-masking bugs.
They escape for a specific and legible reason: the benchmark shape has H=256, which is a power of two and a multiple of 128, so no padding lanes exist and the bug has nothing to express through.
The other eight are caught by margins between 100x and one million times the tolerance.

## The finding that survives, and it is a better one

Escape is not a property of a bug.
It is a property of the pair (bug, test case), and the published work never measured that surface.
Sweeping each operator's full declared schema and both input scales gives, per bug, the fraction of the test space that hides it:

| Bug | Hidden at, corpus inputs | Hidden at, unit-scale inputs |
|---|---|---|
| flash attention, missing alpha rescale | 61.1% | 78.7% |
| softmax, tail mask `other=0.0` | 51.9% | 50.0% |
| softmax, reduction padded to 128 | 68.5% | 50.0% |
| attention, missing 1/sqrt(D) | 61.1% | 26.4% |
| matmul, `acc=` instead of `acc+=` | 25.0% | 25.0% |
| the five uniform arithmetic bugs | 0 to 2.8% | 0 to 2.8% |

Three mechanisms were isolated and verified rather than assumed.

Shape dependence is the known one.
The matmul bug is invisible at K=1 and only there, which is exactly 25% of its schema.

Input scale is a second axis, and it is not monotonic.
The attention bug hides under the corpus's own uniform[-10,10] inputs because those scores saturate every softmax row to one-hot, and a saturated softmax is invariant to a positive rescale, which is precisely the seeded bug.
Under unit-scale inputs no row saturates and the hidden fraction falls from 61.1% to 26.4%.
The flash-attention bug moves the other way.
Its error is governed by how far the running maximum moves between tiles.
At corpus scale that spread averages 54, so the missing rescale is catastrophic and the error reaches 26.
At unit scale the spread averages 0.54, the rescale factor is 0.58, and the error falls to 0.048, which sits just under the fp16 tolerance of 5e-2 and hides.

Signal dilution is a third axis.
The softmax tail-mask error at H=1025 is 3.03e-5 against a 1e-5 tolerance, caught by a factor of 3.
The same bug at H=3 gives an error of 0.961, caught by a factor of 96,000.
The output magnitudes shrink as the reduction grows, so the bug's absolute signal shrinks with it, and at fp16 tolerances the large-H cases fall under the threshold.

## Decision

The "all seeded bugs escape" figure is not usable, in a pitch, a deck, or a writeup.
It does not survive contact with the artifact that is supposed to support it, and the artifact is public, so anyone can check.

Phase 0 as originally planned is cancelled.
It was "reproduce the published result on rented H100s and publish the reproduction".
The authors already published corpus, validator and five-GPU results, so an independent reproduction has no wedge, and the reproduction target turns out not to be a measurement.

Phase 0 is replaced with the measurement above, which is already done, cost nothing, and produces a claim nobody else has made: escape is a property of the test surface, and it is quantifiable per bug.

The plan's component ordering changes.
Components 1 to 4, being op schemas, shape and dtype batteries, fp64 references and tolerance calibration, already exist as working MIT-licensed software in the `gpuemu` daemon.
Rebuilding them is not a differentiator and should not be treated as one.
Component 2 gains a third axis: input scale, measured, not assumed, since it moved the hidden fraction by up to 35 points here and moved it in both directions.
Component 6, mutation scoring, remains unclaimed by anyone and is now the only component that is clearly ours.

## Consequences

We can no longer cite the published escape rate, which removes the single most quotable number the plan had.
We gain a defensible replacement that is ours, is reproducible on a laptop in about two minutes, and implies a product rather than just a benchmark.
The competitive picture is worse than the plan assumed, because four of six components are already free.
The technical thesis is unchanged: single-shape testing misses real bugs, and the fraction it misses is large enough to matter.
