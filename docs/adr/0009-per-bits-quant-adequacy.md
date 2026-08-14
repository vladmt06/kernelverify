# ADR 0009: Per-bits ensemble adequacy for the quantization contract

Date: 2026-08-14
Status: Accepted; the fp16 clause's open condition was discharged the same day, stated at the end

## Context

Phase 0 calibrated K = 3 for the MLX-affine quantization contract at bits = 4 and nothing else.
Sub-4-bit kernel work needs to know whether that number, and the ensemble behind it, hold at other widths.

ADR 0005 changed what the right question is.
On the corpus surface K turned out to be nearly inert: across 9,678 (case, fault) pairs not one verdict moved anywhere between K = 1.0 and K = 6.0, while the ensemble membership moved the floor by up to 8.45x.
The same shape holds here, where the smallest artefact fault sits hundreds of times above tolerance.
So the deliverable is per-bits ensemble adequacy, with K held fixed and secondary.

## Decision 1: the rule was pre-registered, and then it had to be amended

The rule was agreed before the harness ran, and `bench/calibrate_quant_bits.py` carries it in its docstring.
Its second clause scored leave-one-class-out as a pass or fail gate.
That clause was wrong, and the control width found it.

No ensemble can cover a class it holds no member of, because that is what an unrepresented class means.
Used as a pass or fail test, leave-one-class-out therefore fails every ensemble whose classes are genuinely distinct, and fails hardest when they are most distinct.
It measures the opposite of adequacy.

The amendment was made after the bits = 4 control and before any of widths 2, 3 and 8 ran.
That ordering matters and is recorded rather than asserted: the control was a calibration case whose answer Phase 0 had already pinned, not an experimental width.
The harness prints both readings of every width, so the amendment is auditable from its output without a rerun.

| Width | Verdict under the gate as first written | Verdict under the amended gate |
|---|---|---|
| 2 | INADEQUATE | ADEQUATE |
| 3 | INADEQUATE | ADEQUATE |
| 4 (control) | INADEQUATE | ADEQUATE |
| 8 | INADEQUATE | ADEQUATE |

## Decision 2: the amended gate

- **G0** the numpy canonical quantizer is bit-exact against `mx.quantize` at that width.
- **G1** zero false positives for the full ensemble, on held-out correct implementations and on an independent seed draw.
- **G2** every non-equivalent artefact fault caught on every iid-mode case, at 10x margin or better.
- **G3** every class carries at least two members, counted on distinct members rather than on names.

Leave-one-class-out becomes the class-necessity diagnostic.
A ratio above 1 means the class is load-bearing and must be kept; it can never fail an ensemble.

G3 stays a gate in its own right rather than folding into the diagnostic, because it is the precise artifact that produced K = 7.02 on this surface and K = 2.119 on the corpus surface, and a diagnostic would let it drift.
Counting distinct members is not pedantry: `lut-gather` returns bit-identical output to `dequant-pairwise` at bits 2, 3, 4 and 8, so a name count would credit the dequant class with a member that cannot move the floor.
The ensemble holds five numerically distinct members, not the six it names.

## Gate 0 generalised further than the question asked

The Phase 0 unpacker read MLX's packed weights as whole words, at 32 // bits values per word.
`mx.quantize` writes one contiguous little-endian bit stream per row and then reinterprets it as uint32, and the two readings agree only when bits divides 32.
The unpacker was therefore correct at bits 2, 4 and 8 and wrong at 3, 5 and 6, where a value straddles a word boundary.

Left in place it would have been worse than a crash.
`verify_canonical_against_mlx` falls back to treating MLX's output as canonical whenever the comparison fails, so a misread would have silently replaced the verified reference with a misread one at exactly the widths under test.

With it fixed, `canonical_quantize` reproduces `mx.quantize` exactly - codes, scales and biases - at bits 2, 3, 4, 5, 6 and 8, where Phase 0 had established bits = 4 alone.

## Result: adequate at every width

192 records per width: 3 shapes, 2 weight draws, 4 seeds of which 2 are the independent draw, 4 input modes, 2 activation dtypes.
The shape set is `(512, 512)`, `(1024, 4096)`, `(4096, 4096)`; Phase 0's `(2048, 11008)` is dropped and `(1024, 4096)` added, because this sweeps four widths where Phase 0 ran one, and `--full-shapes` restores it.

| bits | G0 | G1 false positives | G2 worst margin | G3 distinct per class | Equivalents | Verdict |
|---|---|---|---|---|---|---|
| 2 | pass | 0 | 161.4x | 3 / 2 | 0 | ADEQUATE |
| 3 | pass | 0 | 278.5x | 3 / 2 | 0 | ADEQUATE |
| 4 | pass | 0 | 266.4x | 3 / 2 | 0 | ADEQUATE |
| 8 | pass | 0 | 272.1x | 3 / 2 | 0 | ADEQUATE |

Worst iid-case margin per artefact fault, against a 10x gate:

| Fault | b=2 | b=3 | b=4 | b=8 |
|---|---|---|---|---|
| bias-dropped | 405x | 384x | 396x | 396x |
| group-size-halved | 321x | 313x | 325x | 324x |
| all-groups-read-group-0 | 309x | 318x | 318x | 319x |
| scales-rotated-one-group | 298x | 318x | 301x | 312x |
| nibble-order-swapped | 282x | 279x | 266x | 277x |
| unpack-mask-too-wide | 161x | 282x | 302x | 272x |

The tightest cell is the widest-mask fault at bits = 2, and legibly so: with only four levels, widening the mask by one bit perturbs a code by a smaller fraction of its range than it does at any other width.

Two faults are new here, measured by the kernel lane and credited to them: the unpack mask one bit too wide, and every group reading group 0.
The second is distinct from `scales-rotated-one-group`, which is off by one group and still varies down the row, where this one collapses the row onto a single group's parameters and so survives any test whose weights happen to be group-uniform.

## The class-necessity diagnostic

Worst member-to-tolerance ratio when a whole class is removed from the floor:

| bits | Drop int-domain (4 dequant members left) | Drop dequant-domain (2 int members left) |
|---|---|---|
| 2 | 5.578x | 1.066x |
| 3 | 3.422x | 1.292x |
| 4 | 3.578x | 1.226x |
| 8 | 2.380x | 1.163x |

Both classes are load-bearing at every width, and the int-domain class increasingly so as the quantization coarsens.
The failures concentrate on constant-rows at float32.
That is the zero-variance regime for the third independent time, after ADR 0003 found it broke the published tolerances and ADR 0004 found it was the only case in 4,060 where the corpus floor binds.
Integer-domain accumulation, `s * sum(x*q) + b * sum(x)`, diverges from dequantize-then-multiply precisely where the input carries no variance for the two groupings to agree on.

## The intermediate-precision boundary is conditioned on activation dtype and on bits

The fp16-dequant boundary case dequantizes to fp16, as some shipped kernels do, and then accumulates in fp32.
Whether it sits inside or outside tolerance decides whether the contract's intermediate-precision clause is numerically enforced.

Cases outside tolerance, and therefore enforced:

| bits | float32 activations | float16 activations |
|---|---|---|
| 2 | 39 / 96 | 0 / 96 |
| 3 | 96 / 96 | 0 / 96 |
| 4 | 96 / 96 | 0 / 96 |
| 8 | 96 / 96 | 0 / 96 |

Two facts, and the second is the one sub-4-bit work needs on the record.

At float16 activations the clause is unenforced at every width, because the absolute base tolerance swamps the fp16 dequantization error.
Phase 0's "half of cases" was exactly the float32 half, and reading it as a bits effect would have been wrong.

The bits dependence runs the other way from the natural guess.
Fine quantization at bits = 8 does not soften enforcement; coarse quantization at bits = 2 does.
A coarser grid means larger scales, a larger legitimate ensemble floor, and therefore a tolerance wide enough to swallow the fp16 dequantization error even at float32 activations.
Within bits = 2 the loss concentrates where variance vanishes: 21 of 24 constant-rows cases go dark, against 12 of 24 in each iid mode.
The verifier's floor does more work exactly where the pack is heading.

## The open condition: is the precision clause enforceable at fp16 at all

The kernel lane measured a shipped wide-tile matvec with only its accumulator moved from float to half - out of contract by construction - passing the shipped tolerance on 4 of 4 probed cases, at typical operating points.
The proposal on the table was to retire the clause at fp16 to a source-level structural attestation.

That is premature, and the corpus surface says why.
The analogous precision fault there, `attention[scores_dtype=float16]`, is out of contract in exactly the same way, and it is separable - but only under particular structure:

| Input mode | Caught, fp32 | Caught, fp16 |
|---|---|---|
| opposed signs | 28 / 44 | 14 / 44 |
| constant rows | 20 / 44 | 13 / 44 |
| corpus uniform[-10,10] | 21 / 44 | 7 / 44 |
| unit-scale uniform[-1,1] | 0 / 44 | 0 / 44 |
| near zero | 0 / 44 | 0 / 44 |

On that surface the clause is numerically enforceable at fp16, on 34 of 220 fp16 cases, and enforceable on exactly zero cases under unit-scale or near-zero iid input.
Typical-point probing lands in the regime that provably cannot separate it.
A null result from there does not distinguish inseparable from unprobed.

The ruling, from the tolerance lane, is conditional.
The clause stays numeric until a battery that includes opposed-signs and constant-rows at the largest reduction length, at fp16 activations, fails to separate the accumulator seam on any case.
If such a run separates nothing, structural attestation is accepted for quantized matmul and the certificate must then distinguish numerically-attested clauses from structurally-attested ones, per dtype.
A relative tolerance component is worth building for the vacuity that ADR 0004 already identified, but it does not fix this: no tolerance of any shape separates two distributions that overlap, and a relative term would not have moved a single one of the zero rows above.

## The open condition, discharged: enforceability follows accumulation topology

The battery run the clause demanded landed as `bench/probe_accum_separation.py` (catalogue/accum-dtype-fault, commit b3fc3b8): 560 cases over the quantized_matmul battery space, including opposed-signs and constant-rows at the battery's largest reduction length D_IN = 1024, at both activation dtypes, with zero control failures.
This lane reran the probe independently before writing this section, and every number below reproduced exactly.

The accumulator seam turned out to be two classes with opposite enforceability, split by accumulation topology:

| Half-accumulator class | Caught, fp32 activations | Caught, fp16 activations |
|---|---|---|
| Sequential (naive one-thread-per-output shape) | 280 / 280 | 192 / 280, up to 23.1x over tolerance |
| Tree (simdgroup-reduction shape) | 280 / 280 | 0 / 280 |

The mechanism is legible.
Sequential fp16 error grows with reduction LENGTH and punches past the legitimate envelope, hardest exactly where the class-necessity diagnostic pointed: constant rows at D_IN = 1024 reach 23.1x over tolerance.
Tree fp16 error grows with reduction DEPTH only, and at every depth this battery can express it stays inside the fp16 rounding envelope the conditioned floor must legitimately allow.

The clause as pre-agreed spoke of "the accumulator seam" as one thing, and the measurement shows it is not; the discharge therefore applies the clause per class rather than pretending it anticipated the split.
The sequential class separates broadly, so C1-at-fp16 stays NUMERICALLY ENFORCED against it.
The tree class separates on no case at all - under the structured modes, at the longest reduction, at every bit width - which is precisely the "separates nothing" condition, so STRUCTURAL ATTESTATION (source-level accumulator dtype) is accepted for quantized matmul for that class alone.

The certificate consequence, an input to the certificates ADR (ADR 0011): the certificate must distinguish numerically-attested clauses from structurally-attested ones per dtype and per accumulation topology, because "C1 holds" is now three different sentences - checked numerically at fp32 everywhere, checked numerically at fp16 against sequential accumulators, and attested from source at fp16 against tree accumulators.

## Consequences

Sub-4-bit kernel work is unblocked with a measured basis rather than an extrapolation from bits = 4: K = 3 and the shipped ensemble are adequate at 2, 3, 4 and 8, with the tightest fault margin 16x above the gate.
K did not move, and on this evidence it should not; membership is what carries the floor, which is the same conclusion ADR 0005 reached on the other surface.

The certificate specification gains a statement it cannot make unconditionally: the intermediate-precision clause is numerically enforced at float32 activations for bits >= 3, partially at bits = 2, and never at float16 activations at any width.

The honest caveats compound in the usual direction.
The adequacy gate was amended mid-experiment, which is recorded above with both readings and the timing so that nobody has to take the sequence on trust.
The ensemble is still self-authored, and a fifth distinct member masquerading as a sixth was found only because the members were compared rather than counted.
