# ADR 0012: Device-arithmetic members join the quant ensemble; the trigger fired and K moved to 4

Date: 2026-08-14
Status: Accepted

## Context

Phase 0 calibrated K = 3 for the MLX-affine quantization contract from CPU members alone and pre-registered the device-arithmetic branch: a correct Metal kernel exceeding the CPU-calibrated tolerance triggers ensemble re-derivation with device members, never a false-positive ship.
The device members now exist (`kernelverify/schemas/quant_device.py`): three raw-MSL implementations, correct by construction, spanning both evaluation domains on the device.
`bench/calibrate_quant_device.py` carries the pre-registered decision rule in its docstring, committed before any measurement (b4dfa82); this ADR is the reading of its run.

The run uses the ADR 0009 grid unchanged: bits {2, 3, 4, 8}, shapes (512, 512), (1024, 4096), (4096, 4096), two weight draws, calibration seeds 0/1 and independent draw 100/101, four input modes, two activation dtypes, 192 records per width.
The grid identity was verified rather than assumed: every quantity the two runs share (CPU member errors, held-out errors, fault errors, boundary errors, base tolerances) is bit-identical between this run's records and the cached ADR 0009 records, on all 768 records.
G0 passed at every width: `canonical_quantize` stayed bit-exact against `mx.quantize`.

## The pre-registered second outcome: the trigger fired

Every device implementation was measured against the CPU-calibrated tolerance `max(base, 3 * cpu_floor)` before any repair.

| bits | Case exceedances | Worst overshoot | Who |
|---|---|---|---|
| 2 | 0 / 192 | covered | - |
| 3 | 1 / 192 | 1.05x | device-factored-simd |
| 4 | 0 / 192 | covered | - |
| 8 | 2 / 192 | 1.30x | device-factored-simd |

The three exceedances in full, all at 512x512, normal-0.02 weights, constant-rows, float32:

| bits | seed | draw class | error | tolerance | overshoot |
|---|---|---|---|---|---|
| 3 | 100 | independent | 3.58e-5 | 3.40e-5 | 1.05x |
| 8 | 0 | calibration | 3.23e-5 | 2.48e-5 | 1.30x |
| 8 | 100 | independent | 3.42e-5 | 2.93e-5 | 1.17x |

So the answer to the pre-registered question is yes: a correct device implementation exceeds the CPU-calibrated tolerance, at bits 3 and 8.
The offender is always the int-domain simdgroup member, and always on constant rows at float32.
That is the zero-variance regime binding for the fourth independent time, after ADR 0003, ADR 0004 and ADR 0009's class-necessity diagnostic.
MLX's own `quantized_matmul`, still held out, exceeded on zero cases at every width, reproducing ADR 0009's G1.

A verifier shipping the CPU-calibrated tolerance against device kernels would therefore flag correct kernels of exactly the shape MLX itself ships.
The pre-registered branch fired and the ensemble re-derived instead.

## Membership before K, and the K the enlarged contract demands

Per the pre-registered repair order, the device-arithmetic class joined the floor first, unconditionally, and only then was K re-derived under Phase 0's own rule: leave-one-member-out over the full nine-name membership plus held-outs against the full floor, counting only cases above the base tolerance.

| bits | Calibration draw | Independent draw | Grid value from calibration | Independent-draw miss |
|---|---|---|---|---|
| 2 | 2.792 | 2.755 | 3.0 | none |
| 3 | 2.087 | 3.156 | 3.0 | MISS: 3.156 > 3.0, next covering value 4.0 |
| 4 | 2.667 | 2.712 | 3.0 | none |
| 8 | 3.901 | 3.500 | 4.0 | none |

The binding case at bits 3, 4 and 8 is `device-factored-simd` at 512x512, constant-rows, float32, on both draws.
At bits 2 it is the CPU `factored-groups` member on heavy-tailed weights.

The shipped K is 4.0, and two independent routes land there: the bits = 8 calibration demand of 3.901, and the bits = 3 independent-draw miss.
The pre-registration requires the miss to be reported, and it is the one instance in this run where the independent draw exceeded what the calibration draw had chosen.

AMBIGUITY, both readings recorded rather than silently resolved.
The pre-registered clause selects the grid value from the calibration draw and moves to the next covering value on an independent-draw miss; the harness computes the maximum of both draws per width and covers that with one grid lookup.
The two readings give the same K at every width in this run (and the same 4.0 overall); the only observable difference is that the harness does not print the bits = 3 miss as a miss, so it is recorded here instead.

### Attribution: the join lowered the demanded K, and the residual moved it

Computed from the same records, the CPU-only six-member ensemble under the same leave-one-member-out rule demands:

| bits | Calibration draw | Independent draw |
|---|---|---|
| 2 | 2.792 | 2.755 |
| 3 | 3.400 | 4.007 |
| 4 | 2.151 | 2.821 |
| 8 | 3.192 | 3.361 |

Two facts follow.
First, Phase 0's K = 3 does not survive this grid even without any device member: the CPU int-domain member `factored-groups` demands up to 4.007 on constant rows at float32, which the grid would cover only at K = 6.
Phase 0 measured the rule at bits = 4 alone, where the demand is under 3, and ADR 0009 held K fixed by pre-registration and gated adequacy instead, so this quantity was never on the record before.
Second, joining the device class reduced the demanded K from 6 to 4: `device-factored-simd` raises the floor under exactly the cases where `factored-groups` sticks out, cutting the bits = 3 demand from 4.007 to 3.156.
Membership repaired most of the gap, in the direction ADR 0005's doctrine predicts, and K carried the residual that three device members could not close.

The pre-registered next upgrade, consistent with that doctrine: if a future width, shape or compiler pushes the joined demand above 4, the first repair is again membership (further device int-domain variants), and K moves only if membership cannot close it.

## The amended gate at K = 4, three classes: adequate at every width

| bits | Records | G1 false positives | G2 worst margin | G3 distinct per class | Equivalents | Verdict |
|---|---|---|---|---|---|---|
| 2 | 192 | 0 | 161.4x | 3 / 2 / 3 | 0 | ADEQUATE |
| 3 | 192 | 0 | 278.5x | 3 / 2 / 3 | 0 | ADEQUATE |
| 4 | 192 | 0 | 266.4x | 3 / 2 / 3 | 0 | ADEQUATE |
| 8 | 192 | 0 | 272.1x | 3 / 2 / 3 | 0 | ADEQUATE |

G1 is zero on both draws, held-out implementations included.
The G2 worst margins are numerically identical to ADR 0009's table despite K moving from 3 to 4 and the floor gaining a class, and the reason was verified from the records rather than waved at: every binding worst-margin cell is a float16 iid case where the base tolerance dominates `K * floor`, so neither change touches it.
Fault detection therefore did not pay for the wider tolerance, on this grid.

## One device member is bit-identical to a CPU member

The structural gate counts members on values, and it caught something the names hide: `device-dequant-loop` carries the same error as `dequant-pairwise` on all 768 records, alongside the known `lut-gather` duplicate.
Probed directly, the outputs are bit-identical, not merely equal in maximum error: 64 of 64 cases and 98,304 of 98,304 output elements agree exactly, across bits 2 and 8, shapes 512x512 and 1024x4096, both weight draws, all four input modes, both activation dtypes.

This is a real GPU result, not a probe artifact: on the same artefact and inputs, the loop member differs from `dequant-serial` (max 1.9e-6), from `device-dequant-simd` (8.1e-6) and from `device-factored-simd` (8.3e-6), and those two simd members differ from every CPU member.

Half of the mechanism is provable.
Scales and biases are fp16 storage values carrying at most 11 significand bits, codes are integers of at most 8 bits, so the product `s * q` holds at most 19 significand bits and is exact in float32 (verified on the artefact); the device's `fma(s, q, b)` and the CPU's multiply-then-add therefore round identically, and the dequantized weights are bit-equal by construction.
The accumulation half is an observation, not a proof: this Metal compiler's vectorization of the sequential fp32 loop rounds identically to the CPU library's small-batch matmul kernel on every probed case.
That identity is a property of the current toolchain and holds no promise; the member stays in the ensemble as written, and if a compiler update ever splits the pair, the floor gains a distinct member rather than losing one.

Consequence for G3, both readings.
Counted within each class, the device class holds 3 distinct members and the gate (at least 2 per class) passes as 3 / 2 / 3.
Counted for novelty across the whole ensemble, the device class contributes 2 members the floor did not already have, and the gate still passes.
The ensemble now carries nine names and seven distinct error signatures.

## The class-necessity diagnostic, at both K values

Never a gate; ratios above 1 mean the dropped class is load-bearing.
Worst member-to-tolerance ratio with a class removed, at the shipped K = 4:

| bits | Drop dequant-domain | Drop int-domain | Drop device-arithmetic |
|---|---|---|---|
| 2 | 0.367 | 1.967 (6 cases exceed) | 0.552 |
| 3 | 0.399 | 0.939 | 0.789 |
| 4 | 0.446 | 0.875 | 0.678 |
| 8 | 0.365 | 0.428 | 0.975 |

At the pre-move K = 3, dropping the device class instead gives 0.736 / 1.052 / 0.904 / 1.300, with 1 exceedance at bits 3 and 2 at bits 8, which is the trigger restated.

So the answer to the pre-registered question is conditional and both halves are on the record.
At the CPU-calibrated K = 3 the device class is load-bearing: without it, correct device kernels are flagged.
At the re-derived K = 4 no class other than int-domain at bits 2 is load-bearing in the diagnostic, but K = 4 itself exists because the device class's own most distinct member demanded it, and the join is what keeps the demand at 4 rather than 6.

## The fp16-dequant boundary under both floors

Cases outside tolerance at float32 activations, out of 96 per width (float16 activations are 0 / 96 in every reading, unchanged from ADR 0009):

| bits | K = 3, CPU floor (ADR 0009 shipped reading) | K = 4, CPU floor | K = 4, device-joined floor |
|---|---|---|---|
| 2 | 39 | 37 | 37 |
| 3 | 96 | 95 | 92 |
| 4 | 96 | 96 | 92 |
| 8 | 96 | 96 | 96 |

The first column reproduces ADR 0009's table exactly from this run's records, which is the intended regression anchor.

AMBIGUITY, both readings recorded: the pre-registered clause says "both floors, CPU-only and device-joined" without pinning K for the CPU-only column.
The harness reads it at the shipped K = 4; the ADR 0009-shaped reading at K = 3 is the first column.
All three are reported, and the decomposition is legible: the K move costs 2 cases at bits 2 and 1 at bits 3, and the membership join costs a further 3 at bits 3 and 4 at bits 4.

The certificate-spec statement updates accordingly: the intermediate-precision clause is numerically enforced at float32 activations on 92 of 96 or more cases for bits >= 3, partially at bits 2, and never at float16 activations, per ADR 0009's discharge unchanged in structure but weakened by the counts above.

## Consequences

The MLX-affine quant tolerance that covers device arithmetic is `max(base, 4 * floor)` over the three-class, nine-name floor; K = 3 over the CPU floor is now a measured false-positive ship against device candidates (3 of 768 cases, up to 1.30x, constant rows at float32).
The shipped oracle in `kernelverify/schemas/native_ops.py` (`K_QUANT = 3.0` over the CPU ensemble) carries exactly that exposure for every device kernel the battery verifies, including the kernel-pack lane's wide-tile results scored at K = 3.
The repair is the joined floor at K = 4, which puts device members (and therefore a Metal session or a precomputed device floor) inside the oracle path; that is a cross-lane change and is flagged to the fleet rather than smuggled into this commit.
ADR 0008's sentence about two calibrated K values becomes: K = 1.5 for unquantized ensembles, K = 4 for the MLX-affine quant contract with the device-joined floor.

The honest caveats.
Everything here is one machine, one Metal toolchain, one compiler optimization level; the loop-pairwise bit-identity in particular is empirical and version-bound.
The K move rests entirely on the constant-rows float32 regime at one shape, the same regime that has bound every tolerance decision on this surface; a battery without zero-variance structured modes would have measured none of this.
And the calibration-draw demand alone would have kept K at 3 at three of four widths; the independent draw is what exposed bits 3, which is the reason the pre-registration insisted on one.
