# Research note: the MLX kernel pack, measured before it is written

Date: 2026-08-14
Branch: mlx-kernel-pack
Status: research only, nothing implemented, nothing committed to a method

This note holds what was measured while Phase 0's K was still being derived.
It exists because three of the four kernels the design doc assigns to the pack target operations MLX already implements, and nobody had measured how much room those implementations leave.

Machine: MacBook Pro, M3 Pro (Mac15,7), 18 GPU cores, 36 GB unified memory, Metal 4.
Software: mlx 0.32.0 (cp314 wheel), Python 3.14.6, numpy 2.5.2, in this worktree's own `.venv`.

## 1. The MLX-affine quantization contract, pinned bit for bit

Any quantized kernel is correct only against the intended quantized weights, so the pack needs that definition pinned before it computes anything.
The contract is `affine_quantize` in `mlx/backend/metal/kernels/quantized.h` at v0.32.0.
A numpy reconstruction reproduces `mx.quantize` exactly - identical packed words, scales and biases - for every bit width (2, 3, 4, 5, 6, 8), every group size (32, 64, 128), float32 and float16 weights, and eight adversarial groups.
The reconstruction is in `bench/mlx_probes/probe_affine_contract.py`.

Note on ownership, resolved between research and commit: while this was written the phase0 lane was running the contract-anchored K on the nine corpus operators with no quantization in it, so the quant-contract K the design doc makes the gate for all S2 and S4 work appeared to be unowned.
It was not, and `83ea74c` landed it as a PASS, with `c8990d6` tightening K from 8 to 3 against a bit-exact quantizer replica.
The contract below is what that replica implements.

The algorithm, per group of `g` consecutive elements in a row:

```
n_bins = 2^b - 1
w_min  = min(group)                    # true minimum
w_max  = max(group, 0.0)               # seeded at 0, NOT at -inf
scale  = max((w_max - w_min) / n_bins, 1e-7)
side   = |w_min| > |w_max|
scale  = side ?  scale : -scale
edge   = side ?  w_min : w_max
q0     = round(edge / scale)           # round half away from zero
scale  = (q0 == 0) ? scale : edge / q0
bias   = (q0 == 0) ? 0     : edge
q_i    = min(round((w_i - bias) / scale), n_bins)     # top clamp only
```

Five things here are not guessable from the documented formula, and each is a place where two honest implementations diverge:

| Trap | What it does | Why it matters |
|---|---|---|
| `w_max` starts at 0, `w_min` does not | an all-negative group's upper end is 0, not its true maximum | changes scale and bias on every all-negative group, which is a large fraction of real weight groups |
| scale is re-derived as `edge / q0` | the largest-magnitude weight lands exactly on a level | scale is not `(max-min)/n_bins`; a naive implementation is wrong on every group |
| `bias = edge` | the anchor is the extreme value, not the minimum | the documented `beta = min` reading is wrong |
| everything is float32 internally | scale and bias are computed in fp32, then cast to the weight dtype on store | quantization uses the unrounded fp32 scale, dequantization uses the stored rounded one, so the codes are not the argmin under the stored scale |
| clamp is top-only, rounding is half-away-from-zero | Metal `round`, not numpy `rint` | numpy's default half-to-even disagrees on exact ties |

The packing rule is uniform across bit widths: a contiguous little-endian bit stream per row, lowest element in the lowest bits, reinterpreted as uint32.
The 3, 5 and 6-bit byte-level cases in the source are that same stream written out by hand.

CPU and GPU `mx.quantize` agree bit for bit.

## 2. MLX's own dequantize disagrees with itself across backends

`mx.dequantize` computes `scale * q + bias`, but not the same way on both backends:

| dtype | GPU | CPU | Gap |
|---|---|---|---|
| float32 | fused multiply-add, matches the fp64-exact result | separate multiply and add | 2.4e-7 |
| float16 | widens to fp32, then rounds once | half-precision arithmetic throughout | 2.0e-3 |

Both are correct implementations of the contract.
The fp16 gap is roughly two ulp at unit scale, from one library, on one machine, between its own two backends.
This is the legitimate-implementation spread the ensemble floor exists to cover, measured on the launch channel itself rather than assumed.

## 3. Device-side spread on a full quantized linear layer

Four implementations that all ship in MLX today, all correct, against an fp64 contract reference (exact `s*q + b` from the stored scales and biases, then an fp64 matmul):

| Case | Worst error vs contract | Relative | Spread across the four |
|---|---|---|---|
| 4096x4096, fp16 | 1.85e-1 | 7.6e-4 | 3.0x |
| 4096x4096, fp32 | 1.0e-4 | 2.4e-7 | 2.5x |
| 2048x2048, fp16 | 1.2e-1 | 7.6e-4 | 1.9x |

The fp16 absolute numbers look alarming and are not.
At an output magnitude of 242 the fp16 ulp is 0.25, and the largest pairwise disagreement between the four implementations is exactly 0.25.
The floor at fp16 is dominated by the final rounding of the output, which no implementation can avoid.

The consequence is specific: **an ensemble whose members return fp32 will under-estimate the fp16 floor by about three orders of magnitude.**
Members have to round to the working output dtype, or a CPU-derived K will not survive contact with the device.
The phase0 lane has confirmed its members already round in storage dtype end to end, and measured the same effect from the other side: on 186 of 4,060 battery cases the shipped ensemble is bit-exact against the fp64 reference while some legal implementation is not, so no multiple of that floor can cover those cases.
In relative terms the device floor sits at 7.6e-4, which is two orders of magnitude under the design doc's 2.48e-2 kill threshold for the quant experiment, so nothing here predicts a kill.

## 4. The performance bar, measured cold

Two measurement rules, both load-bearing, and both of which inflated the first run of this benchmark by up to 30x before they were applied:

- MLX is lazy, so building N graphs and evaluating one measures graph construction. Every figure below puts N copies of the op in one graph and evaluates once.
- A few-MB weight buffer reused across iterations reports system-cache bandwidth, not DRAM. Weights are rotated over a 512 MB working set. Any row above the ceiling means the rotation failed.

Streaming ceiling on this machine, measured: **128-130 GB/s** (fp16 sum over 256 MB), against 150 GB/s theoretical.

| Surface | MLX today | % of ceiling | Headroom |
|---|---|---|---|
| dense fp16 matvec, 4096x4096 | 282 us | 92% | none |
| 4-bit qmv, 4096x4096, M=1 | 75-82 us | 88-98% | none |
| 4-bit qmv, 14336x4096, M=1 | 281 us | 91% | none |
| 8-bit qmv, 4096x4096 | 139 us | 99% | none |
| 2-bit qmv, 4096x4096 | 46 us | 88% | none |
| MoE `gather_qmm`, 8 of 128 experts | 79 us | 70% | ~1.4x |
| MoE as 8 separate `quantized_matmul` | 235 us | 24% | already solved by `gather_qmm` |
| 4-bit qmm, batch 5-8 | 91-146 us | 50-81% | ~1.5-2x |
| fp16 SDPA decode, ctx 4096 | 57 us | 58% | ~1.7x |
| fp16 SDPA decode, ctx 16384 | 166 us | 79% | ~1.3x |

Bit-width sweep at 4096x4096, cold: 2-bit 46 us, 3-bit 62 us, 4-bit 81 us, 5-bit 96 us, 6-bit 114 us, 8-bit 139 us.
Every point is 88-99% of the streaming ceiling and the cost is monotone in bit width.

Two plan premises do not survive this on this chip:

- **S4, fused dequant-GEMV at 1.2-1.5x**, has no room inside MLX here. The operation is already at 88-98% of the achievable ceiling. The plan's number was read off llama.cpp on Max and Ultra parts, which have a different bandwidth-to-compute balance; this measurement does not refute it there, it says the win cannot be built or validated on this machine.
- **S2, sub-4-bit "up to 2x, currently inverted on Metal"**, is not inverted in MLX. Lower bit widths are strictly faster and equally saturated. The inversion is a llama.cpp property, not a Metal property.

The batch sweep has a sharp edge rather than a slope, and section 4b explains it.
That regime is the speculative-decode surface, and it is the largest measured soft spot inside MLX on this machine.

Caveats stated plainly: one chip, one process, best-of-five minima, and the machine was not verified idle.
The run-to-run noise model belongs to the baseline lane and is not duplicated here, so no claim below about 1.3x should be treated as real until that model exists.

The idle-machine caveat is not boilerplate here.
Re-running the same probe while three sibling lanes were working moved the measured streaming ceiling from 128 GB/s to 89 GB/s, a 30% swing, and moved ratios between shapes as well as absolute times.
Any perf number this project publishes has to be taken on a machine with the other lanes stopped, and that applies at least as much to the baseline lane's llama-bench roofline as to this note.

## 4b. What the batch cliff actually is

The first reading of the sweep was that MLX loses half its bandwidth between batch 4 and batch 8 on constant weight traffic.
That reading is wrong, and the correction matters because it changes what a kernel would have to do to win.

`QuantizedMatmul::eval_gpu` in `mlx/backend/metal/quantized.cpp` routes by batch size.
On this chip (`applegpu_g15s`, so architecture generation 15, size `s`) the thresholds are:

| M | Path | Weight passes |
|---|---|---|
| 1 | `qmv_quad` if K is 64 or 128, else `qmv` | 1 |
| 2 to 11 | `qmv_wide` | `ceil(M / 5)` |
| 12 and up | `qmm_splitk` (limit is 12 for K, N both <= 4096) | tiled |

`qmv_wide` sets `n_tiles = ceil(M / 5)` and gives each threadgroup a tile of at most five input vectors.
**Every tile re-reads the entire weight matrix.**
So batch 6 through 10 read the weights twice, and batch 11 reads them three times.
The weight traffic is not constant across the sweep; it is a step function.

Recomputing bandwidth against traffic actually generated, `ceil(M/5) * W`:

| M | Time | Tiles | GB/s counting real traffic |
|---|---|---|---|
| 1-4 | 77-82 us | 1 | 115-123 |
| 5 | 98 us | 1 | 96 |
| 6 | 121 us | 2 | 156 (two concurrent tiles share the read) |
| 7-8 | 150-154 us | 2 | 122-126 |
| 9-10 | 184-187 us | 2 | 101-103 |
| 11 | 224 us | 3 | 126 |
| 12-16 | 214-221 us | `qmm_splitk` | 43 apparent, tiled traffic not modelled |

Against real traffic MLX sits at 96-126 GB/s, which is 75-98% of the 128 GB/s ceiling.
MLX is not wasting bandwidth. It is generating extra bandwidth, by design, because a wider vector tile costs registers.

So the opportunity is well posed rather than free.
A kernel that holds eight input vectors in one threadgroup reads the weights once at batch 8, and the bound is one pass at the ceiling: 9.4 MB at 128 GB/s is 74 us against MLX's 154 us, a ceiling of 2.1x.
The wider tile will pay some register tax against that bound, so 2.1x is unreachable and the honest expectation is unknown until measured.

**How much of that is real is not yet established, and the reason is contention.**
The table above was measured while three sibling lanes were working on the same machine (load average 4.65, another lane's verifier harness resident at 2.1 GB), and every absolute number came out about 50% worse than the same probe on a quiet machine: the streaming ceiling read 89 GB/s instead of 128, and M=1 read 118 us instead of 82.
Worse, the ratio moved too - batch 8 against batch 4 was 1.92x on the quiet run and 1.29x on the contended one, because contention penalises the bandwidth-bound small-batch points more than the larger ones.

What survives that, and what does not:

- The mechanism is established, because it is read from MLX's source rather than inferred from timings: `n_tiles = ceil(M / 5)`, every tile re-reads the weights, and the dispatch switches to `qmm_splitk` at 12. The step at M=6 reproduced in both runs.
- The magnitude of the opportunity is provisional. It needs a re-run on a verified-idle machine before any number in this section is quoted anywhere.

The decisive experiment is one microbenchmark: a `metal_kernel` with a configurable vector-tile width, swept until occupancy collapses, to find where the register wall actually sits on this chip.
That experiment is not run here because implementation is on hold until Phase 0's K lands.

## 5. The custom-kernel door

`mx.fast.metal_kernel` is the launch channel and it takes everything a competitive kernel needs: threadgroup memory, barriers, `simd_sum`, and `simdgroup_matrix` multiply-accumulate all compile and produce correct results.
Mechanics worth recording:

- `grid` is total threads, not threadgroups; ragged grids are fine with an explicit bounds check.
- MLX generates the signature and injects only the built-in attributes the source actually names.
- Buffer address space is chosen by MLX: small inputs arrive as `const constant T*`, large ones as `const device T*`.
- The kernel is templated separately from the input dtypes, so a float32 buffer read into a `T = float16_t` converts, it does not reinterpret.
- `ensure_row_contiguous=False` silently reads the wrong elements from a strided input. The default True is correct and copies.
- A compile error raises `RuntimeError` carrying the MSL diagnostic on eval, with no silent fallback to a stock path.
- JIT cost: 20-90 ms to compile a source string never seen before, sub-millisecond on every later call. A rerun of the same source in a fresh process also started warm, so compilation is cached beyond the process; where, was not established.
- `compile_options={"math_mode": ...}` changes results. `safe` differs from `relaxed` and `fast` in the last ulp on transcendentals; `relaxed` and `fast` agreed on everything tested. The certificate has to pin it.

## 6. What this implies for the pack

The design doc's minimum launch pack is the fused dequant-GEMV plus a sub-4-bit decode kernel.
Both target the surface where MLX is already saturated on this machine, and neither can be shown to win here.
The measured headroom is somewhere else: small-batch quantized matmul (batch 5-8), MoE expert projection at decode, and attention decode at short to mid context.

This is a repointing, not a kill, and it is one machine.
The decision it feeds is D7, the one-week Metal feasibility spike, which should now be aimed at the vector-tile width in `qmv_wide` rather than at a dense dequant-GEMV that has 2% of ceiling left to take.

Target chosen with Vlad on 2026-08-14: the batch 5-8 quantized matmul.
It is the only surface where the incumbent's cost is structural rather than physical - MLX pays a second full pass over the weights to avoid a register wall, and the question of where that wall actually sits on this chip has not been asked.

One consequence for the launch story is worth naming early.
If the win is a wider vector tile, the change is small enough to land in MLX itself, which makes it a natural certificate-backed upstream PR - the acceptance experiment the plan wants, on the MLX channel rather than the llama.cpp one.
That is a strategy question for Vlad, not a kernel question.

## 7. What the pack needs from the other lanes

Recorded here so the requirements are stated before anything is built, not after.

From the Metal runner (`kernelverify/runners/`): execute a `mx.fast.metal_kernel` under the batched subprocess protocol and return numpy arrays; surface a `RuntimeError` carrying the MSL diagnostic as a failed case rather than a crashed run; and record the compile options in the case record, because `math_mode` changes results in the last ulp and the certificate has to pin it.

From Phase 0: the contract-anchored K, and the ensemble members it was calibrated on, with each member rounding to the working output dtype. The fp16 floor is dominated by output rounding, so an fp32-output ensemble under-states it by about three orders of magnitude. That lane has confirmed it already rounds in storage dtype end to end.

From the battery: a quantized-matmul reference with fault seams and catalogue entries (Tranche 1.5). The seams this surface needs, from the contract in section 1, are the ones where an honest implementation can go wrong: `w_max` seeded at -inf instead of 0, `scale = (max-min)/n_bins` without the `edge/q0` re-derivation, `bias = min` instead of `edge`, half-to-even rounding instead of half-away-from-zero, a missing top clamp, the group-boundary off-by-one, and the accumulate dtype.

From the baseline lane: the run-to-run noise model. Every number in this note is a best-of-five minimum on a machine that was not verified idle, so no claimed win below the noise floor is real yet.

## 8. A hand-picked ensemble does not bound the admissible class, and for the pack that cuts the other way

The phase0 lane measured, on the corpus operators, that seeded random summation permutations exceed the hand-picked sequential-order member by up to 8.3x, so a floor built from a few chosen implementations under-states the class it claims to bound.
Checked on the surface the pack ships into, a quantized matvec at fp32 against the MLX-affine contract, K = 4096, 64 rows (`probe_order_class.py`):

| Member, all admissible under the same freedoms | Error vs the fp64 contract |
|---|---|
| blocked at 128 | 2.27e-5 |
| numpy dot, pairwise | 2.86e-5 |
| blocked at 32 / 64 | 4.03e-5 / 4.30e-5 |
| magnitude ascending | 9.73e-5 |
| reversed order | 1.48e-4 |
| sequential accumulate | 2.28e-4 |
| magnitude descending | 2.46e-4 |
| **200 random permutations, max** | **4.79e-4** |

The hand-picked floor is 2.46e-4 and 19% of random admissible permutations exceed it, the worst by 1.94x.
The direction reproduces; the factor is smaller than the attention family's 8.3x, which is expected since a matvec has one reduction and attention has chained ones.
200 permutations is a sample, so 1.94x is itself a lower bound on the class supremum.

A real Metal kernel does not reduce in an adversarial sequential order; it reduces in a simdgroup tree over blocked tiles, which is the *accurate* end of the class - 4.3e-5 for a 64-wide block against 4.79e-4 for the worst permutation, a factor of 11.
That looked like detection margin being handed away on exactly the kernels this pack ships, and the obvious move was to propose that the contract exclude orderings no device can emit.

**That proposal was tested and dropped.** `probe_order_detection_cost.py` builds two floors over the same dequant-GEMV case, one over device-realisable members only (blocked at 32, 64 and 128, a pairwise tree, and the fused per-group form) and one over the full admissible class, then seeds six realistic kernel-side faults and judges each under `max(base_tol, 1.5 * floor)` for both.

| | fp32 weights, fp32 output | fp16 weights, fp16 output |
|---|---|---|
| device floor | 6.89e-5 | 6.19e-2 |
| class floor | 4.94e-4 (7.2x wider) | 6.19e-2 (identical) |
| faults caught, device vs class floor | 6/6 vs 6/6 | 6/6 vs 6/6 |

No detection is lost, at base tolerances of 0, 1e-4, 1e-3, 1e-2 and 5e-2, on either dtype.
At fp16 the two floors are the same number, because output rounding swamps every ordering difference, which is section 3 arriving from the other direction.

The reason the 7.2x costs nothing is that faults on this surface are not ordering-sized.
A quantization fault corrupts which weights are used, not how they are summed, so its signal is enormous next to any reduction-order effect:

| Fault | Error | Multiple of the class tolerance |
|---|---|---|
| unpack mask one bit too wide | 4.87e+2 | 656,000x |
| bias term dropped | 3.41e+2 | 460,000x |
| group index off by one | 2.42e+2 | 326,000x |
| bit-map shift by one code | 2.20e+2 | 296,000x |
| fp16 accumulation | 3.05e-1 | 412x |
| scale rounded to fp16 | 9.05e-2 | 122x |
| **one single 4-bit code wrong by one, in 4096** | **2.79e-2** | **38x** |

The smallest fault that can be constructed on this surface, a single code off by one across a 4096-long reduction, still sits 38x above the wider tolerance.
For a fault to land in the window between the two floors it would have to produce a relative error near 7e-7, which no weight-corrupting fault can reach.

So the phase0 lane's position holds here: the permissive ordering clause costs the pack nothing measurable, and the contract should not be narrowed for the pack's benefit.
The trigger that lane pre-registered - a fault absolved because the floor was stretched to a permutation no device emits - did not fire on this surface, and this note is the evidence for that rather than an argument against it.
The practical consequence is a constraint removed: the pack can verify against the verifier's own contract without declaring a narrower one of its own.

## Postscript, written at commit time

This note was researched against `0975c11` while implementation was held, and is committed onto `aff91fe`, seven commits later.
Three of those commits bear on it and none of them contradict it:

- `83ea74c` and `c8990d6`: the contract-anchored K for MLX-affine quantization exists and passed, and K was tightened from 8 to 3 using a bit-exact quantizer replica and class coverage. Section 1 is that contract; sections 3 and 8 are inputs to that K rather than a competing derivation of it.
- `e091765`: the D7 spike wrote a verified dequant-GEMV that reached 0.97-1.00x of `mx.quantized_matmul` in three iterations, and measured the roofline at 128.4 GB/s against this note's 128-130. Its verdict, that dense Q4 wins on Pro-tier are capped at about 8% and the headroom is MoE dispatch, speculative decode and sub-4-bit, is section 4 reached independently and by a different route.
- `1eb9812`: the battery moved out of `bench/` into `kernelverify/battery/`, so section 7's requirement list now points at a package that exists.

What section 4b says about the vector-tile cap in `qmv_wide` is not covered by that spike, which measured batch-1 dense GEMV.
The batch 5-11 step and the register wall behind it remain unmeasured, and remain the pack's chosen target.

## 9. The register wall, measured, and one lever that failed

Section 4b left the decisive experiment unrun: sweep the vector-tile width until occupancy collapses and find where the register wall sits.
It has now been run, and it produced both the shipped kernel and a falsified follow-up.

### The wall is two walls

Sweeping R (output rows per simdgroup) against M (input vectors held in one pass) at 2560x2560, 4-bit group 64:

| | R = 1 | R = 2 | R = 4 | R = 8 |
|---|---|---|---|---|
| M = 5 | 54.2 | 34.2 | 32.4 | 60.3 |
| M = 6 | 57.6 | 37.8 | 34.0 | 86.3 |
| M = 8 | 81.7 | 52.5 | 47.5 | 195.3 |
| M = 11 | 108.3 | 127.4 | 129.9 | 267.1 |

Two separate limits, not one:

- **Accumulators go as M\*R.** R = 8 spills at every M, by up to 5x.
- **Hoisting all M input vectors costs 8\*M float registers** and spills at M = 11 regardless of R. Restructuring so exactly one input vector is live at a time, with the R rows' unpacked codes hoisted instead, fixed M = 11 and improved every other point.

So MLX's cap of five vectors per tile is a real wall rather than an oversight.
The fix is not to raise the cap; it is to spend the register budget on rows instead of vectors, which buys the single weight pass back at R = 4 up to M = 10 and R = 2 above.
That is the shipped kernel, and it runs 1.22-1.40x MLX at M = 6 to 10 with parity below.

### The lever that failed: staging x in threadgroup memory

At M = 8 the shipped kernel sits at 81 GB/s against a 74 us single-pass bound, so 1.56x remains.
Each threadgroup's 8 simdgroups issue the same x loads and recompute the same per-word x sums, so removing that eightfold redundancy by staging an x tile in threadgroup memory looked like where the 1.56x was.

It is not there.
The staged variant is slower than the shipped kernel at every tile width and every M measured:

| M | 4 | 5 | 6 | 7 | 8 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| staged / shipped, TW = 64 | 0.94x | 0.84x | 0.81x | 0.82x | 0.80x | 0.86x | 0.80x |
| staged / shipped, TW = 32 | 0.87x | 0.79x | 0.78x | 0.77x | 0.79x | 0.81x | 0.80x |

TW = 128 and 256 measured the same way, 0.75-1.00x, and the largest tiles additionally hit the 32 KB threadgroup limit at M >= 8.
The redundant device reads were evidently already being served from cache, so staging pays two barriers and a cooperative load per tile to save traffic that was not costing anything.
The 1.56x is real but it is not in x traffic.

### The measurement mistake this caught, which matters more than the lever

The first comparison of these two kernels was not interleaved: each was timed in its own pass over the shapes.
That measurement said the staged version WON, by up to 1.45x.
Interleaving the arms within a round reversed the sign completely, and every number above comes from interleaved rounds.

The difference between those two readings is entirely power-state drift between passes, on a machine where MLX's own time for one fixed shape moved from 131.7 us to 93.1 us between runs minutes apart.
Any A/B on this machine that does not interleave its arms is measuring the machine's clock, not the kernels.
