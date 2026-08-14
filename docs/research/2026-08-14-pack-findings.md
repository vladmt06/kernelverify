# The verified MLX kernel pack: what was built, what it is worth, and what failed

Date: 2026-08-14
Branch: mlx-kernel-pack
Machine: MacBook Pro, M3 Pro (Mac15,7), 18 GPU cores, 36 GB unified memory
Software: mlx 0.32.0, Python 3.14.6, numpy 2.5.2

**Every performance number in this document is directional, pending binding measurement.**
That label is not a formality.
It means the number was measured on a machine sharing its GPU with a display stack, using the discipline described in section 7, and that section 7 also explains why two earlier versions of some of these numbers were wrong.
Correctness claims carry no such caveat: they are reproducible on any Metal machine and are pinned by tests.

Prior work in `2026-08-14-mlx-pack-feasibility.md`, which measured the surfaces before any kernel was written and is what pointed at the ones below.

## 1. What the pack contains

| Kernel | File | Contract it implements | Status |
|---|---|---|---|
| Wide-tile quantized matvec, bits 2, 3, 4 | `kernelverify/pack/wide_qmv.py` | MLX-affine quantization | verified, directional numbers |
| Fused top-2 MoE decode over quantized experts | `kernelverify/pack/moe_dispatch.py` | Qwen3-class routing + MLX-affine | verified, directional numbers |

Both are authored against `mx.fast.metal_kernel` and verified through the crash-isolated runner before any timing was taken.
That order is enforced in the harnesses: `bench/pack_wide_qmv.py` and `bench/pack_moe_dispatch.py` refuse to print a ratio if any case fails.

Verification coverage:

- Wide-tile matvec: 72 runner-isolated cases, being 2 shapes by 3 bit widths by 6 tile widths by 2 input scales, all passing against the Phase 0 contract oracle at K = 3.
- MoE decode: 6 runner-isolated cases, 3 shapes by 2 input modes, all passing against the shipped `NATIVE_OPS["moe_dispatch"]` reference and tolerance, with routing bit-exact against the contract in every case.
- 32 unit tests across the two kernels, inside a suite of 187.

## 2. The wide-tile matvec, and the defect it fixes

MLX routes batch 2 to 11 of a transposed quantized matmul to `qmv_wide`, which sets `n_tiles = ceil(M / 5)` and gives each threadgroup a tile of at most five input vectors.
Every tile re-reads the entire weight matrix.
Batch 6 to 10 therefore read the weights twice, and batch 11 reads them three times.

This is worth stating precisely, because the obvious reading of it is wrong.
Counted against the traffic MLX actually generates it runs at 75-98% of the achievable bandwidth on this machine.
MLX is not wasting bandwidth; it is generating extra bandwidth on purpose, to stay under a register wall.
The defect is algorithmic, and the fix is to buy the register budget back somewhere else.

Directional, 4-bit, 4096x4096, across clean runs:

| M | 4 | 5 | 6 | 7 | 8 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| ratio vs `mx.quantized_matmul` | 0.99-1.01x | 1.14-1.16x | 1.25-1.28x | 1.36-1.42x | 1.21-1.29x | 1.22-1.24x | 1.11-1.15x |
| MLX weight passes | 1 | 1 | 2 | 2 | 2 | 2 | 3 |

The shape of that curve is the claim: parity where MLX already reads the weights once, and a win from M = 6 where it reads them twice.
It reproduced across five separate interleaved runs.

## 3. Sub-4-bit is the same kernel

`ceil(M / 5)` does not depend on bit width, so the extra pass is paid identically at 2 and 3 bits and the same fix applies.
Bits 2, 3 and 4 are therefore one kernel templated on `BITS`, and the 4-bit path reproduces its previous numbers unchanged.

Directional, 4096x4096:

| M | 4 | 5 | 6 | 7 | 8 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| 2-bit | 1.08x | 1.12x | 1.21x | 1.40x | 1.26x | 1.23x | 1.11x |
| 4-bit | 0.99-1.01x | 1.14-1.16x | 1.25-1.28x | 1.36-1.42x | 1.21-1.29x | 1.22-1.24x | 1.11-1.15x |

**3-bit carries no performance number.**
It is correct and tested at every tile width, and it ships on that.
Its timings reproduced across two interleaved passes at 4096x4096 and did not reproduce at all at 2560x2560, giving ratios from 0.51x to 1.21x for one fixed shape with absolute times 5x higher than the same kernel measured minutes earlier.
In the most recent run the spread canary of section 7 rejected all nine 3-bit rows automatically, with reference-arm spreads of 2.2x to 9.7x, while passing the 4-bit rows in the same run.
The number waits for an idle machine.

## 4. The MoE decode kernel, and an honest split

Two kernels, because routing and dispatch have different shapes: routing needs every expert's logit for a token, dispatch needs the selected experts' weight rows.
Fusing them would recompute routing once per row block, which at `d_ffn = 768` and `R = 4` is 192 redundant copies per token.

Directional, Qwen3-30B-A3B class (d = 2048, 64 experts, ffn 768, top-2):

| tokens | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| full path vs MLX | 2.10x | 1.85x | 1.60x | 1.23x | 1.03x |
| dispatch only, same routing fed to both | 1.14x | 1.40x | 1.06x | 1.03x | 1.09x |

**These two rows must never be collapsed into one number.**
The full-path row includes routing; the dispatch-only row hands both arms identical precomputed routing.
The gap between them is the answer to "where does the win come from", and the answer is that most of it is fusing six MLX op launches into two.
That is a launch-overhead win.
It is real, and a decode loop pays those launches every layer of every token, but it is not the algorithmic win the wide-tile matvec found, and describing it as one would be dishonest.

The dispatch-only row is additionally noisy between runs (1.27/1.23/1.13/1.09/1.05 in one run, the tabulated values in the next), so the defensible statement is "the kernel itself is a small win" rather than any particular figure.

The baseline is not a strawman: `argpartition` is what mlx-lm's own MoE uses, and `argsort` measured identical at 41.7 against 41.3 us at one token.

## 5. Claims are scoped to a tile-width range, not to an operator

`should_dispatch(m)` returns false below M = 5, and the pack routes those to MLX.
At 4 bits that range is a wash at 1.00-1.02x, but at 2 bits it is a measured loss at 0.81-0.89x for M = 1 and 2.

The precedent matters more than the threshold.
A pack kernel is not uniformly better than the incumbent; it is better on a measured interval.
A certificate that says "this kernel is faster" without saying where is making a claim the measurements do not support, so certificates should carry the interval and the boundary function that enforces it.

## 6. Three findings worth reusing

### The register wall is two walls

Sweeping R (output rows per simdgroup) against M (input vectors held in one pass), 2560x2560, microseconds:

| | R = 1 | R = 2 | R = 4 | R = 8 |
|---|---|---|---|---|
| M = 5 | 54.2 | 34.2 | 32.4 | 60.3 |
| M = 6 | 57.6 | 37.8 | 34.0 | 86.3 |
| M = 8 | 81.7 | 52.5 | 47.5 | 195.3 |
| M = 11 | 108.3 | 127.4 | 129.9 | 267.1 |

Accumulators go as M\*R, and R = 8 spills at every M by up to 5x.
Separately, hoisting all M input vectors costs 8\*M float registers and spills at M = 11 whatever R is; keeping exactly one vector live at a time and hoisting the R rows' unpacked codes instead fixed M = 11 and improved every other point.
So MLX's five-vector cap is a real wall, and the way past it is to spend registers on output rows rather than input vectors.

### Stride blocks of codes, not words, below 4 bits

This is the one that will matter to anyone doing GGUF K-quants next.

Below 4 bits, codes stop aligning to 32-bit words.
The natural implementation strides whole words and adapts the bit arithmetic per width.
That implementation is correct and catastrophically slow: 0.52x at 2 bits and 0.26x at 3 bits against MLX at M = 8.

The cause is not the bit arithmetic.
Striding words makes the per-lane *activation* stride grow with the bit width, and the activation loads stop coalescing.
Striding a fixed 8-code block instead keeps the activation access pattern identical at every width and lets only the weight-side arithmetic vary; an 8-code block spans at most 24 bits, so it touches at most two words, read as a pair and shifted once.
Same arithmetic, 2.4x to 4.8x difference.

### Verification at fp16 has a hole, and it is in this pack's own kernel

The contract requires every intermediate at fp32 or wider.
Take the shipped wide-tile matvec and change one thing, its accumulator from float to half, and it is out of contract by construction, the quantized analogue of ADR 0004's fp16-score canary.

It passes.

| case | base_tol | K\*floor | dominates | correct kernel | faulted kernel | caught |
|---|---|---|---|---|---|---|
| typical 4096 | 2.24e-2 | 5.52e-3 | base | 1.84e-3 | 4.37e-3 | no |
| typical 2560 | 1.61e-2 | 3.90e-3 | base | 1.30e-3 | 2.61e-3 | no |
| small outputs | 3.21e-4 | 8.86e-5 | base | 2.95e-5 | 5.36e-5 | no |
| wide dynamic range | 4.15e-1 | 4.00e-2 | base | 1.33e-2 | 4.07e-2 | no |

`base_tol` exceeds `K * floor` by 3 to 10x in every case, so at fp16 the conditioning-aware half of the tolerance contributes nothing for this operator.
Raising K does not fix it: three of the four escape a floor-only rule too, because at fp16 output the half-accumulation error is genuinely comparable to legitimate output rounding.
This is ADR 0004's own pre-registered hole appearing at the typical operating point of the dtype the pack ships, not on a near-zero edge case.

Stated here rather than omitted because the pack's central claim is verification.
A launch artifact that advertises verified kernels while knowing of an unenforced contract clause on its shipping dtype would be making the same kind of claim this project exists to criticise.
The verifier lane has the measurement and the ruling is theirs; the current expectation is either a battery case that separates the fault or source-level accumulator attestation in the certificate.

## 7. How these numbers were measured, and two ways they were wrong first

Anyone reproducing this should apply all four rules, because each was learned by getting a number wrong.

1. **MLX is lazy.** Building N graphs and evaluating one measures graph construction. Every sample here puts N copies of the op in one graph and evaluates once. Getting this wrong inflated the first feasibility benchmark by up to 30x.
2. **Rotate the weights.** A few-MB buffer reused across iterations reports system-cache bandwidth, not DRAM. Weights rotate over a 512 MB working set. Getting this wrong produced rows reading above the machine's own streaming ceiling.
3. **Interleave the arms within a round.** Timing each kernel in its own pass over the shapes produced a table saying a variant won by up to 1.45x; interleaving reversed the sign at every point. The difference was entirely power-state drift between passes.
4. **Check the reference arm's own spread.** Interleaving is necessary and not sufficient: it makes both arms suffer a clock excursion together, it does not detect one. A simdgroup sweep produced a clean-looking table in which MLX's own time for one fixed shape went 150.6 to 345.8 to 355.4 us across rows while MLX did not change. Both benches now reject any row whose reference samples vary by more than 1.5x max-to-min, and exit non-zero.

Rule 4 justified itself on its first run by rejecting exactly the nine 3-bit rows that had been withheld by hand, and passing the 4-bit rows measured alongside them.

**The contention is the display stack, not the other work on the machine.**
When the worst spreads were recorded, load average was 1.79 and the machine was on AC power at 99%; the GPU was shared with Terminal at 36% CPU, VS Code at 25%, Spotify and WindowServer.
An idle gate that checks load average and power will pass in exactly those conditions.
Binding runs need interactive applications closed, which is a stronger condition than a quiet machine.

## 8. Recorded negatives

Kept because they cost time to establish and would otherwise be re-derived.

| Lever | Expected | Measured | Verdict |
|---|---|---|---|
| Stage x in threadgroup memory | remove 8x redundant reads, chase the 1.56x gap | 0.77-0.94x of shipped at TW 32, 64, 128, 256 and M 4 to 11 | falsified; the redundant reads were already cache-served |
| Magic-number unpack | replace int-to-float converts | 117.9 vs 117.2 us baseline, within 1% | no effect |
| Two blocks per iteration for ILP | hide latency | 116.7 vs 117.2 us baseline, within 1% | no effect |
| Stride whole words below 4 bits | natural generalization | 0.52x at 2 bits, 0.26x at 3 bits | falsified; see section 6 |

The magic-number variant has a trap worth recording: folding its constant into the bias term looks free and is not, because it would multiply activations by ~1.2e7 and lose six digits to cancellation.

At M = 8 the shipped kernel sits at 63% of the bandwidth bound and 39% of the machine's measured 5.89 TFLOP/s matmul peak.
Neither resource is saturated, which says the remaining 1.56x is a latency bound rather than an instruction-count or bandwidth one.
The ALU lever is therefore shelved as unmeasurable in current conditions rather than falsified; it needs the idle gate before it can be pushed further.
