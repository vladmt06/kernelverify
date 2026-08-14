# Code-block striding below 4 bits, written for the K-quant work

Date: 2026-08-14
Branch: mlx-kernel-pack
Status: the MLX-affine half is measured and shipping; the K-quant half is pre-registered prediction, measured nowhere yet.

This note promotes one finding out of `2026-08-14-pack-findings.md` section 6 into a standalone reference, because the audience that needs it is whoever writes the first GGUF K-quant kernel, and that person should not have to excavate it from a launch document.
The kernel it describes is `kernelverify/pack/wide_qmv.py`; the measurements behind every number in the first half are recorded there and in the pack findings.

## 1. The problem

A quantized weight row is a contiguous stream of B-bit codes packed little-endian into 32-bit words.
At 4 bits a word holds exactly 8 codes, so words and codes align and either unit can drive the inner loop.
Below 4 bits they stop aligning: a 32-bit word holds 16 codes at 2 bits and 10⅔ codes at 3 bits.
A kernel that supports several widths must choose what each SIMD lane strides over: whole words, or a fixed count of codes.

## 2. What was measured

Both loop shapes were implemented with identical dequantize-and-accumulate arithmetic, so the comparison isolates the striding decision.

| Lane strides over | 2 bits | 3 bits | 4 bits |
|---|---|---|---|
| whole 32-bit words | 0.52x vs MLX | 0.26x vs MLX | baseline |
| fixed 8-code blocks | ships | ships | ships |

Ratios at M = 8, 2560x2560, group 64; the word-striding row is the recorded negative in pack findings section 8.
Holding the arithmetic constant and changing only the striding moved throughput by 2.4x at 2 bits and 4.8x at 3 bits.
The word-striding kernel was correct; correctness never signalled that anything was wrong, which is why the number had to be measured.

## 3. The mechanism

The cost is not in the bit arithmetic, it is in the activation loads.

Each lane's dot-product needs the activation elements that correspond to the codes it just unpacked.
If a lane strides whole words, the number of codes per iteration is 32/B, so it varies with the width: 8 at 4 bits, 16 at 2 bits.
The per-lane activation footprint then also varies with the width, and the 32 lanes of a simdgroup spread their loads across a span that grows as B shrinks.
The wider that span, the more cache lines one simdgroup touches per iteration, and the activation loads stop coalescing.

If a lane strides a fixed 8-code block instead, the activation side never sees the bit width at all.
Every lane reads exactly 8 halves (two `half4` loads) per iteration at every width, from the same addresses it would read at 4 bits.
Only the weight side changes, and it stays cheap:

- an 8-code block spans 8B bits, at most 24 below 4 bits, so it touches at most two words;
- the kernel reads the pair, joins them into 64 bits, shifts once by the block's bit offset, then extracts 8 codes with one mask per code;
- a 64-element quantization group is a whole number of 8-code blocks, so scale and bias are constants within a block and are fetched once per block.

The rule, stated generally: **when a kernel spans multiple bit widths, pick the loop unit so the activation access pattern is width-invariant, and let only the weight-side arithmetic vary.**
The activation side is shared, dense fp16 traffic that the hardware coalesces; the weight side is a shift-and-mask pipeline that tolerates irregularity.
Put the width-dependence on the side that tolerates it.

## 4. What this predicts for GGUF K-quants

Everything below is pre-registered prediction, not measurement.
The repo's standing order applies: K-quants get their own quantization contract first (`kernelverify/schemas/quant_contract.py` states this), the shared gate in `kernelverify/pack/verify.py` judges every candidate, and only then do kernels get written and timed.

The K-quant layouts, from the `ggml-common.h` table vendored into `bench/gguf_info.py`:

| Format | super-block | bytes/block | bits/weight |
|---|---|---|---|
| q2_K | 256 | 84 | 2.625 |
| q3_K | 256 | 110 | 3.4375 |
| q4_K | 256 | 144 | 4.5 |
| q5_K | 256 | 176 | 5.5 |
| q6_K | 256 | 210 | 6.5625 |

The fractional bits per weight say the layout is not one contiguous code stream: a super-block carries embedded scale metadata, and several formats store codes split across a low-bit plane and a high-bit plane rather than as single B-bit fields.
The exact field layout is precisely what the K-quant contract must pin, byte-exact against llama.cpp's reference, before any kernel reads it.

Predictions, each with its falsifier:

- **The striding rule transfers.**
  Choose a fixed code count per lane per iteration, shared across all K-quant widths, and the activation pattern stays width-invariant exactly as here.
  Falsifier: implement one format both ways with identical arithmetic, as in section 2; if the ratio is near 1.0x, the rule did not transfer and the mechanism analysis for K-quants is different.
- **Split bit planes do not break the rule, they just double it.**
  A fixed code block whose low bits and high bits live in different planes touches a bounded number of words in each plane, so the read-pair-and-shift pattern applies per plane.
  The block unit should be chosen so both planes' reads stay word-pair-bounded.
  Falsifier: if per-plane reads dominate the profile, the binding constraint has moved to the weight side and the activation-invariance argument no longer decides the loop shape.
- **The block unit must divide the scale granularity.**
  Here a 64-wide group is a whole number of 8-code blocks, which is what lets scale and bias be per-block constants.
  K-quant sub-block scales sit at finer granularity inside the super-block; the code-block unit must divide that granularity or every block pays a scale boundary check.
- **Profitability gates do not carry over.**
  At 2 bits this kernel loses at M = 1 and 2 (0.81 to 0.89x) and the pack routes those widths to MLX below M = 5.
  Weight traffic shrinks with bits per weight, so whatever fixed overhead a kernel has looms larger as the format gets smaller, and each K-quant format needs its own measured dispatch threshold rather than an inherited one.

## 5. The method is the transferable part

The word-striding negative cost a full implementation to establish, and it was worth it: the natural generalization was correct, plausible, and 4x slow, and nothing short of the paired measurement would have said so.
The K-quant work should budget for the same experiment shape up front: two loop shapes, identical arithmetic, one measured ratio, before committing the kernel's structure.
