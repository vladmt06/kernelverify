# ADR 0019: bfloat16 joins the contract, carried as uint16, with the Metal subnormal band excluded

Date: 2026-08-19
Status: Accepted

## Context

Everything this repository has verified so far is float32 or float16.
Training is neither.
Sprint 1 gates kernels inside mlx-lm's QLoRA training step (the pre-registration of 2026-08-19), and the shipped artifacts that step runs on are bfloat16, so the contract had to be able to say the word before any training kernel could be gated at all.

Three facts were reproduced on this machine before anything was written.

- The pinned artifact stores bfloat16.
  `bench/.models/qwen3-4b-4bit-g64/model.safetensors` holds 651 BF16 tensors; a layer's `q_proj.scales` and `q_proj.biases` are BF16 while `q_proj.weight` is the U32 packed code stream.
- mlx-lm keeps that dtype for the whole model.
  `mlx_lm/utils.py` sets the model dtype from the scales' dtype and casts every floating weight to it, so bfloat16 in the file is bfloat16 in the training step.
- The contract could not express it.
  The eps table in `kernelverify/schemas/native_ops.py` held float16 and float32 only, and the runner's tensor dtypes had no 16-bit carrier of any kind.

## Decision 1: the eps entry, by the table's existing convention

The table's rule is one ulp at 1.0, which is `2**-(stored mantissa bits)`: 23 for float32, 10 for float16, 7 for bfloat16.
The entry is therefore `2**-7`, written exactly.

This is not a detail.
bfloat16's eps is 8 times float16's, so every tolerance the repository has written for float16 is 8 times too tight for bfloat16.
Borrowing float16's entry would have made the gate flag correct training kernels, which is the exact failure mode ADR 0012 caught in a different guise.

## Decision 2: uint16 is the carrier, because numpy cannot hold bfloat16

numpy has no bfloat16 dtype and refuses the conversion outright: `np.array(bf16_array)` raises `RuntimeError`.
The only route from an MLX bfloat16 array to the host is a `uint16` view of its raw bits.

So `uint16` joins `TENSOR_DTYPES`, on exactly the precedent `uint32` already set for packed quantized codes: the host copies bytes and counts them, and the shader declares `bfloat` for itself.

`kernelverify/schemas/bfloat16.py` owns the two conversions and is the only place that knows the layout.
`from_bits` is exact in the strong sense, because bfloat16 is float32's top 16 bits and every bfloat16 value is a float32 value.
`to_bits` rounds to nearest with ties to even, which is what the hardware store does, and is the direction that loses information.

Both are cross-checked against MLX's own bfloat16 in tests, per this repository's ground-truth honesty rule, over all 65536 bit patterns rather than a sample.

## Decision 3: the Metal subnormal band is outside the bfloat16 contract

The exhaustive cross-check found one disagreement, and it is not in the encoder.

Measured over all 65536 bfloat16 bit patterns, decoded to float32 and re-encoded through MLX:

| Stream | Patterns that disagree with the exact encoding | Where |
|---|---|---|
| MLX CPU | 0 of 65536 | - |
| Metal GPU | 254 of 65536 | every subnormal, `|x| < 2**-126`, flushed to zero |

The 254 are exactly the bfloat16 subnormal band, 127 patterns of each sign, and Metal writes all of them as zero.
MLX's own CPU stream preserves them.

The consequence for a gate is direct: a bfloat16 kernel judged against a reference computed on the host, or on MLX's CPU stream, disagrees with the device on any case that lands in that band, and the disagreement is the device, not the kernel.

The exclusion is declared rather than coerced.
`to_bits` is left as the exact round-to-nearest-even encoding, because that is the definition of bfloat16 and a helper that silently flushed would hide the fact from every future reader.
Instead the band is named as outside the contract's admissible class, the same shape of ruling ADR 0014 made for its own excluded cells, and the measured fact is pinned by a test so an MLX release that changes either stream turns the suite red rather than quietly widening what the gate claims.

Consequence for case generation: a bfloat16 case whose inputs or reference land in `|x| < 2**-126` is outside the contract and may not be judged.
No case in the training gate has any business there, since the smallest quantity a QLoRA step handles is many orders of magnitude larger, so the exclusion costs nothing that is being measured.

## What this does not decide

It does not calibrate a K for bfloat16.
The eps entry sets the base tolerance's scale; the conditioning-aware multiplier for bfloat16 ensembles is a separate calibration and is not claimed here.

It does not add bfloat16 to any existing operator's schema.
Nothing in today's battery runs at bfloat16, and nothing was retro-fitted; the entry exists for the training gate that sprint 1 builds.

It does not say anything about float16 or float32 behaviour, which is unchanged, or about subnormal handling in any dtype other than bfloat16.
