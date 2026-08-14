# The mlx-lm end-to-end spike: fused kv_attention on Qwen3-4B decode

Lane C of the 2026-08-14 plan (design doc vlad-next-lanes-design-20260814.md, rulings D4, D9, D12).
The question, exactly as pre-registered: does the verified fused kv_attention kernel change end-to-end decode tokens/s on Qwen3-4B under mlx-lm at B=1, DH=128, KV bits 4 and 8, at T=512 and T=896, with generated tokens capped so the final step's logical cache length stays at or under the 1024-entry capacity.
T=1024 as a prompt length is banned (D9): that shape overruns the compile-time softmax buffer after one generated token.

Sections 1 to 4 were written and committed before any timing number existed; sections 5 and 6 hold the measurements.

## 1. Pre-stated rules

- Demotion rule (D12.7): if the stock fp16-cache arm beats the fused arm on decode tokens/s, the result is demoted regardless of the primary fused/stock-quantized ratio.
  Written here before any measurement ran.
- Pivot cost (D12.6): if the generation-level delta is swallowed by noise, the integration pivots to prefill/batch serving, and that is a NEW kernel plus a new verification cycle, not a re-aim of this one.
  The fused kernel is decode-shaped: one threadgroup per (batch row, query head), softmax over cached positions in threadgroup memory.
  Prefill and batch serving change the arithmetic intensity and the parallel decomposition, so nothing measured here transfers.
- Primary ratio: fused/quantized vs stock/quantized only (D4).
  The fp16-cache arm is a labeled third arm and is never merged into the primary ratio.
- Timing only in a quiet machine window (D12.2), `machine_state.idle_check` clean before and after; rows whose incumbent-arm spread exceeds the 10% tokens/s class limit are withheld.

## 2. The correctness bound that preceded measurement

t over capacity is now rejected loudly at both kernel doors (committed separately, 9fd2ad1, before any measurement).
The kernel body poisons the output with NaN and returns when the logical length exceeds TCAP, because a kernel cannot raise and the overrun it replaces is silent corruption of the compile-time-sized threadgroup softmax buffer.
The MLX door additionally raises ValueError before dispatch, where the bound is knowable host-side.

## 3. What the pre-decided integration point forced, and how it stayed verified

The interception is below the QuantizedKVCache update call (D12.1): the update appends the step's entry and the router then launches the fused kernel instead of the stock op composition.
Two properties of that point forced kernel-door changes (committed at eed05a3, re-verified 6/6 through the shipped oracle with the pre-change error values reproduced exactly):

- mlx-lm's cache buffers are padded to multiples of 256 rows, and the ruling forbids slicing them, because slicing charges the stock path's growing contiguity copies to our arm.
  The kernel therefore takes the LOGICAL cached length as a scalar, separate from the physical rows-per-head stride it reads from the buffer's own shape.
  A new unit test judges the kernel on a garbage-padded buffer against the oracle on the logical prefix.
- Qwen3-4B is grouped-query attention, 32 query heads over 8 KV heads.
  Query head h now reads cache head h / (H / HKV), the same contiguous-block mapping mlx-lm's own quantized SDPA uses, and equal head counts degenerate to the pre-integration kernel exactly.
  A new unit test verifies the 4:1 mapping against the tiled-cache oracle case, exact because quantization is per row.

Two integration facts recorded as assumptions, applied to every arm equally:

- The checkpoint (mlx-community/Qwen3-4B-4bit) is bfloat16; the frozen contract's dtypes are float16 and float32, so the model is cast to fp16 for all three arms.
  fp16 and bf16 are the same lane width on Metal, so tokens/s transfers.
- The two quantized arms differ semantically at the step's own entry: stock attends to it in quantized form, the contract reads it at full precision.
  Both are valid implementations of the model step; decode cost is shape-determined, so the timing comparison is unaffected.

The patch is runtime-only (bench/spike_mlx_e2e.py); no mlx-lm file is edited and nothing here ships as product code.
Wiring integrity, checked before any timing: every decode-shaped call in a shadow round went fused (324 of 324 over 9 steps of 36 layers, both bit widths), zero fallbacks outside whitelisted prefill, and the fused output sat within 0.011 of an fp64 dequantized-cache reference at output magnitude 36, under one fp16 ulp, while stock's own distance was 0.174.
That 15x is the fp16-score arithmetic the contract's canary exists to catch, in production form.

## 4. The minimum detectable effect, derived before the A/B

The microbenchmark prior (pack findings, section 9): the fused kernel beats the mlx-lm-shaped op composition 9-10x at T=64-512, 3.3x at T=1024, at B=1, H=8, DH=128, no GQA, unpadded buffers.
The E2E shape differs (32 query heads over 8 KV heads, padded buffers, stock paying its slice-contiguity copies), so the MDE uses per-op times re-measured at the true E2E shape, stock arm sliced exactly as mlx-lm slices.

Arithmetic.
Let t_attn_stock and t_attn_fused be the per-op attention times at the E2E shape, L = 36 layers, and t_tok the stock-quantized arm's measured per-token decode time.
The expected per-token saving is L x (t_attn_stock - t_attn_fused), and the expected fractional tokens/s gain is that saving divided by t_tok.
Equivalently: attention share s = L x t_attn_stock / t_tok, and gain = s x (1 - 1/r) at per-op ratio r.
The noise floor is the stock arm's round-to-round tokens/s spread.
Decision rule, pre-stated: if the expected gain is below the observed noise floor, the generation-level probe is reshaped (longer generations, more rounds) if arithmetic says the reshaped floor drops below the effect; otherwise the honest report is that the arithmetic already answers the question, and section 5's A/B is not run as a decider.

Measured inputs to the derivation (quiet window, idle-gated):

<!-- MDE numbers from bench/spike_mlx_e2e.py --mde go here -->

## 5. The generation-level A/B

<!-- results from bench/spike_mlx_e2e.py --ab go here -->

## 6. Verdicts

<!-- primary ratio verdict, demotion-rule outcome, integration recommendation -->
