# Machine baselines - M3 Pro, 36GB, macOS 26.5.2

Measured 2026-08-14 on the development machine while a coding session was active.
Treat absolute numbers as first-pass; ratios between same-harness measurements are the stable signal.
Re-run on an idle machine before any published claim, per the eng-review amendment.

## Roofline (bench/machine_roofline.py)

| Metric | Value |
|---|---|
| Achieved bandwidth (copy) | 128.4 GB/s (85% of 150 GB/s spec) |
| Achieved bandwidth (reduce) | 127.2 GB/s |
| Achieved fp16 matmul (4096^3) | 5.89 TFLOPS |

## End-to-end decode, Qwen3-4B 4-bit, batch 1

| Stack | Decode | Prefill | Bandwidth utilisation |
|---|---|---|---|
| llama.cpp master (a94d563), Q4_K_M | 47.3 tok/s | 650 tok/s (pp512) | ~92% of achieved roofline |
| mlx 0.32.0 + mlx-lm 0.31.3, MLX-affine 4bit | 38.3 tok/s | (short-prompt, not comparable) | ~69% of achieved roofline |

Reading: on this Pro-tier chip, llama.cpp's dense Q4 decode is within ~8% of the physics.
The 2023-pinned community table (70-74% for Pro tier) is badly stale - the stale-baseline risk from the eng review materialised.
RETRACTED (2026-08-14): "MLX trails llama.cpp by 23%" was a harness artifact, not a stack property.
The 38.3 came from the mlx_lm.generate path, which pays sampling and detokenisation per token, while llama-bench times the decode loop only; measured on matched harnesses (mlx_lm.benchmark), MLX decode came out AHEAD on tok/s on the same machine.
The honest cross-stack comparison needs each stack's own byte model (MLX 4-bit group-64 streams ~9% fewer bytes per token than Q4_K_M), under which both stacks land close on bandwidth utilisation; binding numbers await the baseline harness run.

## T5 spike: fused dequant-GEMV vs mx.quantized_matmul (bench/spike_dequant_gemv.py)

Three iterations of a ~70-line MSL kernel via mx.fast.metal_kernel, each verified 6/6 through the crash-isolated runner against the Phase 0 contract (K=3) before timing.

| Iteration | Change | Ratio vs MLX (best shape) |
|---|---|---|
| 1 | naive, 1 simdgroup per threadgroup | 0.41-0.99x |
| 2 | 8 simdgroups per threadgroup | 0.96-0.98x |
| 3 | vectorised half4 loads, fma | 0.97-1.00x |

Spike verdict for D7: writing a VERIFIED kernel that matches MLX's native op took under a day (feasibility: yes); beating it at batch-1 dense GEMV is not where the end-to-end gap is (both ops are launch-bound at decode shapes and near-identical in wall time).
Speed wins on this machine class must come from the surfaces the research named: MoE dispatch, speculative decode, sub-4-bit formats, and Max/Ultra tiers.

## Corrections and additions (same day, from the baseline worktree's raw-Metal measurements)

Cross-session results from branch `baseline` (worktree kv-baseline, GPU-timestamp instrumentation, +/-1.6% repro), pending merge; recorded here so the numbers above are not quoted uncorrected.

- Denominator correction: decode is READ-dominated and the read ceiling is 135.4 GB/s (copy 130.1 idle; the 128.4 above was shared-machine copy, within 1.3% of idle).
  Utilisation computed against copy overstates by ~5%.
- Byte-model correction: file_size x tok/s is the wrong byte model (the embedding table is a gather, not streamed per token; tied embeddings change it again; the output head runs once per decode call, which also breaks prefill flop counts).
  The "~92% of achieved roofline" line above carries that error; per-model utilisation must use the GGUF tensor-table cost model (bench/gguf_info.py on the baseline branch).
  Directional conclusions survive: batch-1 dense Q4 decode is near its ceiling (independently measured 93.6% of roofline at n=1 with the correct model).
- ALU ceiling cross-validation: measured FMA peak 6.30 TFLOP/s (fp32 6.24; fp16 buys nothing on M3 - no tensor units pre-M5), so the 5.89 TFLOPS matmul above is 93% of the true ceiling.
- NEW measured surface - the small-batch hole: llama.cpp's Metal path drops to 19-30% of roofline at batch 8-16 (vs 93.6% at batch 1, 77% at 512), reproduced at kernel level and end to end.
  A decode-only, batch-1 view hides this entirely; local agent and server workloads live exactly there.
- q3_K mat-vec runs ~1.55x off its peers at long reductions only (shape-dependent; invisible at 0.5B shapes) - real-hardware confirmation of the sub-4-bit inverted-performance class.
- Fixed dispatch intercept: 0.99 ms/token on this machine (21.6% of a 0.5B's token time, 2.7% of a 7B's); small-model utilisation numbers are depressed by it and say nothing about kernels.
- Timing-floor caveat (from the metal-runner worktree): GPU timings at ~200us shift together up to 4x with power state; only ms-scale dispatches repeat to ~0.1%.
  The spike's per-op times above (150-590us) sit inside that regime: treat the ratios as UNMEASURED, not merely provisional, until re-measured with interleaved A/B sampling at ms-scale batching.
