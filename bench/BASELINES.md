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
| mlx-lm 0.32, MLX-affine 4bit | 38.3 tok/s | (short-prompt, not comparable) | ~69% of achieved roofline |

Reading: on this Pro-tier chip, llama.cpp's dense Q4 decode is within ~8% of the physics.
The 2023-pinned community table (70-74% for Pro tier) is badly stale - the stale-baseline risk from the eng review materialised.
MLX trails llama.cpp by 23% end-to-end on the same model class, yet its quantized_matmul matches ours and is not the bottleneck (see spike), so MLX's gap lives outside this op class (attention path, graph dispatch).

## T5 spike: fused dequant-GEMV vs mx.quantized_matmul (bench/spike_dequant_gemv.py)

Three iterations of a ~70-line MSL kernel via mx.fast.metal_kernel, each verified 6/6 through the crash-isolated runner against the Phase 0 contract (K=3) before timing.

| Iteration | Change | Ratio vs MLX (best shape) |
|---|---|---|
| 1 | naive, 1 simdgroup per threadgroup | 0.41-0.99x |
| 2 | 8 simdgroups per threadgroup | 0.96-0.98x |
| 3 | vectorised half4 loads, fma | 0.97-1.00x |

Spike verdict for D7: writing a VERIFIED kernel that matches MLX's native op took under a day (feasibility: yes); beating it at batch-1 dense GEMV is not where the end-to-end gap is (both ops are launch-bound at decode shapes and near-identical in wall time).
Speed wins on this machine class must come from the surfaces the research named: MoE dispatch, speculative decode, sub-4-bit formats, and Max/Ultra tiers.
