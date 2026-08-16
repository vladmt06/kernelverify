# Faster and honest decode kernels on Apple silicon: what the literature says

Status: PRE-REGISTERED 2026-08-16, before any paper was read for this document.

Recorded assumptions of this review, made because the lane brief and the plan are the trace available to it:

- Sections 1 to 3 below are copied from Task R1 of `docs/superpowers/plans/2026-08-16-audit-amendments.md`, which is the ruled plan this lane traces to.
  Two names in section 1 - "pivot design E3" and "PolyQ" - come from that plan's own wording and from a design doc under `~/.gstack/projects/kernelverify/` that is not readable from this worktree, so they are reproduced as the plan wrote them and are not sourced independently here.
- Search is arxiv plus the code repositories papers link to.
  A paper is cited by arxiv id throughout; a claim about this repo is cited by ADR number or file path.

## 1. The question

One person, one MacBook, one open-weight model, batch-1 decode.
We want that person's model to run FASTER and in LESS MEMORY than Apple's stock path, and every kernel that gets it there to be produced by a generator whose every candidate the verifier gates.

What this repo has already measured, and this review must not re-derive (ruling D1 of the 2026-08-16 plan review):

- Dense 4-bit decode already runs at ~92% of the memory-bandwidth ceiling on this machine (ADR 0007); the remaining ~8% is the most any dense-4-bit GEMV kernel can ever win at batch 1. That is not where "faster" comes from.
- The fixed cost outside the kernel is 0.99 ms per token (ADR 0007, "the small-model gap is dispatch cost").
- Apple's own sub-4-bit (3-bit, 2-bit) path is inefficient by construction (pivot design E3, PolyQ's 2^b table wall), and 3-bit reads fewer bytes than 4-bit, so an honest 3-bit kernel near the ceiling would be BOTH faster than stock 4-bit AND smaller in memory. Our current 3-bit kernel loses to stock at batch 1 (ADR 0015: 0.937x-0.98x at M=1); the headroom is real and unclaimed.
- Apple's stock batch-1 kernel rounds in half precision internally (ADR 0014); an honest kernel accumulates in fp32 or better, and that is our contract's line.

Sub-question A1 (sub-4-bit kernels): what published kernel designs bring 2/3-bit weight-only GEMV close to the bandwidth ceiling at batch 1 on a unified-memory GPU without narrowing intermediate precision - packing layouts, dequant-on-the-fly schemes, lookup-table methods and their limits, register/occupancy strategies?
Sub-question A2 (around the kernel): for a single-user decode loop on a MacBook, where does the non-GEMV time go and which published techniques reclaim it honestly - dispatch and launch overhead (our 0.99 ms/token), speculative and parallel decoding, prefill, kernel fusion of the per-token tail?
Sub-question B (generator architecture): what published approaches to generating or searching GPU kernels (LLM-authored, autotuned, search-space) have a correctness gate, what did the gate catch, and how did they keep candidates honest?

## 2. Inclusion criteria (a paper is IN only if all hold)

- Addresses quantized matvec/GEMV or the decode step of an LLM, on a GPU.
- Reports speed against a named strong baseline (not against an unoptimised reference).
- States its accumulation precision, or gives enough kernel detail to infer it.
- For sub-question B: the generated kernels were checked for correctness by something more than "the benchmark ran".

## 3. Rubric (0-3 each; recorded per paper in the table)

- Evidence: 3 = open code + reproduced by others; 2 = open code; 1 = numbers only; 0 = claims.
- Applicability to Metal/unified memory: 3 = demonstrated on Apple silicon; 2 = memory-bound technique with no CUDA-only dependency; 1 = needs a hardware feature Metal lacks (tensor-core int8, warp shuffle semantics differ); 0 = CUDA-specific.
- Honesty by our contract: 3 = fp32 accumulate throughout; 2 = int accumulate then fp32 (our int-domain class); 1 = mixed with a narrower intermediate somewhere; 0 = fp16 accumulate.
- Bytes read per weight at M = 1, relative to MLX affine 3-bit (3 bits + one fp16 scale and one fp16 bias per 64-group = 3.5 bits/weight): 3 = under 3.25 bits/weight with the group parameters counted; 2 = 3.25-3.5; 1 = 3.5-4.0; 0 = at or above 4-bit dense. This is the mechanism at batch 1 (bandwidth-bound) and it is derivable from any paper's packing description; the paper's own reported gain and the hardware it was measured on go in a NOTE column, never in the score (ruling OV-7: CUDA-measured speedups do not transfer).

## 4. Candidates (filled by R1 step 4)

### 4.0 How the pre-registered rubric was applied where an axis does not fit

Three readings had to be made to score papers that the four axes do not cleanly describe.
They are recorded here rather than exercised silently, and they were fixed before the table below was filled.

- The four inclusion criteria of section 2 are written about weight-quantized matvec, so criteria 1 to 3 bind sub-questions A1 and A2, and criterion 4 is the bar for sub-question B.
  A generator or verifier paper is not excluded for failing to be a GEMV paper; that is what criterion 4 exists to say.
- The honesty axis scores the arithmetic a technique changes.
  A technique that does not change the arithmetic at all (speculative decoding, launch-overhead removal) has no accumulation precision to state, so it is scored `n/a-v` when the paper DEMONSTRATES output equivalence and is OUT under criterion 3 when it neither states a precision nor demonstrates equivalence.
  This keeps criterion 3 from being either vacuous or a free pass.
- The bytes-per-weight axis is about weight traffic at M = 1, so a paper that compresses the KV cache instead of the weights scores `n/a` on it and its compression is stated in the NOTE column instead.
  A paper scored `n/a` on that axis cannot be ranked by the section 5 product and is excluded from the A1 ranking by construction.

### 4.1 The table

| paper | year | sub-q | technique in one line | evidence | Metal | honest | bytes/weight | reported gain (hardware) | why it might beat stock at batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| QTIP, arXiv 2406.11235 | 2024 | A1 | trellis-coded quantization: a bitshift trellis whose codebook is computed per weight rather than stored, at 2, 3 and 4 bits | 2 | 1 | 3 | 3 (about 2.0 bits/weight at 2-bit; no per-group scale table) | matches QuIP#'s throughput at 32x the effective dimension; 188 tok/s at 2-bit on a 7B (RTX6000 Ada, 960 GB/s) | 2 bits/weight against MLX affine 3-bit's 3.5 is 1.75x less weight traffic, and batch-1 decode is bandwidth-bound (ADR 0007), so the traffic ratio IS the ceiling ratio |
| LUT-GEMM, arXiv 2206.09557 | 2024 | A1 | binary-coding quantization with a table of precomputed activation partial sums, so no weight is ever dequantized | 2 | 2 | 1 | 3 (q + 16/g bits/weight = 3.125 at q = 3, g = 128) | 2.1x over OPTQ at 3-bit OPT-175B; batch-1 token latency 51.6 ms row-wise, 46.5 ms at g = 128 (A100-80GB) | 3.125 bits/weight and an inner loop of table lookups and adds, which is work a bandwidth-bound kernel has spare |
| FLUTE, arXiv 2407.10960 | 2024 | A1 | offline restructuring of the packed weights plus a vectorized, duplicated lookup table, so unpacking a non-evenly-divisible width costs few bit operations | 2 | 1 | 1 | 3 (3 bits + one fp16 scale per 128 = 3.125 bits/weight) | 2-4x over existing GEMM kernels at batch < 32, 1.5-2x end to end (A100-80GB, A6000) | the unpack cost that makes 3-bit slower than 4-bit is moved offline, which is exactly the deficit ADR 0007 measured in q3_K |
| Any-Precision LLM, arXiv 2402.10517 | 2024 | A1 | bit-plane packed weights plus a per-output-channel centroid table; one memory image serves every width from 3 to 8 bits | 2 | 2 | 0 | 3 (about 3.0 bits/weight at 3-bit; centroid table is per output channel, not per group) | x3.99, x4.97 and x3.84 over cuBLAS fp16 at (1,4096)x(4096,4096) (RTX 4090, RTX 4070 Laptop, Jetson AGX Orin) | bit planes let the kernel read exactly the bits the requested width needs and nothing else, which is the cleanest possible traffic story at M = 1 |
| Lossless but Not Free, arXiv 2607.17283 | 2026 | A2 | an empirical anatomy of speculative decoding across five draft/target configurations on a consumer Apple-silicon laptop | 2 | 3 | n/a-v | n/a (weight reads per accepted token fall by the acceptance factor, not by packing) | 1.61x at K = 6 with acceptance falling from 69.7% at K = 1 to 37.8% at the optimum; 3 of 5 configurations DECELERATE (consumer Apple-silicon laptop) | it is the only batch-1 mechanism here that needs no better kernel: one weight pass serves several tokens |
| When Quantization Is Free, arXiv 2605.05699 | 2026 | A2 | one fused Metal kernel doing sign-randomized FFT, per-channel scale, per-group abs-max and int4 nibble packing for the KV cache | 2 | 3 | 3 | n/a (KV cache: 4 bits + one fp32 scale per d-vector = 4.25 bits/value at d = 128) | int4 KV decode at 37.0 ms/tok against 39.4 fp16 on SmolLM2-360M and 211.9 against 246.8 on 1.7B, 3x cache compression, dPPL 0.000 on Qwen short prompt (Apple M1, 8-core GPU) | the one measured case in this review of quantization paying for itself at decode on unified memory, and the paper's own stated mechanism is dispatch, not compute |
| Open-TQ-Metal, arXiv 2604.16957 | 2026 | A2 | fused compressed-domain attention: int4 KV dequantized in registers with online softmax, no intermediate matrices materialized | 2 | 3 | 3 | n/a (KV cache: 4 bits, group 32, 3.2x effective compression) | 48x over dequantize-then-attend at 128K context; decode latency 1.6 ms at 1K and 9.9 ms at 128K (Apple M1 Max) | it removes KV traffic at Sq = 1, which is our regime, on our backend |
| The Correctness Illusion, arXiv 2606.20128 | 2026 | B | measures the industry's allclose-on-one-shape oracle against a stricter one on a controlled 24-kernel corpus | 3 | 2 | n/a | n/a | the standard oracle certified 9 of 9 intentionally buggy variants as passing; the stricter oracle caught all 9 (five NVIDIA architectures) | it is this repo's premise; its corpus is vendored at `vendor/gpuemu-corpus` and ADR 0001 is our reproduction of it |
| Metal-Sci, arXiv 2605.09708 | 2026 | B | evolutionary (1+1) LLM kernel search on Apple silicon Metal with a hard correctness gate and a held-out size sweep | 2 | 3 | n/a | n/a | in-distribution self-speedups from 1.00x to 10.7x (Apple M1 Pro); the held-out sweep caught a sample covariance off by about 10 sigma and a 2.95x that became 0.23x | the closest published thing to the optimiser we want, on our hardware, and its ten tasks contain no decode and no GEMV |
| A Contract-Grade Verifier for LLM-Generated GPU Kernels, arXiv 2608.12700 | 2026 | B | twelve adversarial gates, several tolerance-free, one of them "accumulates in fp16 where the reference keeps an fp32 total" | 1 | 0 | n/a | n/a | 2,638 machine-generated kernels audited: 39.5% broken beyond any tolerance argument, 62.1% carrying at least one violation; the standard test accepted 1,487 kernels the verifier rejected against 14 the other way (NVIDIA Blackwell) | independent confirmation that the fp32-accumulate line is the line that catches things, at an audit scale we have not attempted |
| VOLTA, arXiv 2511.12638 | 2025 | B | an equivalence checker for GPU kernels, claimed sound and complete for a stated class, applied to hand-, LLM- and compiler-optimized kernels | 1 | 1 | n/a | n/a | verifies convolutions, matrix multiplications and attention variants | a proof beats a test wherever it applies, and where it does not apply is exactly our tolerance question |
| ProofWright, arXiv 2511.12294 | 2025 | B | an agentic framework that formally establishes safety properties and semantic equivalence for LLM-generated CUDA | 1 | 0 | n/a | n/a | safety properties verified for 74% of generated kernels; semantic equivalence only for a class of element-wise kernels | it marks where formal methods stop today: element-wise, which a reduction is not |

Six scores in that table were read from a source rather than from a paper's prose, and each is named here so a reader can check it.

- Any-Precision LLM scores honest 0 because its released kernel accumulates in half precision: `__half partial_sum[maxm * multi_row]`, sixteen weights accumulated per thread before any conversion, and the cross-lane reduction done with `__hadd` (`any_precision/modules/kernels/matmul.cuh`, SNU-ARC/any-precision-llm).
  That is the same arithmetic ADR 0014 ruled out of contract in MLX's own batch-1 kernel, reached independently by a different design.
- QTIP scores honest 3 for the same kind of reason in the other direction: its accumulators are `float4 reg_p[2]`, its inner product is `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, and its cross-warp reduction is a `float` (`qtip-kernels/src/inference.cu`, Cornell-RelaxML/qtip).
- QTIP scores Metal 1 for the same line: that mma is a tensor-core instruction, and ADR 0007 records that llama.cpp's device init on this machine reports the tensor API disabled for pre-M5 devices, so the kernel as published has no instruction to compile to here.
  The trellis itself is portable; the paper says its decode needs only "bitshifting by kV bits, which is supported on virtually all hardware".
- FLUTE scores honest 1 on its own sentence: "we implement in-register accumulation in FP32 and globally reduce partial sums in FP16" (arXiv 2407.10960).
  The narrow step is the cross-split reduction, and batch 1 is the most split-heavy regime a GEMV kernel has, so the one fp16 step sits exactly on our path.
- LUT-GEMM scores honest 1 by INFERENCE from kernel detail, not from a statement: the paper describes an fp16 input vector, an fp16 scaling matrix and a table of precomputed sums of mu activations held in shared memory, so the partial sums are formed and stored at fp16 before anything wider sees them (arXiv 2206.09557).
  That is the same shape as the eight-activation half-precision sub-sum ADR 0014 excluded in `affine_qmv`, and it is marked INFERRED because the paper never states an accumulator type.
- The Correctness Illusion scores evidence 3 because its corpus is open AND reproduced by someone else: this repository vendors it at the commit pinned in `vendor/PINNED.txt` and ADR 0001 records the escape-rate reproduction.

### 4.2 Excluded because

- BaseRT, arXiv 2607.00501 - criterion 3: the paper states no accumulation precision anywhere and gives no kernel detail to infer one ("each kernel integrates dequantisation directly into the inner loop" is the whole description), and it reports no perplexity, no output comparison and no correctness check of any kind.
  It is excluded despite being the strongest published batch-1 claim on our exact chip family: up to 1.56x decode over llama.cpp and 1.35x over MLX on M3 and M4 Pro, at batch 1, with code at `github.com/basecompute/baseRT`.
  A paper that claims to beat both incumbents on this machine and shows no evidence its output is right is the product thesis stated as a negative.
- Vec-LUT, arXiv 2512.06443 - criterion 1: its kernels are CPU kernels integrated into llama.cpp, not GPU kernels.
- Multi-Scale Dequant, arXiv 2605.13915 - criterion 1: its target is an Ascend NPU with decoupled compute units, and the dequantization bottleneck it removes is a property of that decoupling rather than of a GPU whose ALUs do the dequant.
- QServe (arXiv 2405.04532) and Atom (arXiv 2310.19102) - criterion 1 as measured: both quantize activations as well as weights and both report batched serving throughput, not the M = 1 decode GEMV; W4A8 and W4A4 are also a different contract question from weight-only.
- SBVR (arXiv 2509.18172), CodeGEMM (arXiv 2512.17970), AnyBCQ (arXiv 2510.10467), RaZeR (arXiv 2501.04052) - criterion 3: none states its accumulation precision in the material this review could read, and none links a kernel source this review could open.
- Marlin (PPoPP 2025) - criterion 3 as assessed: this review could not read a source that states its accumulation precision, and its 4-bit packing with group scales scores 0 on the bytes-per-weight axis regardless, so the ranking in section 5 could not move if the fetch succeeded.
- The launch-overhead family - ClusterFusion++ (arXiv 2604.23553), Mirage Persistent Kernel (arXiv 2512.22219), Hybrid JIT-CUDA Graph (arXiv 2604.23467), TaxBreak (arXiv 2603.12465) and "Memory-Bound but Not Bandwidth-Limited" (arXiv 2605.30571) - criterion 3 as assessed: none of the material this review could read states an accumulation precision, and each one's mechanism is a CUDA-runtime construct (CUDA Graphs, thread-block clusters, TMA descriptors, persistent megakernels) whose Metal counterpart, the indirect command buffer, none of them evaluates.
  They agree with ADR 0007's finding in words - "at batch 1 single-token decode, the binding constraint is the launch of the many small kernels that make up a transformer layer" (arXiv 2605.30571) - and none of them measures it on our backend.
  The PDF of arXiv 2605.30571 did not render for this review, so its numbers are not quoted anywhere above.

### 4.3 Seen and not assessed

Named so a later pass does not have to re-find them, and so nothing here is mistaken for a judgement:
KernelBenchX (arXiv 2605.04956), KForge (arXiv 2511.13274), STARK (arXiv 2510.16996), "Learning When to Optimize" (arXiv 2605.28213), Xe-Forge (arXiv 2605.26118), "Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization" (arXiv 2509.14279), T-MAN (arXiv 2511.11248), ELUTQ (arXiv 2510.19482), "Native LLM and MLLM Inference at Scale on Apple Silicon" (arXiv 2601.19139), "Above the Inner Loop: Exceeding Accelerate at LLM Prefill GEMM on the M1 AMX" (arXiv 2606.25426).

One of them is worth naming twice.
Rigel (arXiv 2606.12765) is titled "Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU", and this review read only its title.
If a tensor compute path exists and is reachable on M4, then ADR 0007's reading that the tensor API is disabled for pre-M5 devices bounds THIS machine (M3 Pro) and not the product's future hardware, which changes what a tensor-core-dependent technique like QTIP is worth.
That is an open question for a later pass, not a finding of this one.

## 5. Synthesis (filled by R2)

## 6. What the verifier would need (filled by R2)

## 7. The optimiser's shape (filled by R2)
