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
- Apple's own sub-4-bit (3-bit, 2-bit) path is inefficient by construction, and 3-bit reads fewer bytes than 4-bit, so an honest 3-bit kernel near the ceiling would be BOTH faster than stock 4-bit AND smaller in memory.
  The "pivot design E3" and "PolyQ 2^b table wall" references travel from the plan's own skeleton wording; the design document naming them is outside this worktree and this review could not open it, so they are unsourced here. Our current 3-bit kernel loses to stock at batch 1: the committed pricing recording `bench/results/qmv-boundary-pricing-2026-08-15.json` reads 0.937x at M = 1, and ADR 0015 buckets M = 1 to 3 as a loss without restating the ratio.
The headroom is real and unclaimed.
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
- Bytes read per weight at M = 1, relative to MLX affine 3-bit, which is 3 bits plus one fp16 scale and one fp16 bias per 64-group and therefore 3.5 bits/weight.
  Bands, with the group parameters always counted: 3 = under 3.25 bits/weight; 2 = 3.25-3.5; 1 = 3.5-4.0; 0 = at or above 4-bit dense. This is the mechanism at batch 1 (bandwidth-bound) and it is derivable from any paper's packing description; the paper's own reported gain and the hardware it was measured on go in a NOTE column, never in the score (ruling OV-7: CUDA-measured speedups do not transfer).

## 4. Candidates (filled by R1 step 4)

### 4.0 POST-HOC amendment to section 2, made after the papers were in hand

Status: this section was written in the same commit that filled the table below (`db65724`), NOT in the pre-registered skeleton (`71c6748`).
It relaxes a pre-registered rule after seeing the candidates, which is the exact move pre-registration exists to prevent, so it is labelled rather than absorbed.
The merge review of 2026-08-16 caught it.

Why the relaxation was needed rather than merely convenient: section 2's criteria are written about weight-quantized matvec, and read literally they exclude EVERY generator paper for not being a GEMV paper, which makes sub-question B unanswerable by construction.
That is a flaw in the pre-registration, discovered mid-review.
The honest handling is to say so here and to let the reader discount what depends on it, not to rewrite section 2.

WHAT DEPENDS ON IT: three papers are IN only under this amendment and fail section 2 as originally written.
Metal-Sci (arXiv 2605.09708) fails criterion 1 - this document's own row says its ten tasks contain no decode and no GEMV.
VOLTA (arXiv 2511.12638) fails criterion 2 - it reports no speed against any named baseline.
Open-TQ-Metal (arXiv 2604.16957) fails criterion 2 as written, its baseline being an unoptimised dequantize-then-attend reference.
None of the three is in the section 5 ranking or the top three, so no spike and no go/no-go decision rests on them; they inform sub-question B and the Metal context only.

Three readings had to be made to score papers that the four axes do not cleanly describe.

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
- Marlin (PPoPP 2025, arXiv 2408.11743) - criterion 3 as assessed: this review could not read a source that states its accumulation precision, and its 4-bit packing with group scales scores 0 on the bytes-per-weight axis regardless, so the ranking in section 5 could not move if the fetch succeeded.
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

## 5. Synthesis

### 5.1 The ranking, and what it says about the literature

The pre-registered order is the product of the honesty score and the bytes-per-weight score, with evidence as the tie-break.
Only the A1 rows can be ranked: the axis is weight traffic at M = 1, and section 4.0 fixed that a paper scored `n/a` on it is out of this ranking by construction.

| rank | paper | honest x bytes | evidence | Metal | kept |
|---|---|---|---|---|---|
| 1 | QTIP, arXiv 2406.11235 | 3 x 3 = 9 | 2 | 1 | yes |
| 2 | LUT-GEMM, arXiv 2206.09557 | 1 x 3 = 3 | 2 | 2 | yes |
| 3 | FLUTE, arXiv 2407.10960 | 1 x 3 = 3 | 2 | 1 | yes |
| 4 | Any-Precision LLM, arXiv 2402.10517 | 0 x 3 = 0 | 2 | 2 | no |

Ranks 2 and 3 tie at 3 and tie again on evidence at 2, which the pre-registered tie-break does not resolve.
The tie is broken by the Metal axis (LUT-GEMM 2 against FLUTE 1) and the extension is recorded here rather than applied silently; it changes the order of two techniques that are both kept, so nothing rides on it.

The ranking says one thing plainly, and it is the finding of this review.
Exactly one published sub-4-bit GEMV design in this table both reads fewer bytes per weight than MLX affine 3-bit and keeps an fp32 total, and its published kernel gets its speed from a tensor-core instruction this machine does not have (ADR 0007).
The other two designs reach the same traffic and pay for it with a narrower intermediate: FLUTE by reducing partial sums in fp16 across splits, LUT-GEMM by forming activation partial sums in an fp16 table.
The fourth accumulates in half precision outright, which its own released source shows.
So the product opening is not "port a paper".
It is that the honest version of these packings has not been published for any backend, and the backend where it matters most for a person on a laptop has no entrant at all.

### 5.2 Rank 1: QTIP (arXiv 2406.11235)

**Mechanism.**
Weights are coded by a bitshift trellis: consecutive weights share bits of one bit stream, and the decoder walks the stream shifting kV bits at a time, so no per-group scale table has to be read at all.
At 2 bits that is about 2.0 bits per weight against MLX affine 3-bit's 3.5 (3 bits plus one fp16 scale and one fp16 bias per 64-group).
The released kernels are for the HYB code, whose decode is a small table gather rather than an arithmetic expression, and the accumulation is fp32 throughout (`float4 reg_p`, an `f32.f16.f16.f32` mma, a `float` reduction).

**Why it should still win on Metal at batch 1.**
Batch-1 decode on this machine is bandwidth-bound and dense 4-bit already sits at about 92% of the bandwidth ceiling (ADR 0007), so the only large lever left is reading fewer bytes.
1.75x less weight traffic is a 1.75x lever on the part of the token that is not the 0.99 ms fixed cost (ADR 0007).
The decode work the trellis adds is arithmetic, and arithmetic is what a bandwidth-bound kernel has spare: ADR 0007 measured 6.24 TFLOP/s of fp32 against 135.4 GB/s, a ridge point of 46.7 flop/byte, and a decode of a few instructions per weight is nowhere near it.

**The risk that it does not.**
Three, in order of size.
The published kernel's speed comes from `mma.sync`, a tensor-core instruction, and ADR 0007 records the tensor API disabled on this machine, so the Metal version is a different kernel and its performance is unmeasured rather than ported.
QTIP also applies incoherence processing, which means a Hadamard rotation of the activations at run time and its inverse on the output; that is arithmetic our operator does not contain today and it costs time the paper's own numbers include but our roofline does not model.
And 2-bit quality is a separate claim from 2-bit speed: nothing in this review measures whether a 2-bit Qwen3-4B is a model anyone wants, and the pricing protocol will happily certify a fast kernel for a bad model.

**The cheapest spike on this machine.**
Strip it to the part that carries the traffic win and drop the rest.
Write one Metal kernel that decodes a bitshift-trellis 2-bit stream with a table gather and accumulates in fp32, at ONE shape, `down_proj` 2560x9728, and at M = 1 only.
That shape because it has the largest reduction dimension of the six Qwen3-4B dispatch shapes (ADR 0015) and ADR 0007 measured that the sub-4-bit kernel deficit needs a large reduction dimension to appear at all.
Verify first with `bench/pack_wide_qmv.py`'s gate at that shape, then time under `bench/price_qmv_boundary.py`'s protocol (interleaved arms, canary spread capped at 1.5, `ratio_lo > 1` is the only WIN), against stock MLX at the width the product would ship.
Skip incoherence processing in the spike: without it the arithmetic stays inside a class the contract already has members for (section 6), and if the traffic win does not appear without the rotation it will not appear with it.

### 5.3 Rank 2: LUT-GEMM (arXiv 2206.09557)

**Mechanism.**
Binary-coding quantization writes each weight as a sum of q signed binary matrices times per-group scales, and the kernel precomputes, once per input vector, a table of every possible sum of a mu-element sub-vector of activations.
The matvec then reads weight bits and gathers table entries: no weight is ever dequantized and no multiply happens in the inner loop.
At q = 3 with g = 128 that is 3.125 bits per weight, against MLX's 3.5.

**Why it should still win on Metal at batch 1.**
The table is small (about 1 KB per eight hidden dimensions by the paper's own accounting) and lives in shared memory, which is Metal threadgroup memory with no CUDA-only dependency named anywhere in the paper.
The inner loop becomes lookups and adds, which is the cheapest thing a bandwidth-bound kernel can be doing while it waits, and the paper's batch-1 numbers are the ones it leads with (51.6 ms to 46.5 ms per token on OPT-175B).
It is also the only top-three technique whose mechanism removes multiplication from the inner loop entirely, which matters on a machine where fp16 buys no arithmetic over fp32 (ADR 0007: 6.30 against 6.24 TFLOP/s).

**The risk that it does not.**
The mechanism's cost is a table build per input vector, and at batch 1 there is exactly one input vector per token to amortize it over, so the build is a fixed per-token cost added to a token that already carries 0.99 ms of fixed cost (ADR 0007).
The table also has to be rebuilt for every layer, which multiplies that cost by the layer count.
And the honest version is not the published version: the fp16 table has to become an fp32 table, which doubles the shared-memory footprint and can cost occupancy, so the spike measures a kernel the paper did not.

**The cheapest spike on this machine.**
Same shape, same protocol, same M = 1: one Metal kernel that builds the sub-vector sum table in fp32 in threadgroup memory and accumulates the gathered sums in fp32, at `down_proj` 2560x9728.
Time the table build separately from the matvec inside the same interleaved pass, because the two have different scaling and a single number cannot say which one lost.
The pre-registered reading is the protocol's own: `ratio_lo > 1` against stock MLX 3-bit or it is not a win.

### 5.4 Rank 3: FLUTE (arXiv 2407.10960)

**Mechanism.**
The packed weight matrix is restructured offline so the bit-unpacking a non-evenly-divisible width needs becomes a handful of aligned vector loads, and the dequantization table is duplicated and vectorized so table reads do not serialize on shared-memory bandwidth.
At 3 bits with group 128 it reads 3.125 bits per weight.

**Why it should still win on Metal at batch 1.**
The deficit it attacks is exactly the one ADR 0007 measured in llama.cpp's q3_K: a format that moves 24% fewer bytes than q4_K and is still 17% slower in tokens per second at the 7B's shapes, which no traffic model permits and which therefore has to be unpack cost.
FLUTE's answer is to move that cost to quantization time, and offline restructuring is backend-independent by construction: it is a permutation of bytes in a file, not an instruction.

**The risk that it does not.**
As published this technique is out of contract, not merely imprecise: "we implement in-register accumulation in FP32 and globally reduce partial sums in FP16", and the cross-split reduction is where batch-1 GEMV does most of its reducing.
That is clause C1's line and the same finding ADR 0014 recorded against MLX's own batch-1 kernel, so a faithful port would be a kernel our verifier is built to flag.
The rest of the kernel is Ampere-shaped (the paper says it is "mostly optimized for Ampere-generation GPUs"), so what transfers is the offline restructuring idea and not the kernel.

**The cheapest spike on this machine.**
Do not port the kernel; test the claim.
Take our existing 3-bit path, produce a restructured weight file offline (bit-slices aligned so an MSL kernel reads whole words per thread), and measure only whether the same fp32-accumulating kernel gets faster at M = 1 at `down_proj` 2560x9728 when the bytes are laid out differently.
That isolates the one portable idea from everything CUDA-shaped around it, and it is the cheapest of the three spikes because it changes no arithmetic at all, which also means its gate result is a formality rather than a question.

### 5.5 What sub-question A2 adds, which is more than the kernels do

Two A2 findings change what the A1 spikes are worth, and one of them may be worth more than all three.

**Speculative decoding is the only mechanism here that moves batch-1 without a better kernel, and it is measured on our hardware.**
arXiv 2607.17283 measures five draft/target configurations on a consumer Apple-silicon laptop: the best reaches 1.61x wall-clock at K = 6, and three of five configurations DECELERATE.
One of the losing reasons is the finding to read twice: "the quantized Metal backend executes 'parallel' verification serially".
Verification of K drafted tokens is a batch-(K+1) matvec, which is the tile-width regime ADR 0015 priced, and where our own wide-tile kernel is a measured WIN at M = 5 to 9 (1.02x to 1.38x across all six shapes).
So the kernel this repo already has, which ADR 0015 and the plan both record as useless at batch 1, is a kernel for the exact operation that a speculative decoder does once per accepted run.
That is not a claim that it works; it is a claim that the two measurements meet, and that the meeting point is testable with things this repo already owns.

**Quantization paying for itself at decode has been measured once, on Apple silicon, and the mechanism was dispatch.**
arXiv 2605.05699 reports an int4 KV cache running FASTER than fp16 across 256 to 4096-token prefixes on Apple M1 (37.0 against 39.4 ms/token on SmolLM2-360M, 211.9 against 246.8 on a 1.7B), with the whole transform in fp32 and quality preserved.
Its own explanation is ours: "the cost is dispatch, not compute", and the fused single-dispatch kernel is what closed a 12-17% eager-mode penalty.
That is an independent measurement of ADR 0007's 0.99 ms/token on a different stack, and it says the fusion lever is real on this backend, which none of the CUDA launch-overhead papers can say.
arXiv 2604.16957 is the same lever at Sq = 1 with fp32 accumulation and open Metal shaders, so the technique has two independent Apple-silicon entries and no CUDA dependency at all.

## 6. What the verifier would need

Every spike above produces a kernel the verifier has to judge, and the judgement is only meaningful if the shipped tolerance's floor contains a member that rounds the way that kernel rounds.
That is not a formality here: ADR 0016 is the record of what happens when it fails.
`factored-groups` formed its per-group sums with numpy's pairwise reduction, which is exact on a constant row, so it was the most accurate member exactly where real kernels are least accurate, and the shipped tolerance flagged a correct device kernel on 10 of 1,536 serving records.
The standing order from ADR 0012 is membership before K, and it applies to each technique below before its gate result may be read.

The classes, as the contract has them today: the six CPU members of `kernelverify/schemas/quant_contract.py::ENSEMBLE` split into a dequant-domain group (`dequant-pairwise`, `dequant-serial`, `lut-gather`, `dequant-reversed`) and an int-domain group (`factored-groups`, `factored-serial`, both of the form `s * sum(x*q) + b * sum(x)`), with a third device-arithmetic class in `kernelverify/schemas/quant_device.py` (`DEVICE_CLASS = "device-arithmetic"`).

| technique | arithmetic class | why | eligible outcome |
|---|---|---|---|
| QTIP (arXiv 2406.11235) | NEW | the multiply-accumulate is dequant-domain, but incoherence processing puts a Hadamard rotation of the activations and its inverse inside the operator, and no member of any class rotates | NEW CLASS: membership block first |
| QTIP without incoherence processing | dequant-domain | a trellis decode by table gather is an exact gather of representable values, which is `lut-gather`'s shape with a different address computation; addressing is not an arithmetic class | GO / NO-GO |
| LUT-GEMM (arXiv 2206.09557) | int-domain | `sum_i alpha_i * (sum_j b_ij x_j)` is the contract's `s * sum(x*q) + b * sum(x)` with q in {-1, +1}; the table changes the ORDER of the inner sum, not its domain | GO / NO-GO |
| FLUTE (arXiv 2407.10960), as published | dequant-domain, and out of contract | table-gather dequant then fp32 mma is `lut-gather`'s shape, but the fp16 cross-split reduction is an intermediate narrower than binary32, which is clause C1 - the same ruling ADR 0014 made against MLX's own batch-1 kernel | NO-GO by construction; only the fp32-reduction variant is eligible for GO |

**What the contract would need for the NEW case.**
A QTIP kernel with incoherence processing computes `R2^T * f(W', R1 * x)` where the rotations are fast Hadamard transforms, so the operator's reference is no longer "quantized matvec" and the admissible class is no longer described by the six members.
Before its gate result could be read, the ensemble would need at least one member that performs the same rotation with its summation order spelled out, and by AGENTS.md's own lesson it has to round the way a real Hadamard kernel rounds: a butterfly of log2(d) stages accumulated in the stage order the kernel runs, not numpy's vectorized transform, which is the precise mistake ADR 0016 had to repair in `factored-groups`.
The class needs two members, not one, for the reason `quant_contract.py` already states beside `factored-serial`: leave-one-out calibration over a one-member class measures class absence rather than within-class spread.
Then K is re-derived over the enlarged membership, in that order, never the reverse (ADR 0012, ADR 0016).

**One risk inside a class that already exists.**
LUT-GEMM is int-domain and therefore GO/NO-GO eligible, and the class it lands in is the one ADR 0016 flagged as resting on a single outlying member: the six-CPU leave-one-out spread is 7.561 for `factored-groups` against 2.166 for `factored-serial`.
A LUT-GEMM-shaped kernel sums each mu-element sub-vector as a shallow tree, which is neither the chain `factored-groups` now runs nor the order `factored-serial` varies, so the class contains its domain but not necessarily its rounding.
That does not change the eligible outcome, which section 6 fixes before any spike runs; it is recorded as the reason a NO-GO on this technique would need reading twice before it is believed.

### 6.1 The go/no-go, pre-registered before any spike runs

Three outcomes, and which one a technique is eligible for is fixed by the table above, not by what its spike measures.

- **(i) GO.** The technique's M = 1 spike measures WIN under the pricing protocol (`ratio_lo > 1`, the whole ratio interval clear of 1.0 from below on a round whose reference arm held steady) AND passes `bench/pack_wide_qmv.py`'s gate at that shape.
- **(ii) NO-GO.** The spike measures LOSS or REFUSED under the same protocol, or it fails the gate while its arithmetic class already has ensemble members.
  Recorded and stopped.
  REFUSED means undecided and never means measured-bad (ADR 0015), so a REFUSED cell stops the technique for this spike without licensing any claim about it.
- **(iii) NEW CLASS.** Section 6 named the technique's class as NEW, so the shipped tolerance's floor holds no member of that class and a flag would be the ten-false-positives mechanism of ADR 0016 all over again.
  Its spike's gate result is READ ONLY AFTER a membership block: a member of that class added under its own pre-registration, then K re-derived under the standing membership-before-K order (ADR 0012).
  After the membership block lands, (i) and (ii) apply unchanged.

Two clauses that bind all three outcomes.

Every spike is verify-then-time: the gate runs before any timing, and a candidate that fails the gate is never timed, because a speed number about an unverified kernel is the thing this repo exists to refuse (ADR 0015 records that every timed cell of the boundary pricing was verified first).
And no spike's number is a product claim: the M = 1 spike measures one shape at one width against stock MLX, and the end-to-end baseline every technique must ultimately beat is the batch-1 per-stream tokens/s that M2 pins, whatever `wide_qmv` does there.

## 7. The optimiser's shape

The generator does not exist.
No module under `kernelverify/` emits a candidate kernel: `kernelverify/runners/specialize.py` substitutes into a fixed template, `kernelverify/pack/` holds three hand-written Metal kernels (`wide_qmv.py`, `kv_attention.py`, `moe_dispatch.py`), and nothing proposes a new one.
Almost everything downstream of the generator does exist, and it exists in the shape the sub-question B papers say it should, which is why this section is a wiring diagram and not a design.

### 7.1 The loop, as it would sit on this repo

```
    generate                gate                     price                    keep
  (MISSING)  ->  pack_wide_qmv.verify()  ->  price_qmv_boundary protocol  ->  emit_pack_certificates
      |                  |                             |                            |
      |            pack/verify.py                interleave.py                routed_windows.py
      |          NATIVE_OPS reference           MeasurementLock              (derived at import
      |          ensemble floor, K_QUANT        idle gate, budget             from the recording)
      |                  |                             |
      +-- specialize.py  +--- GateEvidence ------------+--- a LOSS or REFUSED cell stops here,
          or raw MSL          (evidence.py)                 and is recorded, not re-argued
          (KernelSpec)
```

A candidate enters as a `KernelSpec` (`kernelverify/runners/spec.py`): raw Metal shading language plus the bindings that say what tensor sits at each buffer index and the launch grid the author intended, because Metal's compiler knows the type at buffer 0 and not its name, shape or intended thread count.
A family of candidates enters through `kernelverify/runners/specialize.py`, which fills `$NAME` placeholders in a template spec and returns a new validated spec; it lives outside the runner deliberately, so the runner only ever sees finished specs and the specialization a candidate was built from stays with the caller, where a certificate can cite it.
An unfilled placeholder and an unused substitution both raise, so a generator that emits a malformed family fails loudly at the door.

Every candidate is verified before it is timed.
`bench/pack_wide_qmv.py` enforces exactly that order and says so in its own docstring: no timing is printed unless every case verifies through the crash-isolated runner against the shipped verdict.
The verdict itself comes from one place, `kernelverify/pack/verify.py`, which is the only route between a pack surface and a pass/fail claim, so no gate states a tolerance of its own and a change to the shipped formula cannot leave an old copy applying in a corner.
The gate returns a `GateEvidence` structure (`kernelverify/pack/evidence.py`), one `SpecializationEvidence` per compile-time specialization and one `CaseEvidence` per isolated case, so nothing downstream parses stdout.

A verified candidate is priced under `bench/price_qmv_boundary.py`'s protocol: arms interleaved within each round, a reference-arm canary whose spread caps at 1.5x, `ratio_lo > 1` for WIN, the whole interval below 1.0 for LOSS, REFUSED otherwise, all under the machine-wide `MeasurementLock` and the footprint budget (`bench/machine_state.py`, `bench/memory_guard.py`, `bench/interleave.py`).
A kept candidate becomes routing (`kernelverify/pack/routed_windows.py`, derived at import from the committed recording and pinned by three sha256s: the recording, the kernel source, the launch config) and a certificate (`bench/emit_pack_certificates.py`, which captures the generated translation unit in a fresh process and behaviorally validates it before writing anything).

### 7.2 What exists, and what is missing

| piece | state | where |
|---|---|---|
| candidate representation | exists | `kernelverify/runners/spec.py` (`KernelSpec`: source, bindings, launch) |
| candidate family expansion | exists | `kernelverify/runners/specialize.py` (`$NAME` substitution, both mistakes raise) |
| crash-isolated execution on Metal | exists | `kernelverify/runners/device.py`, `metal.py`, `worker.py` |
| the correctness gate | exists | `bench/pack_wide_qmv.py` -> `kernelverify/pack/verify.py` -> `NATIVE_OPS` reference, ensemble floor, `K_QUANT` |
| structured gate evidence | exists | `kernelverify/pack/evidence.py` (`GateEvidence`) |
| the pricing protocol | exists | `bench/price_qmv_boundary.py`, `bench/interleave.py`, `bench/machine_state.py`, `bench/memory_guard.py` |
| the certificate emitter | exists | `bench/emit_pack_certificates.py` |
| evidence-derived routing | exists | `kernelverify/pack/routed_windows.py` |
| **the generator** | **missing** | nothing in the repo emits a candidate kernel |
| **the search space** | **missing** | tile width M and rows-per-simdgroup R exist only as pinned launch-config DATA in `routed_windows.py`; packing layout, unroll, threadgroup-memory budget and accumulator layout are not parameters anywhere |
| **the candidate store** | **missing** | a candidate's source, its provenance, its gate verdict and its price have no single home; `routed_windows.py` and the certificates hold survivors only, so a rejected candidate leaves no record and can be re-proposed forever |
| **the held-out shape sweep** | **missing** | the gate covers the dispatch shapes the pack routes to, which are the shapes a generator would be tuned on |
| **the tolerance-free gates** | **missing** | NaN and infinity propagation, run-to-run determinism, and shape-change robustness are not a standing gate set |

### 7.3 What the sub-question B papers say to add, and what they say we already have

Three of the four B rows converge on things this repo either has or is one step from.

**The held-out sweep is the gap with a measured price.**
Metal-Sci (arXiv 2605.09708) ran the same evolutionary loop this section describes, on Apple silicon Metal, and its held-out size sweep caught two failures that in-distribution scoring could not see: a sampler whose covariance was off by about 10 sigma at a size the template enumeration never covered, and a kernel reporting a 2.95x in-distribution speedup that collapsed to 0.23x on a held-out size because it fell back to a quadratic path outside its enumerated set.
Both are the same failure mode: an optimiser tunes what it is scored on.
Our gate scores the dispatch shapes the pack routes, and a generator would be tuned on exactly those, so the loop needs shapes and widths withheld from the generator's feedback and read only at the end.
This repo already has the discipline in its calibration harnesses (the independent draw of ADR 0009 and ADR 0012, and V5 of the current plan is making that draw a test rather than a construction), so what is missing is the sweep in the pack gate, not the idea.

**The tolerance-free gates are cheap and we do not have them as a set.**
arXiv 2608.12700 audited 2,638 machine-generated kernels behind twelve adversarial gates, several of which need no tolerance at all: a kernel that returns an ordinary number where the true answer is a NaN or an infinity, one that differs from run to run, one that breaks when the shape changes, one that accumulates in fp16 where the reference keeps an fp32 total.
39.5% of the audited kernels were broken beyond any tolerance argument.
The fourth of those gates is our contract's own line and we enforce it through the tolerance rather than directly; the first three are not a standing gate anywhere in this repo.
They cost nothing to run and they cannot produce a false positive, which is the class of check that belongs in front of a tolerance rather than inside one.

**What we already have is the thing the field is missing.**
The Correctness Illusion (arXiv 2606.20128) recommends op-schema-aware boundary shapes, per-operation and per-dtype tolerances against an fp64 CPU reference, multiple shapes and dtypes per operator, and full error distributions.
That list is this repository's battery, measured rather than proposed: ADR 0001 through ADR 0005 are the shape, dtype and structured-mode axes and the conditioning-aware tolerance, and the mutation scoring in `bench/score_oracles.py` scores a test policy against a synthesised fault population rather than against a fixed corpus.
The formal-methods entries mark the boundary of the alternative: ProofWright (arXiv 2511.12294) establishes semantic equivalence for a class of element-wise kernels, and a quantized matvec is a reduction, not an element-wise kernel; VOLTA (arXiv 2511.12638) claims soundness and completeness for a stated class of GPU kernels and this review could not determine from its abstract how it treats a reordered floating-point reduction, which is the entire question a tolerance exists to answer.

### 7.4 The one thing to build first

The loop above has exactly one hole that stops it from running end to end, and it is not the generator.
It is the candidate store: without a place where a candidate's source, its provenance, its gate verdict and its price live together, every spike in section 5 produces a number in a terminal and the next session re-proposes the same kernel.
`routed_windows.py` records what won and `bench/results/` records the recordings, so the store's shape is already set by the two things it has to feed.
A generator writing into a store the gate and the pricing protocol already read is a smaller change than a generator wired directly into either.
