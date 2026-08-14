# ADR 0007: The machine baseline - llama.cpp master against a measured roofline

Date: 2026-08-14
Status: Accepted

## Context

Phase 2 of the plan is an optimiser that generates faster kernels with the verifier gating every candidate, and Vlad's Mac on the Metal backend is a first-class target.
An optimiser cannot report a speedup without two numbers this repo did not have.
The first is what the incumbent already achieves on this machine, because "faster" is meaningless without an incumbent.
The second is what the machine can do at all, because a kernel can beat a bad baseline by 2x and still sit 10x below the hardware, and only the second number tells you which of those you are looking at.

This ADR establishes both, and pins them so a later candidate is compared against a recorded state rather than a remembered one.

## What was measured, and where it lives

| Thing | Value |
|---|---|
| Machine | Mac15,7, Apple M3 Pro, 6 performance + 6 efficiency cores, 18-core GPU, 36 GiB unified, macOS 26.5.2 |
| Power state | AC, low power mode off (recorded per run; the first probe on battery read 6% low on bandwidth) |
| llama.cpp | commit `a94d563ed`, build 10423, upstream master, checkout at `/Users/vlad/llama.cpp` |
| Build | `-DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`, backends MTL + BLAS |
| Models | Qwen2.5-0.5B at q4_K_M, q8_0 and f16; Llama-3.2-1B q4_K_M; Qwen2.5-7B q4_K_M, at `/Users/vlad/models/gguf` |
| Flags | llama.cpp defaults throughout, because the baseline has to be what a user actually gets |

Scripts and results:

| Path | What it does |
|---|---|
| `bench/metal/roofline_probe.mm` | measures the machine's ceilings on the GPU and the CPU |
| `bench/roofline.py` | builds and runs the probe, writes `bench/results/roofline.json` |
| `bench/gguf_info.py` | reads a GGUF tensor table into the flop and byte models |
| `bench/baseline_llamacpp.py` | runs llama-bench, places each result on the roofline |
| `bench/baseline_kernels.py` | runs upstream's own kernel perf set, scores each case against the roofline |

## Decision 1: the ceilings are measured on this machine, not read off a spec sheet

A spec ceiling is ALU count times boost clock, and no kernel sustains it.
The probe measures streaming bandwidth from a read-dominated kernel with four loads in flight, and FMA throughput from eight independent accumulator chains, both timed with the command buffer's own GPU timestamps and reported as the best of twelve passes.

| Ceiling | Measured | Spec | Measured / spec |
|---|---|---|---|
| GPU memory bandwidth | 135.4 GB/s | 153.6 GB/s (LPDDR5-6400, 192-bit) | 88% |
| GPU fp32 FMA | 6.24 TFLOP/s | 6.44 TFLOP/s (18 cores x 128 ALUs x 2 x 1.398 GHz) | 97% |
| GPU fp16 FMA | 6.30 TFLOP/s | - | - |
| GPU fp16 FMA, packed half4 | 6.32 TFLOP/s | - | - |
| CPU streaming triad, 12 cores | 121.3 GB/s | - | - |
| CPU fp32 GEMM via Accelerate | 1.53 TFLOP/s | - | - |

Ridge point, where a kernel stops being bandwidth-bound: 46.7 flop/byte on the GPU, 12.6 flop/byte on the CPU.

Three consequences fall out of the table itself.

fp16 buys no arithmetic on this machine.
6.30 against 6.24 TFLOP/s is a 1% difference, and packing into half4 adds another 0.3%.
Halving precision here buys bandwidth and register pressure, never flops, so any future kernel that drops to fp16 for speed has to justify it on traffic.

The FMA ceiling is also the matmul ceiling.
llama.cpp's own device init on this machine reports the tensor API disabled for pre-M5 devices, so there are no matmul units to reach past the ALUs, and a tuned GEMM probe would only reproduce the FMA number from below.
That is why no separate GEMM ceiling was measured.

The CPU needs its own ceilings because llama.cpp's CPU path is a real backend here.
Scoring a CPU result against a GPU ceiling produced a 13% figure that meant nothing; against the CPU's own ceilings the same run is 56%.

Reproducibility: across repeat invocations the ceilings move by at most 1.6%, after the CPU probes were raised to best-of-50 because best-of-5 was still swinging 39%.

## Decision 2: the flop and byte models come from the model's tensor table

The first version of this baseline charged `2 * model_n_params` flops per token, which is the usual shorthand.
It scored Qwen2.5-0.5B prompt processing at **110.7% of the machine's measured flop ceiling**, which is impossible, and therefore falsified the model rather than the measurement.

Three corrections, all read from the GGUF tensor table by `bench/gguf_info.py`:

| Correction | Why the shorthand is wrong | Size of the error |
|---|---|---|
| The token embedding table is a gather, not a matmul | it is indexed, not multiplied | 22% of all parameters on a 0.5B model with a 151k vocabulary |
| The output head runs once per decode call, not once per prompt token | llama.cpp materialises logits only for the last position | 272 MFLOP charged 512 times instead of once, at pp512 |
| Attention score matmuls scale with context, not with parameter count | they are absent from a parameter count entirely | adds a term that grows as T squared |

With the corrected model no cell exceeds its ceiling, and the largest model lands at 88.8% of bandwidth, which is where a well-tuned generation path should be.
`tests/test_gguf_info.py` pins all three corrections.

## Result: llama.cpp at master on this machine

Every row is scored against the ceiling that applies to its own backend and its own regime.

| Model | Backend | Test | tok/s | Achieved | Of its ceiling | Bound by |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b q4_k_m | metal | pp512 | 5678 | 4.19 TFLOP/s | 66.3% | compute |
| qwen2.5-0.5b q4_k_m | metal | tg128 | 218.4 | 85.8 GB/s | 63.3% | memory |
| qwen2.5-0.5b q8_0 | metal | pp512 | 6044 | 4.46 TFLOP/s | 70.5% | compute |
| qwen2.5-0.5b q8_0 | metal | tg128 | 191.5 | 100.7 GB/s | 74.4% | memory |
| qwen2.5-0.5b f16 | metal | pp512 | 6270 | 4.63 TFLOP/s | 73.2% | compute |
| qwen2.5-0.5b f16 | metal | tg128 | 115.8 | 114.5 GB/s | 84.6% | memory |
| llama-3.2-1b q4_k_m | metal | pp512 | 2457 | 4.87 TFLOP/s | 76.9% | compute |
| llama-3.2-1b q4_k_m | metal | tg128 | 136.2 | 109.2 GB/s | 80.7% | memory |
| qwen2.5-7b q4_k_m | metal | pp512 | 365.7 | 4.81 TFLOP/s | 76.1% | compute |
| qwen2.5-7b q4_k_m | metal | tg128 | 27.5 | 120.2 GB/s | 88.8% | memory |
| qwen2.5-0.5b q4_k_m | cpu | pp512 | 855.0 | 0.63 TFLOP/s | 41.2% | compute |
| qwen2.5-0.5b q4_k_m | cpu | tg128 | 193.7 | 76.0 GB/s | 62.7% | memory |
| llama-3.2-1b q4_k_m | cpu | pp512 | 436.5 | 0.86 TFLOP/s | 56.5% | compute |
| llama-3.2-1b q4_k_m | cpu | tg128 | 115.3 | 92.5 GB/s | 76.3% | memory |

Two things in that table are worth stating plainly.

Generation on a real-sized model is close to done.
The 7B reaches 88.8% of measured bandwidth, and the per-kernel table below shows the individual mat-vec kernels at 90 to 97%.
There is no 2x hiding in single-stream generation on this machine, and any proposal that claims one is claiming to beat the memory system.

The CPU is not the weak backend people assume.
On the 0.5B, CPU generation at 193.7 tok/s is within 12% of Metal's 218.4, because at that size the GPU is dominated by per-kernel dispatch rather than by work.

## The finding: the hole is at batch 3 to 16, at both levels of measurement

Sweeping prompt width walks arithmetic intensity across the ridge point, and the walk is not monotonic.

| Test | flop/byte | tok/s | GB/s | TFLOP/s | Bound by |
|---|---|---|---|---|---|
| pp1 | 3.1 | 133.9 | 107.1 | 0.33 | memory |
| pp2 | 5.5 | 254.4 | 101.7 | 0.56 | memory |
| pp4 | 10.4 | 273.5 | 54.7 | 0.57 | memory |
| pp8 | 20.1 | 299.7 | 30.0 | 0.60 | memory |
| pp16 | 39.6 | 756.2 | 37.8 | 1.50 | memory |
| pp32 | 78.5 | 1586.6 | 39.7 | 3.12 | compute |
| pp512 | 1241.9 | 2430.8 | 3.9 | 4.81 | compute |

At pp1 the machine runs at 79% of its bandwidth ceiling.
At pp512 it runs at 76% of its flop ceiling.
In between, at pp4 and pp8, it reaches neither: 40% and 22% of the bandwidth it was getting one step earlier, and under 13% of the flops it reaches later.

Upstream's own kernel perf set reproduces the same shape at a single matmul, at m=4096 and k=14336, scored against the roofline value at each case's own intensity:

| n | median flop/byte | median GFLOPS | median % of roofline |
|---|---|---|---|
| 1 | 2.9 | 341 | 94.4% |
| 2 | 5.8 | 581 | 88.2% |
| 3 | 8.7 | 703 | 60.1% |
| 4 | 11.6 | 643 | 42.3% |
| 5 | 14.4 | 574 | 28.4% |
| 8 | 22.9 | 633 | 20.7% |
| 512 | 769.7 | 4840 | 76.5% |

The mechanism was then measured rather than guessed, by timing the same q4_K matmul as the column count grows:

| n | time | vs n=1 | per column |
|---|---|---|---|
| 1 | 255.3 us | 1.00x | 255.3 us |
| 2 | 273.7 us | 1.07x | 136.9 us |
| 3 | 421.6 us | 1.65x | 140.5 us |
| 4 | 715.8 us | 2.80x | 178.9 us |
| 5 | 984.9 us | 3.86x | 197.0 us |
| 8 | 1400.8 us | 5.49x | 175.1 us |
| 512 | 12069.4 us | 47.27x | 23.6 us |

The second column is nearly free, which is what weight reuse looks like.
From the third column on, total time grows almost in step with the column count, which is what re-streaming the weights per column looks like.
At n=512 the GEMM path takes over and a column costs 23.6 us.
So a column at n=5 costs **8.3x** what the same column costs at n=512, on the same operator, the same shapes, and the same hardware.

f16 holds reuse further, staying flat from n=1 to n=5 (909 to 953 us) before degrading at n=8, so the cliff position is quantisation-dependent rather than fixed.

## Per-kernel baseline, and one outlier worth naming

Mat-vec at m=4096, n=1, k=14336, the shape that generation actually runs:

| type_a | GB/s | % of roofline |
|---|---|---|
| q8_0 | 131.4 | 97.1% |
| q4_K | 129.7 | 95.8% |
| q6_K | 129.3 | 95.5% |
| f16 | 129.2 | 95.4% |
| q4_1 | 128.4 | 94.9% |
| bf16 | 127.9 | 94.5% |
| q5_K | 127.9 | 94.4% |
| q5_0 | 126.7 | 93.6% |
| q5_1 | 122.3 | 90.3% |
| q2_K | 120.8 | 89.2% |
| **q3_K** | **73.6** | **54.3%** |

Every format sits between 89% and 97% of the memory system except q3_K, which reaches 54.3%.
It is the same shape as every other row, and q3_K moves fewer bytes per weight than q4_K, yet takes 344 us against q4_K's 255 us.
The follow-up section confirms this end to end and bounds it: it is an upstream kernel inefficiency that needs a large reduction dimension to appear, and it nearly vanishes at small ones.

93 of 184 MUL_MAT cases are deliberately left unscored: they broadcast or batch an operand, which changes the traffic model, and inventing one would be worse than leaving the cell empty.

## Follow-up: the three inferred claims, measured

The first version of this ADR shipped one finding as measured and three as inferred.
`bench/probe_baseline_gaps.py` closes all three, and one of them came back different from what was written.

### The q3_K deficit is real, and it is shape-dependent

The flag came from a single case in upstream's perf set, at m=4096 and k=14336.
Two checks, on shapes that case never touched.

Rebuilding the same 0.5B at six pure single-type quantizations, so only the mat-vec kernel changes:

| Pure type | Weight bytes | tok/s | GB/s | % of peak |
|---|---|---|---|---|
| Q2_K | 0.25 GB | 260.5 | 66.3 | 48.9% |
| Q3_K | 0.27 GB | 266.2 | 70.6 | 52.1% |
| Q4_K | 0.33 GB | 236.8 | 77.6 | 57.3% |
| Q5_K | 0.37 GB | 220.5 | 80.5 | 59.4% |
| Q6_K | 0.50 GB | 193.5 | 96.9 | 71.5% |
| Q8_0 | 0.53 GB | 190.2 | 100.0 | 73.9% |

At 0.5B shapes there is no q3_K cliff.
The column is a smooth trend in model size, which is the fixed per-token cost measured below, and q3_K comes out ahead of q2_K here and 1.1x behind q4_K rather than 1.75x.
Across two runs q3_K and q2_K swap order, which is the 6% run-to-run spread small models carry, so the only safe reading is that they are indistinguishable.
That alone would have falsified the claim as written.

Repeating it at the 7B's shapes, where the reduction dimension is 18944 rather than 4864:

| Pure type | Weight bytes | tok/s | GB/s | % of peak |
|---|---|---|---|---|
| Q3_K | 3.04 GB | 25.03 | 76.1 | 56.2% |
| Q4_K | 3.98 GB | 30.32 | 120.7 | 89.1% |

Here it is unambiguous.
q3_K moves 24% fewer bytes than q4_K and is still 17% slower in absolute tokens per second, which no traffic model permits.
The gap is 1.59x in achieved bandwidth, on a second measurement path entirely independent of the perf set that raised it, and it reproduced across two runs at 1.53x and 1.59x.

So the deficit is real but needs a large reduction dimension to appear: about 1.55x at the 7B's shapes, indistinguishable from noise at the 0.5B's, and the perf set found it at k=14336 in between.
The two 7B files are requantized from Q4_K_M and are numerically junk; they exist only to hold the right tensor types at the right shapes for a timing test, and must never be used for anything else.

### The batch valley is a property of the backend, not of one shape

Sweeping prompt width on two models a factor of 14 apart in size:

| n | 0.5b, % of roofline | 7b, % of roofline |
|---|---|---|
| 1 | 59.8% | 93.6% |
| 2 | 55.4% | 81.9% |
| 3 | 50.4% | 58.3% |
| 4 | 41.1% | 38.7% |
| 6 | 31.6% | 26.9% |
| 8 | 24.7% | 19.5% |
| 12 | 22.5% | 29.3% |
| 16 | 21.9% | 29.9% |
| 32 | 28.3% | 59.0% |
| 64 | 42.0% | 68.4% |
| 512 | 66.0% | 77.0% |

Both fall into it, both bottom out between n=8 and n=16, and both recover only past n=32.
The 7B enters the valley from 93.6% at n=1 and loses 74 points of roofline by n=8, which is the sharpest gap anywhere in this baseline.
Every cell in this table reproduced within 1.5 points across two runs.

### The small-model gap is dispatch cost, and it is 0.99 ms per token

Fitting time per token against bytes per token across the five baseline models, over a 10x range in size:

| Fit | Value |
|---|---|
| Fixed cost per token | 0.99 ms |
| Implied bandwidth from the slope | 123.8 GB/s, which is 92% of the measured ceiling |
| Worst residual | 0.418 ms |

The slope landing within 8% of the independently measured bandwidth ceiling is what makes the intercept believable: the model is not absorbing error into a free parameter.

| Model | ms/token | Share that is fixed cost |
|---|---|---|
| qwen2.5-0.5b q4_k_m | 4.58 | 21.6% |
| qwen2.5-0.5b q8_0 | 5.22 | 19.0% |
| llama-3.2-1b q4_k_m | 7.34 | 13.5% |
| qwen2.5-0.5b f16 | 8.64 | 11.5% |
| qwen2.5-7b q4_k_m | 36.38 | 2.7% |

That is the whole explanation for small models looking inefficient.
Nothing about the 0.5B's kernels is worse than the 7B's; it just pays the same fixed millisecond against a twelfth of the work.

## Consequences

The optimiser now has a target list ordered by measured headroom rather than by intuition.

| Target | Headroom | Confidence |
|---|---|---|
| Batched matmul at n = 3 to 16 | 3x to 5x against the roofline; 8.3x per column against upstream's own n=512 path | mechanism measured; confirmed at kernel level and end to end on two model sizes |
| q3_K mat-vec at large k | about 1.55x at the 7B's shapes, on two independent measurement paths | confirmed twice, and bounded: it disappears into noise at 0.5B shapes |
| Fixed per-token dispatch cost | 0.99 ms/token, worth 21.6% of a 0.5B's generation time | measured, with the fit's slope validating it against the bandwidth ceiling |
| Single-stream generation on a 7B | 88.8% of bandwidth, so roughly 1.1x exists at most | measured directly |

The batch 3 to 16 hole is the interesting one for a product, because that is where speculative decoding and small-batch serving live, and it is 5x off the machine rather than 10% off.

What a candidate has to beat, and by how much.
Metal rows reproduce within 0.3 to 2% standard deviation over 5 reps, except small-model generation at 6%.
CPU rows are far noisier, up to 19%.
So a Metal improvement under about 3% is not distinguishable from noise at this rep count, and any CPU claim needs more reps before it means anything.

The caveats that travel with these numbers.
The traffic model counts weights, the KV cache and the output, and ignores intermediate activation traffic, so achieved-bandwidth figures are slight underestimates and the true percentages are marginally higher than shown.
Two mat-vec kernels measured at 100 to 101% of an earlier, looser bandwidth ceiling, which is what forced the read probe to be tightened; the current ceiling is the highest rate any probe on this machine has sustained, and it is a lower bound on the truth rather than a proven maximum.
Everything here is one machine, one thermal state and one llama.cpp commit, which is exactly why the commit, the flags, the power state and the model files are all recorded in the result JSON.
