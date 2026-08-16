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

| paper | year | sub-q | technique in one line | evidence | Metal | honest | bytes/weight | reported gain (hardware) | why it might beat stock at batch 1 |
|---|---|---|---|---|---|---|---|---|---|

## 5. Synthesis (filled by R2)

## 6. What the verifier would need (filled by R2)

## 7. The optimiser's shape (filled by R2)
