# ADR 0008: Native operators, and the artefact-axis question answered by measurement

Date: 2026-08-14
Status: Accepted

## Context

The corpus supplies ten operators with external fp64 references, but the product's attack surfaces (eng review Premise 3) need operators the corpus never had: quantized matmul, MoE expert dispatch, and KV-cache attention.
Certifying a kernel requires a reference, fault seams, a schema, and a calibrated tolerance for its operator; that expansion was named the true critical path of the pack (eng review decision 3A).

ADR numbering note: 0005 is the corpus-K anchoring (phase0 worktree), 0006 the Metal runner (metal-runner worktree), 0007 the machine baseline (baseline worktree); numbers are serialized by the coordinating session because three worktrees drafted a 0005 independently.

## Decision 1: a native operator registry beside the corpus, same schema shape

`kernelverify/schemas/native_ops.py` carries, per operator: a meta dict in the exact shape corpus meta.json uses (so the case space, input modes, and every battery policy work unchanged), an in-process fp64 reference, a per-family tolerance with its own calibrated K, and an augment hook for deterministic inputs a case implies.
`KERNEL_TO_OP` in the catalogue unifies resolution; `build_verdicts` branches on the registry.
Ground-truth honesty rule: every native reference is cross-checked against an independent implementation in tests (MLX's own ops).

Two calibrated K values coexist and must never collapse into one constant: K=1.5 for unquantized ensembles (anchored, not merely fitted, by the corpus-K work on the phase0 worktree) and K=3 for the MLX-affine quantization contract (Phase 0).

## Decision 2: weights are not activations

The input modes generate every tensor at activation ranges; a weight matrix at uniform[-10,10] is an operating point no model has, and at D_IN=1024 it overflows fp16 outright, turning every fp16 case into a range failure instead of a fault measurement.
The augment hook rescales generated weight tensors to a realistic magnitude while preserving the mode's structure.
Case dims are chosen for fault expression, not perf realism: dilution (ADR 0001) makes short reductions the stronger signal.

## Decision 3: the fingerprint names the ensemble membership

The verdict-cache tag carried only K, but K is the smaller half of the tolerance: swapping an ensemble member moves every floor while a K-only tag stays byte-identical.
Found independently on the phase0 worktree (its commit f47dc39) whose corpus-side sweep also showed the risk lives in membership, not K (zero verdict movement across K=1.0..6.0 over 9,678 pairs).
The fingerprint now names every floor, quant, and moe ensemble member.

## The population: 56 faults, 49 viable, 7 equivalent with reasons

Eleven native mutations joined the catalogue.
Nine are viable, broadly detectable (quantized_matmul: scales rotated 414/560 cases, nibble order swapped 448/560, bias dropped 560/560, group size halved 448/560, fp16 intermediate dequant 280/560 - matching Phase 0's boundary measurement; moe_dispatch: renormalization dropped 287/320, expert off-by-one 320/320, fp16 router 149/320, top-1-for-top-2 302/320).
Two measured equivalent, both for honest reasons now recorded in tests: softmax-after-top-k is mathematically identical to renormalizing the full softmax, and exact logit ties never occur under continuous draws, so the tie-break direction never expresses under any battery input mode.
The tie-break exclusion is itself informative: that fault class needs an artefact-style structured-router case to express, which the catalogue-growth discipline can add when a real incident demands it.

## The pre-registered question, answered: the artefact axis is NOT yet justified

TODOS.md holds the artefact-fault axis with the phase-2 prediction that input-space policies would drop sharply against artefact faults.
Measured on the enlarged population, the prediction did not survive:

| Policy (49 viable faults) | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| Single shape, fp32 (benchmarks) | 79.6% | 79.6% | 79.6% | 79.6% |
| Random, all input modes | 90.3% | 97.1% | 99.5% | 99.8% |
| Boundary singles + random (ADR 0002) | 95.9% | 97.1% | 99.2% | 99.8% |
| Boundary pairs + random (ours) | 95.9% | 98.0% | **100.0%** | **100.0%** |

The 100.0% cells are exact (zero misses over 40 seeded runs); the corpus subset stays 100% from B=4; the only small-budget survivors are the same two pair-located faults as ADR 0003 (l2norm epsilon, flash init-max-zero).
Why the prediction failed: anchored against the contract, artefact corruption expresses at most input points - the input axis reaches it because the CONTRACT, not the input, carries the fault's signature.
The axis therefore stays deferred, with its justification bar unchanged: a fault class with no input-space signal at any case (the cancelling permutation remains the known member, and it is provably equivalent numerically - the structural artefact check, not a test policy, is the tool for it).

## Consequences

The pack's kernels have their verification substrate: MoE dispatch kernels verify against the pinned Qwen3-class routing contract today, quantized kernels against the Phase 0 contract, and the first pack kernel (the wide-batch quantized matvec, 1.2-1.4x verified win at batch 6-10) already used it.
KV-cache attention follows the same registry path next.
The sales table strengthens: a single-shape harness holds at 79.6% against the enlarged population while the boundary-pairs battery holds exact 100% at 16 evaluations per operator, now including fault classes no published corpus contains.
