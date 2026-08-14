# TODOS

## Artefact-fault axis for the verifier

- What: extend the mutation catalogue with 8-10 compiler-artefact faults (rotated bit maps, off-by-one block boundaries, stale permutations after a merge, packed-block dtype mismatches) behind new kernel seams (bit_map=, permutation=, block_boundary_offset=), add the structural artefact check (exact comparison of decoded artefact against the declared plan), and record the cancelling-permutation equivalence proof as the textbook exclusion.
- Why: no input-space test policy can reach artefact faults (a permutation applied consistently to weights and activation basis is bit-identical at every input); the phase-2 analysis rated this the weakest competitive threat surface in the field.
- Pros: extends the moat into quantized-artifact territory exactly where Phase 0's contract machinery goes; the predicted policy-table collapse on the enlarged population is itself a publishable measured result.
- Cons: real catalogue and seam work (~a week with CC assistance); not needed for the current 90-day gates.
- Context: fully designed as candidate C3 in the phase-2 spec (data/kv-papers-a1 supporting docs), with the kill criterion pre-registered: if the boundary-pairs policy stays near 100% on the enlarged population, the axis is redundant; if many new mutations measure equivalent, the class does not express.
- Depends on / blocked by: Phase 0 (contract-anchored K for MLX-affine quant) passing; the artefact schema from kernelverify/schemas/ contract work.
- Status update (2026-08-14, ADR 0008): the predicted policy collapse did NOT materialise - contract-anchored artefact faults are broadly input-detectable (boundary-pairs held exact 100% at B=16 on the 56-fault population).
  The bar for this axis is now: a fault class with zero input-space signal at every case; the cancelling permutation remains the known member and is structural-check territory, not test-policy territory.

## Post-spike mlx-lm integration plan

- What: after the runner lane's mlx-lm spike reports, write the full integration plan (ship the verified kernels into mlx-lm as a usable pack) and run it through /plan-eng-review before any product code.
- Why: the spike exists to price this step; without a captured pointer its numbers risk sitting unconverted while the fleet works elsewhere.
- Pros: the motivation, the ruled pre-registration (design doc vlad-next-lanes-design-20260814.md, D4/D9/D12), and the entry point survive a pause of any length.
- Cons: none of substance; one entry.
- Context: the spike's pre-registration is fully ruled (three arms, MDE derived before running, stock-fp16 demotion rule, T=512/T=896 under the 1024 capacity); the integration decision pivots on whether the decode delta beats spread, and a swallowed delta means the prefill/batch pivot is a NEW kernel plus a new verification cycle, not a re-aim.
- Depends on / blocked by: the runner lane's spike findings report.
