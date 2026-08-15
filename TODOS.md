# TODOS

## Extend the memory budget + lock pattern to the pricing and A/B harnesses

- What: apply the survival pattern that `bench/calibrate_quant_serving.py` now carries to `bench/price_qmv_boundary.py` (the pack boundary-pricing harness) and the four-arm `bench/serve_sub4bit.py` (the serving A/B harness): the phys_footprint budget guard with its distinct refusal exit (ctypes `proc_pid_rusage`, since Jetsam kills on footprint while RSS under-reads it by an order of magnitude), the machine-global single-instance lock keyed to each harness's identity with dead-pid reclaim, and the `kern.memorystatus_level` available-memory refusal before each large cell.
- Why: the night of 2026-08-15 produced three Jetsam kills from the serving calibration (66.7, 69.4 and 39.5 GB Python footprints on a 36 GB machine), and the 03:29 event was a two-harness collapse: the 39.5 GB serving run plus a 27.9 GB pricing-probe Python whose idle gate co-fired; any large-footprint harness without the budget and lock can reproduce that collapse.
- Pros: the pattern is already built, tested and measured in `calibrate_quant_serving.py` (budget guard, `acquire_lock`/`release_lock`, `require_available_memory`, and the refusal-exit vocabulary), so the extension is mostly wiring; the runner-level buffer-pool fix in `kernelverify/runners/device.py` already protects every harness that dispatches through `CompiledKernel.run`, which removes the largest single leak class fleet-wide.
- Cons: the two harnesses live on other lanes' branches, so the wiring belongs to whoever next touches those files, and each harness needs its own lock identity rather than a copy of the serving lock path.
- Context: tonight's fix landed on branch phase0; the attribution probe convicted per-case Metal buffer allocation (dirty MTLBuffer pages never return to the OS on this stack, ~0.83 GB retained per lm_head dispatch), and the cross-harness serialization gap itself is coordinator territory, fixed at the coordinator level and out of scope for the in-repo pattern.
- Depends on / blocked by: this fix landing on phase0 and reaching the branches that carry `price_qmv_boundary.py` and `serve_sub4bit.py`; applied when those files are next touched.

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

## Pack-operator battery extension: the certificate upgrade path

- What: extend the fault catalogue and the mutation-scored battery to the pack operators (quantized_matmul wide-tile, moe_dispatch routing and dispatch, kv_attention), so pack certificates can graduate from the pack-gate policy with its handful of fixed cases to the 16-eval battery with a measured escape rate against a synthesized fault population.
- Why: today every pack certificate honestly names the pack-gate policy and its true case budget (D10), which is a fixed-case sweep, not a mutation-scored battery; the catalogue_fingerprint field reads "none" because no fault population exists for these operators, and that is the certificate's weakest block.
- Pros: the market evidence says this is the differentiator, not a nice-to-have - PolyQ-class mixed-precision kernel compilers (arXiv 2607.14618) validate end-to-end via perplexity only, and Meganeura (arXiv 2608.01563) gates kernels on fixed numeric thresholds; neither has per-kernel batteries scored by mutation against a fault population, which is exactly what this repo already does for the corpus operators.
- Cons: each pack operator needs keyword seams in a reference implementation plus catalogue entries (the wide_qmv striding and kv_attention new-entry seams are the obvious first faults); real work, on the order of the ADR 0008 artefact-axis effort.
- Context: bench/emit_pack_certificates.py is the consumer - once the battery covers a pack operator, its certificate's policy block, budget, and catalogue_fingerprint upgrade in place, and the C4 softmax clause on kv_attention/routing can move from source-attestation to numeric once overflow-provoking structured modes exist for those operators.
- Depends on / blocked by: nothing external; the pack contracts are frozen in NATIVE_OPS and the certificate schema already separates the policy tiers.
