# TODOS

## The shipped K is not derived over the shapes it must cover

- What: make the quantized K derivation read every harness's records - the device grid's three synthetic shapes AND the serving grid's six Qwen3-4B shapes - and take the maximum demand, instead of each harness deriving a K from its own shapes and printing it as `shipped K`.
- Why: on 2026-08-15 the device harness printed `shipped K: 3.0` from a peak demand of 2.766 over its three shapes, while the serving grid demanded 3.120 at 2560x9728; adopting the printed line would have shipped a K that the next serving run refuses through its own DEMAND MISS branch (ADR 0016).
  Nothing in the code connects the two, so the safeguard today is a human noticing.
- Pros: the number a harness prints becomes the number that ships; a new shape family (a second model, a new serving grid) automatically participates instead of silently sitting outside the derivation.
- Cons: couples two harnesses that currently run independently; a third small script that reads both committed record sets and derives K once keeps them independent and is probably the better shape.
- Context: ADR 0016's decision section has both readings and the cell that binds each; `bench/results/quant_device_adequacy.json` and `bench/results/quant_serving_adequacy.json` are both committed, so the inputs already exist.
- Depends on / blocked by: nothing technical.

## The int-domain class rests on one outlying member

- What: decide whether the quantized contract's int-domain class needs a second member, and if so which real kernel it stands for; then re-derive K over the enlarged membership.
- Why: the ADR 0016 repair moved `factored-groups` from the most accurate member on constant rows to the least, and its leave-one-out spread against the six-member shipped floor went from 4.076 to 7.561 while the next int-domain member, `factored-serial`, reads 1.692.
  That is an adequacy statistic and not a live flag - the shipped floor contains the member, so a candidate rounding like it is judged against a floor holding its own error - but it says one member carries a whole class.
- Pros: the standing repair order (membership before K) says this is the membership question to ask before any K move; a second int-domain member would also make the class's leave-one-class-out diagnostic meaningful.
- Cons: a companion member was proposed for this class once and withdrawn, because it targeted the statistic while the false positives stayed live; those are now closed, so the same proposal has to be re-argued on its own merits rather than as a fix for something else.
- Context: ADR 0016, section "The open question the repair exposes"; the numbers are in `tests/test_quant_contract_members.py::test_k_stays_four_and_the_member_now_carries_its_class_alone`.
- Depends on / blocked by: nothing; it is a contract-population decision for the coordinator and Vlad.

## The fp16 ensemble floor is inert: compute it before storage-dtype rounding

- What: at float16 activations every quant contract member returns `.astype(x.dtype)`, so the output's own fp16 rounding dominates every internal difference and all eleven implementations report the identical max error; the floor then measures one common rounding step instead of class spread, `base_tol` wins in 768/768 fp16 serving records, and `k_demand`'s `e / floor` reported 195 where the shipped tolerance was exceeded 12.4x.
  The repair is to compute the ensemble floor BEFORE the final storage-dtype rounding (members return their pre-rounding fp32 values to the floor computation, the verdict comparison unchanged), so the fp16 floor measures implementation diversity again and the demand metric points at the right cell size immediately.
- Why: this inert floor is the root cause of the misread ADR 0013 number (a K demand read as an error magnitude), and the both-readings print ADR 0014 shipped is the label on the symptom, not the fix.
- Pros: makes `k_demand` a real K demand at fp16; would have reported 12.4x instead of 195.2 and pointed at the B1 cell size on the first run; changes no verdict semantics.
- Cons: redefines the floor half of the shipped tolerance, so the whole K derivation chain (ADR 0005, 0009, 0012) needs re-measurement under the new floor, which is a pre-registered calibration pass, not a patch.
- Context: mechanism dissected in `bench/.cache/b1-cliff-investigation.md` (hypothesis 2, confirmed one level down) and recorded in ADR 0014's reporting clarification.
- Depends on / blocked by: coordinator ruling plus a pre-registered rerun of the K calibration; must be planned through /plan-eng-review like every lane task.

## B16 mechanism demonstration (optional corroboration for the ADR 0014 exclusion)

- What: a bit-exact CPU emulation of `affine_qmm_t`'s threadgroup half tile, with the fp32-tile ablation, the way `bench/.cache/b1-cliff/qmv_emulate.py` proved the B1 mechanism; the B16 exclusion stands on the structural bar (verified source reading, D2) and this would add the mechanism proof at its measured strength.
- Why: B16's numbers (1.9x the floor median, up to 5.3x at fp16, masked under `base_tol` at max 0.34x) are recorded without a mechanism decomposition, and a demonstrated mechanism would make the exclusion's corroboration symmetric with B1's.
- Pros: CPU-only, no GPU slot, no measurement lock; the emulation harness pattern already exists beside the investigation.
- Cons: pure corroboration - D2 makes it unnecessary for the exclusion to stand - so it should never displace lane work that moves a gate.
- Context: ADR 0014 records the B16 exclusion and names this as the optional follow-up; the tile structure is quoted in `bench/.cache/b1-cliff-investigation.md` (loose ends).
- Depends on / blocked by: nothing.

## llama.cpp batched-serving baseline (deferred by D6)

- What: add llama.cpp n_parallel decode cells to `bench/measure_baselines.py`, so the serving matrix carries both stacks and the mlx-only `batch_decode` cells gain a cross-stack counterpart.
- Why: deferred by ruling D6 of the 2026-08-15 pivot review (design doc vlad-pivot-kernel-design-20260815.md) because the producer's llama.cpp wrapper cannot measure parallel decode honestly today.
- Pros: completes the serving comparison the product claim lives in; the mlx-only labels on the `batch_decode` cells can then be lifted cell by cell as real A/B groups form.
- Cons: new parser against an unversioned text format is exactly the kind of code that breaks silently on a llama.cpp bump, which is why it gets its own block rather than riding along in this one.
- Context: `llama-bench`, the binary the producer wraps via `-o json`, has no parallel-sequence mode at the pinned commit (verified 2026-08-15: its only batch knobs are `-b/--batch-size` and `-ub/--ubatch-size`, which chunk one sequence's prompt, not independent streams); the honest path is `llama-batched-bench`, which does run n_parallel decode streams but emits a text table rather than JSON, so it needs a new sample runner plus a dedicated output parser under the same interleave, idle-gate, and dispersion discipline as every other cell.
- Depends on / blocked by: nothing technical; deliberately NOT built in the 2026-08-15 block, and it must be planned through /plan-eng-review before dispatch like every lane task.

## STEP 3's per-block records are checkpointed and never read back

- What: `bench/calibrate_quant_serving.py` writes `grid_partial` (the records accumulated so far) after every (shape, draw, seed) child, but `--resume` reuses steps at STEP granularity only, so nothing ever reads that key; the run drops it (`steps.pop("grid_partial", None)`) once the step finishes.
  The work is to make a crash inside STEP 3 resumable from the last finished block: reuse `grid_partial` when the fingerprint matches, and skip the (shape, draw, seed) blocks it already holds.
- Why: STEP 3 is the ~40-minute step and the one most likely to meet a refusal or a Jetsam kill, and today a crash in its last block costs the whole step even though every finished block's records are already on disk and bit-exact.
- Pros: the records are per-(shape, draw, seed) and rng-self-contained by construction (that is what child isolation bought, and `test_real_child_grid_iteration_round_trips_bit_exactly` proves the boundary), so resuming block-wise splices nothing that was measured under different conditions; the checkpoint write already happens.
- Cons: it breaks the "STEP granularity only" rule the first amendment pre-registered, which exists because a mid-step restart is unsound wherever the rng stream crosses iterations - so the change needs the argument written down that the grid's stream does NOT cross blocks, plus a test that a resumed block-wise run produces records identical to an uninterrupted one, and it needs coordinator sign-off since it edits a pre-registered rule.
- Context: found in the 2026-08-15 merge review of branch phase0, alongside the fingerprint and lock defects that were fixed in that pass; the fingerprint now carries code identity, so a block-wise resume can no longer splice across a code change either.
- Depends on / blocked by: nothing technical; needs the pre-registration amendment ruled with the coordinator before implementation.

## The JSON writers would silently coerce numpy integers and booleans

- What: `write_checkpoint`, `write_child_result` and the results write in `bench/calibrate_quant_serving.py` all pass `default=float`, which is exact for the float64/float32/float16 values records carry today but would turn an `np.integer` into a float and an `np.bool_` into 1.0/0.0 the moment a record grows such a field.
  The work is a `default=` that dispatches on type (integer to int, bool to bool, floating to float) and raises on anything else, plus a test that each numpy scalar type crosses the boundary as its own kind.
- Why: the continuity anchor compares records EXACTLY, and a batch index arriving back as 16.0 or a `heldout_draw` flag as 0.0 would be an equality failure with no visible cause - or worse, a silent one on the `float(a) != float(b)` comparisons the anchor uses.
- Pros: small and local (one function, three call sites); removes a whole class of future boundary bug from the one serialization path the isolation amendment depends on.
- Cons: pure prophylaxis today - every field that crosses the boundary is currently a float, a str, a bool or a Python int, and `test_child_records_round_trip_bit_exactly` covers those; a raise on an unknown type could stop a run that would otherwise have limped.
- Context: raised in the 2026-08-15 merge review of branch phase0 as a latent defect, explicitly not a live one; the review's own framing was "if records ever grow to include them".
- Depends on / blocked by: nothing; do it whenever a record gains a non-float numeric field.

## A parent killed mid-child orphans that child

- What: `spawn_measurement` runs each measurement iteration with `subprocess.run`, so a SIGKILL to the parent (Jetsam, or a coordinator stopping a lane) leaves the child running with the GPU and its whole footprint, and the machine-wide measurement lock is released by the dead parent while the orphan keeps measuring.
  The work is to bind the child's lifetime to the parent's: a new process group plus a kill on parent exit, or the child watching for its parent's death (`getppid()` change) between records.
- Why: the failure mode is the one this harness was hardened against - two large Pythons on a 36 GB machine - reached from the opposite direction, and it defeats the lock rather than the budget: the next run acquires the freed lock and starts measuring beside the orphan.
- Pros: the child already checks its own budget between records, so a parent-death check has an obvious place to live and costs nothing; the isolation design is otherwise complete.
- Cons: process-group signalling has its own edge cases (a child that spawns nothing is easy, but the kill must not race a normal exit), and the window is small - the parent does almost nothing while a child runs.
- Context: raised in the 2026-08-15 merge review of branch phase0; nothing in the three Jetsam kills is known to have hit it, and it stays a hypothesis about a kill landing on the parent rather than the child.
- Depends on / blocked by: nothing.

## Extend the memory budget + lock pattern to the pricing and A/B harnesses

- What: apply the survival pattern that `bench/calibrate_quant_serving.py` now carries to `bench/price_qmv_boundary.py` (the pack boundary-pricing harness) and the four-arm `bench/serve_sub4bit.py` (the serving A/B harness): the phys_footprint budget guard with its distinct refusal exit (ctypes `proc_pid_rusage`, since Jetsam kills on footprint while RSS under-reads it by an order of magnitude), `machine_state.MeasurementLock` (the ONE machine-wide lock, imported not copied - a per-harness lock is what let the 03:29 collapse happen), and the `kern.memorystatus_level` available-memory refusal before each large cell.
- Why: the night of 2026-08-15 produced three Jetsam kills from the serving calibration (66.7, 69.4 and 39.5 GB Python footprints on a 36 GB machine), and the 03:29 event was a two-harness collapse: the 39.5 GB serving run plus a 27.9 GB pricing-probe Python whose idle gate co-fired; any large-footprint harness without the budget and lock can reproduce that collapse.
- Pros: the pattern is already built, tested and measured in `calibrate_quant_serving.py` (budget guard, `machine_state.MeasurementLock`, `require_available_memory`, and the refusal-exit vocabulary), so the extension is mostly wiring; the runner-level buffer-pool fix in `kernelverify/runners/device.py` already protects every harness that dispatches through `CompiledKernel.run`, which removes the largest single leak class fleet-wide.
- Cons: the two harnesses live on other lanes' branches, so the wiring belongs to whoever next touches those files; each one passes its own name to `MeasurementLock` for the refusal message, but they all take the same lock.
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
