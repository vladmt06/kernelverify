# TODOS

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

## Simplify-pass structural deferrals (2026-08-15 review, four-angle)

- What: move each pack family's verify() and family facts (doors, clauses, contract version, tolerance model, domain) out of `bench/emit_pack_certificates.py`'s seven family-keyed dispatch tables into per-family descriptors in `kernelverify/pack/`, iterated by a registry; and add a packaging seam (pyproject + editable install) so the 21 `sys.path.insert` preambles and the two loudly-guarded kv-kernels checkout pins die.
- Why: today a fourth pack kernel requires edits in five-plus emitter functions, and the path hacks let a main-repo script import lane-worktree code (now guarded with a provenance print, but the seam is the fix).
- Pros: the emitter stops knowing everything; adding a kernel becomes one descriptor; imports become boring.
- Cons: touches the certificate emitter's surroundings, so it wants its own review and a re-emission check.
- Context: findings 2 and 6 of the altitude review during the 2026-08-15 simplify pass; the corpus-primitives package lift (`kernelverify/corpus/` absorbing the helpers product code path-hacks out of frozen `measure_escape.py`) belongs to the same seam family and was also deferred.
- Depends on / blocked by: pack lane merge (certificate re-emission machinery must be quiet when this lands).

## Simplify-pass small mop-ups

- What: extract `bench/mlx_probes/_common.py` for the tripled probe helpers; inline `make_run_case`/`result_outputs` pass-throughs in `kernelverify/schemas/quant_device.py` (call sites live in `bench/calibrate_quant_device.py`, wrapper test goes with them); adopt `kernelverify.battery.core.fingerprinted_pickle_cache` in `bench/calibrate_k.py` for `contract_k.pkl`; dedup the verify-then-bench `main()` scaffold tripled across the pack gate scripts (stdout must stay byte-identical).
- Why: each was verified duplicated or dead by the 2026-08-15 four-angle review but sat across territory or stdout-risk lines that made it not worth forcing that night.
- Pros: finishes the dedup story; nothing else.
- Cons: none of substance; four small diffs.
- Context: reuse findings 8 and 10, simplification finding 11, and the package agent's follow-up note from the simplify pass.
- Depends on / blocked by: nothing; bundle into any future touch of those files.

## 4-bit pricing run for the wide-qmv routed windows

- What: run `bench/price_qmv_boundary.py` with `BITS = 4` over the same six Qwen3-4B decode dispatch shapes and the same M = 1..16 sweep, then derive its 4-bit windows into `kernelverify/pack/routed_windows.py` beside the 3-bit ones.
  The run needs a coordinated quiet slot like every other heavy harness, and it is queued for the next free measurement window rather than jumping the current one.
- Why: `should_dispatch` is bit-keyed as of the per-shape routing change, and 4-bit has no recording, so it routes NOWHERE and MLX serves every 4-bit shape (ruled D1, 2026-08-15).
  That is the honest reading of no evidence, but it also means the pack sits idle at the width most quantized models actually ship at.
- Pros: the probe, the guard vocabulary, the per-shape derivation and the gate coverage all exist already, so the run is a parameter change plus a second recording; the widening rule it will be read under is pre-registered in the probe's docstring, dated before the run.
- Cons: it is another ~30-minute exclusive slot on the one machine, and 4-bit is the width where the old microbenchmarks measured the kernel closest to a wash below the window (1.00-1.02x), so the run may well justify no routing at all - which is a real outcome, not a wasted slot.
- Context: the 3-bit recording is `bench/results/qmv-boundary-pricing-2026-08-15.json`, committed, and its derivation is pinned by `tests/test_pack_routed_windows.py`; 2-bit is cut from the block entirely (D4) and is not part of this item.
- Depends on / blocked by: a free coordinated measurement window; nothing technical.

## An R sweep re-prices every routed window, not just the cells it moves

- What: whenever `rows_per_simdgroup` is re-measured and its register-wall thresholds move, re-run the boundary pricing and re-derive every window in `kernelverify/pack/routed_windows.py`, rather than editing the R function alone.
- Why: a routed window is a claim about one kernel body at one launch config, so a cell priced at R = 4 says nothing about the same cell launched at R = 2; the module pins the per-M R for exactly this reason and refuses to derive a table whose recording disagrees with the pinned launch config.
- Pros: the refusal is already wired, so a bare R change fails loudly at import instead of shipping windows that describe a kernel nobody ran; the re-derivation itself is mechanical once the new recording exists.
- Cons: it makes any R experiment cost a full pricing slot before it can ship, which is the intended price and not a defect.
- Context: the current pins are R = 4 up to M = 10 and R = 2 above it, matching every point of the 2026-08-15 recording; `KERNEL_SOURCE_SHA256` carries the same discipline for the kernel body itself, so editing the MSL has the same consequence.
- Depends on / blocked by: nothing; it is a standing rule that applies to whoever next moves R.
