# TODOS

## The selection rule can return a candidate below the shipping floor

- What: fix `select_first_operation` so section 4.3's 1.10 floor filters the candidate set BEFORE the tie band and the tie-breaks run, per Amendment 5 clause 24.
- Why: the floor is currently tested only against the largest gain, and the winner is then chosen from inside the tie band by peak footprint, so a candidate whose own arithmetic says it cannot reach R10's shipping floor can still be returned as SELECTED.
  Reproduced by execution on 2026-08-20, not by reading: a leader at gain 1.1012 with a tie-band member at 1.0812 returns the member, verdict SELECTED, at 1.0812 against a floor of 1.10.
- Pros: the fix is a reordering inside one function and the registered text for it is already committed in Amendment 5.
- Cons: it changes the outcome of a rule that is pre-registered, so the amendment had to land first, which it now has.
- Context: `bench/profile_rules.py` `select_first_operation`, the `if largest < GAIN_FLOOR` branch and the tie band below it; found by an independent Codex audit of the knob plan.
- Depends on / blocked by: nothing; it is step 7 of the current increment and is worth doing even if that increment stalls.

## The shared interleaver takes exactly two arms and the profile needs n

- What: generalise `interleave.interleaved_samples` from two arms to n, keeping the drift canary and the per-round structure, so a four-point dial ladder against a four-point floor ladder can be interleaved within a round.
- Why: Amendment 5 credits `ratio_lo` from a distribution of per-round slopes, and a slope needs every setting of the dial measured inside the same round.
  With a two-arm sampler the settings would be measured in separate passes and any drift between them would enter both fitted slopes.
- Pros: the sampler is already the single shared one, so the generalisation lands in one place.
- Cons: it touches a function every existing pricing harness calls, so the two-arm callers must be proven behaviour-neutral rather than assumed to be.
- Context: `bench/interleave.py`, the `interleaved_samples(build_a, build_b, rounds, guard)` signature; ADR 0004's two-halves mistake is why this is generalised rather than copied.
- Depends on / blocked by: nothing; step 8 of the current increment.

## The end-to-end harness and the profile now measure different widths

- What: amend section 8 so the end-to-end measurement runs at the same widths Amendment 5 registered for the profile, or state in writing why it should not.
- Why: `bench/train_lora_e2e.py` registers `max_seq_length=2048` and mlx-lm never trains at a fixed length, because its iterator pads each batch only to one plus the next multiple of 32 above that batch's own longest row.
  Amendment 5 resolved this for the PROFILE by registering two derived UltraChat bands and retiring the Dolly slice from it, and it explicitly did not reach section 8.
  So the operation the rule selects would be chosen at one width and the shipping number measured at another, and the width dependence is measured rather than argued: attention is 0.042 of the step at width 97 and 0.161 at width 769, because its score matrix is quadratic in the sequence length while the head, the projections and the loss are linear in it.
- Pros: the corpus tooling, the band arithmetic and the two pinned bands will already exist once the profile's increment lands, so this is a re-registration rather than new machinery.
- Cons: section 8 is committed pre-registration and its fairness conditions reference the dataset directly, so the amendment has to restate them rather than point at the profile's.
- Context: `bench/train_lora_e2e.py`; Amendment 5 clause 14 names this gap in writing and clause 20 registers what the profile uses instead; `bench/pin_dolly.py --report` reproduces the distribution on CPU in about a minute.
- Depends on / blocked by: the profile's corpus step, which produces the two bands this would adopt.

## Four test modules build their own Metal device instead of sharing conftest's probe

- What: convert `tests/test_device_buffer_pool.py`, `tests/test_serving_adequacy.py`, `tests/test_serving_survival.py` and `tests/test_quant_device_members.py` from their local `try: _DEVICE = MetalDevice() / except RuntimeError` blocks and local `needs_metal` skipif markers to `conftest.METAL_DEVICE` and `conftest.requires_metal`, which is what AGENTS.md now tells a new test to use.
- Why: each local block spawns its own device probe at import, so a run pays for five probes where one would do, and each local `needs_metal` skips correctly while never attaching the `gpu` marker - the exact drift that let seven dispatching tests sit inside the safe subset until the 2026-08-17 merge gate measured them.
- Pros: one probe per run instead of five, one definition of what a Metal test is, and AGENTS.md's gating rule becomes true of the tree rather than aspirational.
- Cons: 34 call sites across four files, and `test_serving_survival.py` and `test_serving_adequacy.py` use their local `_DEVICE` object for more than gating, so the conversion is not a pure find-and-replace and wants its own green run.
- Context: `tests/conftest.py:13-16` states the one-probe rule; AGENTS.md's marker section names these four as predating it; surfaced by the merge gate's standards axis on 2026-08-17.
- Depends on / blocked by: nothing; deliberately kept off `lane/test-infra-and-record` because these four modules gate correctly today - an in-process `MetalDevice()` raises rather than aborting - so this is consolidation, not a fix.

## A single cold first round can make a low-ceiling cell unreadable

- What: give the noise floor resistance to one outlier, or make the warm-up reliably prevent one, and re-register any cell whose ceiling is the same order as a cold round before measuring it.
- Why: `machine_state.spread_pct` is `(max - min) / median` over `ROUNDS = 5`, which a single sample dominates completely.
  Across the two binding speculative-decode runs of 2026-08-18, a first round more than 1% colder than the rest of its arm-cell appeared in 4 of 50 and 3 of 50 arm-cells, landing on a different cell each time and reaching -5.27% at worst.
  Wherever it lands, that cell's floor becomes the size of the artefact.
  It cost the K = 4 follow-up its reading: the composed and kernel point estimates reproduced to within 0.05 points, and the cell still read `not-a-decider` because its 4.16% ceiling could not clear a 5.32% floor built from one sample.
- Pros: cells whose ceiling is a few percent become readable at all, which is every cell at the small-K end of the speculative grid; and the fix is bounded, since `spread_pct` has one definition and one home.
- Cons: changing what `spread_pct` means moves `MAX_SPREAD_PCT` and every harness that reads it, including the published A/B, so it is a change to a shared registered quantity and needs its own pre-registration rather than an edit; raising `ROUNDS` instead costs run time linearly and does not by itself make `(max - min)` resistant to anything.
- Context: `docs/research/2026-08-18-spec-decode-k4-followup.md` sections 7 and 8; `bench/machine_state.py` `spread_pct`; the per-arm rounds at 0.6B K = 4 were 69.71, 73.44, 73.62, 73.56, 73.62.
- Depends on / blocked by: nothing, but it must not be done as a way of re-reading the K = 4 cell, which section 8 of that document closes.

## Batched decode is measured on equal-length prompts admitted all at once

- What: measure the engine's ragged path, prompts of different lengths admitted together, which exercises the right-padded prefill and the left-padded decode cache that the equal-length grid never touches.
- Why: `docs/research/2026-08-18-batch-decode-e2e.md` section 3 gives every stream the same 512-token window so that the width of every decode pass is exactly B, which is what makes the exact routed count and the decider reading possible; a real queue does not do that, and `BatchKVCache.finalize` rolls the cache into a left-padded layout whose masks this run never sees.
- Pros: it is the nearest question to the one this lane answers, and the engine already implements the whole padding path, so the work is a pre-registration and a harness variant rather than new machinery.
- Cons: with ragged lengths the decode width is still B but the per-stream cache depths differ, so the per-stream throughput a reader wants is no longer the aggregate divided by B, and the metric needs registering before the run rather than after.
- Context: `mlx_lm/generate.py` `PromptProcessingBatch.prompt` right-pads and `BatchKVCache.finalize` rolls to left padding; section 7 of the batched pre-registration excludes this explicitly.
- Depends on / blocked by: the batched grid reading out first, so there is a fixed-width number to compare against.

## The server's default admission decodes a burst above eight streams at width eight

- What: measure the engine as `mlx_lm.server` actually runs it, with requests arriving over time and `prefill_batch_size` at its default of 8, and read what the kernel does when the width is whatever the scheduler happens to produce.
- Why: found while pre-registering the batched grid and reproduced from the engine's scheduler on 2026-08-18: `BatchGenerator._next` admits at most `prefill_batch_size` prompts per turn, so a burst of more than 8 streams has its first 8 admitted and decoded at width 8 for at least one step before the rest join, and width 8 is inside the routed window.
  The batched grid registers `prefill_batch_size = 16` precisely so that every decode call has full width B, which means it measures the engine but not the server's own admission.
- Pros: it is the only remaining question between "the kernel helps at a fixed width" and "the kernel helps a person running a server", and the finding above says the routed window is reached under the default without anyone arranging it.
- Cons: with arrivals and turnover the width varies within a run, so there is no fixed M, no exact routed-call expectation and no per-cell ceiling; the honesty instruments this repository relies on do not survive the move, and replacements have to be designed rather than reused.
- Context: `mlx_lm/generate.py` `BatchGenerator._next` and its `prefill_batch_size` default; `mlx_lm/server.py` `--decode-concurrency` 32 and `--prompt-concurrency` 8; section 7 of `docs/research/2026-08-18-batch-decode-e2e.md` excludes it.
- Depends on / blocked by: nothing technical, but it must not be read as a re-run of the batched grid; it is a different question with different instruments.

## Speculative decode is measured only under greedy sampling

- What: extend the e2e pre-registration to non-greedy sampling, with its own identity rule.
- Why: the 2026-08-18 run fixed `temp = 0.0` for every arm so that token sequences are comparable and `kernel-diverged` means something; a real user is often not greedy, and mlx_lm's speculative acceptance test changes shape once sampling is stochastic.
- Pros: the claim would cover the way the model is usually run rather than one corner of it.
- Cons: it is a different experiment, not a parameter sweep - the acceptance rule changes, and the two-way token identity that keeps the current run honest cannot be reused as written, so it needs a fresh pre-registration.
- Context: `docs/research/2026-08-18-spec-decode-e2e.md` sections 3, 4 and 7.
- Depends on / blocked by: nothing.

## Speculative decode is measured on one prompt of 64 tokens

- What: repeat the e2e grid over a second prompt family and a longer generation than 128 tokens.
- Why: acceptance rate is a property of the prompt as much as of the draft model, and every number in the 2026-08-18 run rests on one 64-token prompt from one fixed seed; section 7 of that document says so explicitly and claims nothing beyond it.
- Pros: says whether the acceptance rates, and therefore the whole O1 and O3 picture, are a property of this stack or of this prompt.
- Cons: multiplies the grid's runtime by the number of prompts, and a prompt family chosen after seeing the first result is its own selection problem, so the family wants registering in advance.
- Context: `docs/research/2026-08-18-spec-decode-e2e.md` section 3 (`PROMPT_T = 64`, `GEN_TOKENS = 128`) and section 7.
- Depends on / blocked by: nothing.

## An fp16 cell at a batch the eligibility table never ruled on reads as in-contract

- What: make `kernelverify/schemas/heldout_eligibility.py` refuse loudly when a record asks about an (heldout, batch, dtype) cell at float16 and batch > 1 that carries NO ruling, instead of returning "eligible" by absence; today the table keys the B16 exclusion on batch 16 exactly, while its own docstring says MLX dispatches `affine_qmm_t` at batch 12 and above.
- Why: the grid's batches are 1/2/8/16, so nothing reaches the gap today, but the serving harness's own batch-regime probe put the arithmetic split at 11/12 for three shapes; the day a grid adds batch 12 or 14, a structurally out-of-contract MLX kernel would read as an admissible held-out and could fire a DEMAND MISS that ADR 0014 already ruled is not a membership gap.
- Pros: absence of a ruling becomes a refusal rather than a silent "eligible", which is the same fail-loud discipline the kernel-source hash guard already applies in that module.
- Cons: widening the key from 16 to ">= 12" instead would be the wrong fix and must NOT be done casually - ADR 0014's D2 bar requires a verified source reading showing a narrower intermediate on THAT cell's dispatch path, and only batch 1 and batch 16 have one; the refusal is the honest shape until someone reads the source for 12-15.
- Context: `heldout_eligibility.py` lines 88-102 (the EXCLUSIONS dict) and its module docstring lines 8-10; ADR 0014's "the evidence bar is structural" section; surfaced by the merge gate's spec axis on 2026-08-16.
- Depends on / blocked by: nothing.

## Two corrections from the withdrawn companion-member plan never landed

- What: (a) `kernelverify/schemas/quant_device.py`'s module docstring still says the device member has "the shape of MLX's own quantized matvec" - MLX chunks its accumulation at 8 elements while the member chains at 64, so the shapes differ in the one property the member exists to model; (b) record the emulation-fidelity note about ascending-butterfly ordering beside it.
- Why: the plan that proposed the companion member was withdrawn, but it carried two corrections that were independent of the withdrawn idea and were meant to land regardless; a docstring that overstates the fidelity of a calibration instrument is how a reader concludes the member models something it does not.
- Pros: two prose fixes, no behaviour change, no re-measurement.
- Cons: none.
- Context: the withdrawn plan is `~/.gstack/projects/kernelverify/vlad-companion-member-20260815.md` (tasks T3 and its notes); surfaced by the merge gate's spec axis on 2026-08-16.
- Depends on / blocked by: nothing.

## The drawn-sample ensemble and the Accelerate version are unbound

- What: (a) rebuild the quantized ensemble as a DRAWN SAMPLE of the admissible-implementation contract rather than a hand-written dict of six functions, the way `kernelverify/tolerance/floor.py` derives its ensemble as a prefix of the generated contract population; (b) pin the Accelerate/BLAS version the fp64 references are computed against, and record it in the evidence headers.
- Why: the verification-design audit found the quant ensemble is not drawn from any contract module - `kernelverify/tolerance/contract.py` has zero mentions of quantization - so "the floor represents the admissible class" rests on six hand-picked functions; and a macOS or numpy update that changes Accelerate's reduction order would move every committed fp64 reference with nothing recording that it happened.
- Pros: (a) makes the floor a sample of a stated population instead of a curated list, which is the repo's own standard everywhere else; (b) makes an environment-driven drift loud instead of silent.
- Cons: (a) is a contract-population design block with its own ADR, not a refactor - the quantized contract's clauses do not exist in code yet; (b) needs a decision about what to do when the pin no longer matches (refuse, or re-derive and record).
- Context: audit of 2026-08-15 (verification design, questions 1 and 6); `quant_contract.ENSEMBLE` lines 214-221 vs `tolerance/floor.py`; surfaced again by the merge gate's spec axis.
- Depends on / blocked by: the int-domain membership decision above, which touches the same population.

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
- Context: `llama-bench`, the binary the producer wraps via `-o json`, has no parallel-sequence mode at the pinned commit, and that half of this entry still holds (re-verified 2026-08-18 at commit a94d563: its own argument parser never reaches the generic `-np/--parallel`, and its only batch knobs are `-b/--batch-size` and `-ub/--ubatch-size`, which chunk one sequence's prompt rather than running independent streams).
  The honest path is still `llama-batched-bench`, which runs `-npl` parallel decode streams, and the binary is already built at `/Users/vlad/llama.cpp/build/bin/`.
  CORRECTION, 2026-08-18: this entry said that binary "emits a text table rather than JSON, so it needs a new sample runner plus a dedicated output parser".
  At the pinned commit it accepts `--output-format jsonl` and prints one JSON object per (pp, tg, pl) cell, so the parser is `json.loads` per line and the cost of this item is lower than the entry claimed.
  Two caveats survive the correction: it is JSON Lines rather than the JSON array `llama-bench -o json` produces, so `bench/external.py`'s reader does not fit unchanged; and it has no `-r/--repetitions`, so repeats come from re-invocation and every sample pays the process start and model load.
  It also has a `-tgs` flag that decodes the sequences one after another at the same `-npl`, which is a serialised control arm from the same binary and the same clock.
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

## The held-out draw differs from calibration by seed only

- What: pre-register and run a held-out draw for the device-arithmetic K derivation that varies SHAPE (the six Qwen3-4B serving shapes, not the three calibration shapes) and WEIGHT DISTRIBUTION (a draw the calibration never saw), so G1 on the held-out draw can actually disagree with calibration.
- Why: the 2026-08-16 amendment (ruling 1A) made G1 falsifiable in principle - K is set from calibration seeds 0/1 and tested on seeds 100/101 - but both draws use the same three shapes and the same two weight distributions, so the test is honest in the record and weak in practice; the outside voice of the plan review named it.
- Pros: the serving records already exist as a natural distinct draw, so half the design is done; a real held-out disagreement is the first evidence the K rule could ever produce against itself.
- Cons: a new measurement design needs its own pre-registration and ADR, and a serving-shape draw costs the ~45-minute cold grid rather than the 7-minute device grid.
- Context: `bench/calibrate_quant_device.py` docstring amendment of 2026-08-16 and ADR 0017; `choose_k` takes per-width `(cal, indep)` record lists, so a different held-out source is a change to what `indep` holds, not to the rule.
- Depends on / blocked by: ADR 0017 (the M1 re-run under the amendment) landing first, so the two changes are never mixed in one reading.

## Optimiser: first spike plan

- What: turn section 7 of `docs/research/2026-08-16-faster-honest-decode-kernels-literature.md` into a lane plan for the first generate-gate-keep spike, in the order that section fixes: the candidate store first, then one hand-written candidate through the whole loop, then the generator.
  The three techniques the review ranked, their per-technique spikes and their pre-registered go/no-go outcomes are in sections 5 and 6 of the same file and are binding on that plan.
- Why: the literature review found that the optimiser's downstream half already exists (the gate, the pricing protocol, the certificate emitter, the evidence-derived routing) and that its upstream half is three named absences (the generator, the search space, the candidate store), so the first spike is a wiring exercise with one missing part rather than a build.
  The review also found the ranking's headline: exactly one published sub-4-bit GEMV design both reads fewer bytes per weight than MLX affine 3-bit and keeps an fp32 total, and its speed comes from a tensor-core instruction ADR 0007 records as unavailable on this machine, so the honest version of these packings is unpublished for any backend and is the product opening.
- Pros: every spike in section 5 is one shape at M = 1 under protocols this repo already runs, so each costs one short measurement slot rather than a build; the go/no-go for each is pre-registered before any of them runs, including which of the three outcomes each technique is eligible for.
- Cons: one of the three techniques (QTIP with incoherence processing) is ruled NEW at the operator level, so its gate result may not be read until a membership block lands under ADR 0012's membership-before-K order, which is a contract renegotiation and not a spike; the review's own recommendation is to spike it without incoherence processing first, which keeps it inside a class the contract already has members for.
- Context: `docs/research/2026-08-16-faster-honest-decode-kernels-literature.md` sections 5, 6 and 7; the arithmetic classes come from `kernelverify/schemas/quant_contract.py::ENSEMBLE` and `kernelverify/schemas/quant_device.py`; the measured facts the review builds on are ADR 0007 (bandwidth ceiling, 0.99 ms/token, the q3_K deficit), ADR 0014 (MLX's fp16-internal batch-1 kernel), ADR 0015 (our 3-bit kernel loses at M = 1 and wins at M = 5 to 9) and ADR 0016 (what a missing or unrealistic ensemble member costs).
- Depends on / blocked by: M2, the end-to-end four-arm A/B, which pins the batch-1 per-stream baseline every spike is ultimately measured against.

## kv_attention's K is borrowed; nothing has ever calibrated one over KV_MEMBERS

- What: build a pre-registered adequacy harness for kv_attention modelled on `bench/calibrate_quant_bits.py` - its rule in the module docstring, committed before any measurement - that derives K over `KV_MEMBERS` on a grid of (B, H, T, DH, BITS) and both activation dtypes, with a held-out implementation and an independent draw, and reports both the calibration and the independent demand.
- Why: `kernelverify/schemas/native_ops.py::kv_tolerance` divides by an ensemble floor and multiplies by `K_QUANT = 4.0`, a value ADR 0012 derived for quantized_matmul over quantized_matmul's nine-name ensemble on quantized_matmul's grid; ADR 0008's claim that each family ships "its own calibrated K" is false for this operator, and every kv_attention certificate has been resting on the reuse.
- Pros: CPU-only - the four members and the fp64 reference are all numpy, so this needs no GPU slot and no coordinated quiet window, unlike every other calibration this repo has queued; the grid, the leave-one-out rule, the K grid and the gate vocabulary all already exist and are copied, not invented.
- Cons: the answer may be that 4.0 is wrong in either direction, and a K that moves obliges re-emission of the kv certificates and a bump of a kv ensemble version; the operator's floor also carries a softmax, so the zero-variance regimes that bind every other tolerance decision here may not be the binding ones and the grid needs its own thought.
- Context: `native_ops.py::kv_tolerance` (its docstring states the borrow), `KV_MEMBERS` beside it, `bench/emit_pack_certificates.py::contract_version_for` (the certificate prose that now says "K borrowed from quantized_matmul, uncalibrated"); surfaced by the verification-design audit of 2026-08-15 and labelled by task V2 of the 2026-08-16 amendments plan.
- Depends on / blocked by: nothing technical; it is a CPU harness that can run beside any GPU measurement.

## The closing idle check discards a whole run on one sample

- What: decide what evidence the end of a run needs, and give `check_idle_after` that instead of a single `idle_check()` snapshot; a streak like the opening gate's, a mid-run sampler that records when the window broke, or a rule that separates a blip from a busy machine.
- Why: the opening gate requires five clean samples thirty seconds apart before a run may start, and the closing check requires one, so the two ends of the same run are held to different standards.
  Measured on 2026-08-18 and 2026-08-19: two batched-decode runs of forty-four minutes each completed all ten cells with zero diverged rounds, zero hard fallbacks and exact routed counts at every cell, and both were discarded by their closing sample, the first on `airportd at 46% CPU` and the second on `WindowServer at 15% CPU` against a threshold of 15.
  The two runs agreed to within 0.4 points on every cell, so what was thrown away was not a doubtful measurement.
- Pros: a run is the expensive thing here and a sample is the cheap thing, so the asymmetry is backwards; and the closing check is one function with one caller pattern, so whatever replaces it lands in one place.
- Cons: the closing sample is the only evidence the harness has that the quiet window held for the whole run, and a weaker rule admits a run whose middle was disturbed, which is the failure the check exists to catch; a mid-run sampler is new machinery on the honesty path and has to decide what a disturbed round means before it can annotate one.
- Context: `bench/machine_state.idle_check` samples load averages, `ps -Ao pcpu` competitors above `BUSY_PROCESS_PCT = 15.0`, and power state, all at one instant; `bench/serve_sub4bit.check_idle_after` returns 1 when that instant is dirty; the detached runner's opening streak is five clean samples.
- Depends on / blocked by: nothing technical, but it must not be done while a run of this lane is waiting to bind, because changing the rule that judges a measurement after seeing the measurement is what the pre-registration discipline exists to prevent.

## Two tolerance-free gates are blocked on a ruling, not on work

- What: decide whether the compiler loop declares, as its own policy rather than as a contract amendment, that it will only KEEP kernels that are deterministic and that write only inside their declared extent; then build the guard-rows and determinism gates against that policy.
- Why: five gate designs were adversarially reviewed on 2026-08-19 and all five were broken, three of them because they refuse kernels the tolerance contract admits.
  Amendment 1 of the sprint pre-registration resolved three of those by abstention, and two have no abstention escape, because abstaining on the very property they test leaves them testing nothing.
  A determinism gate in particular would contradict `kernelverify/report/certificate.py`, which already states in its protocol block that output values are not asserted identical.
- Pros: the funnel gets two more cheap screens before the expensive tolerance gate, and the loop stops generating kernels it would never keep.
- Cons: the policy narrows what the loop accepts below what the verifier admits, which must be said plainly wherever a kept kernel is quoted, and it excludes formulations that are standard on Metal (cross-threadgroup split-K can only be assembled through the output buffer, and simdgroup stores have no partial-store variant).
- Context: prereg amendment 1; the ruling was put to Vlad on 2026-08-19 as options A (amend the contract), B (funnel policy, recommended) and C (build neither).
- Depends on / blocked by: Vlad's ruling. No code is blocked otherwise.

## metalrunner is not installable, and the distribution shape is undecided

- What: decide whether `pip install metalrunner` ships one distribution containing the parts of kernelverify it needs at runtime, or two distributions with metalrunner depending on kernelverify; then add the packaging and a fresh-venv install test.
- Why: R7 promises `pip install metalrunner`, and today the package runs only from a clone.
  `pyproject.toml` states in its own words that this project is deliberately not packaged, so adding a `[project]` table changes a documented fact rather than adding a field.
- Pros: the product becomes installable by someone who is not looking at this repository, which is the whole of "operational on a MacBook".
- Cons: a user needs the routing tables, the certified kernel sources and the pack dispatch at runtime, but not the mutation catalogue, the vendored corpus or the bench harnesses, so a naive one-distribution answer ships hundreds of megabytes of verification apparatus to every user.
- Context: `metalrunner/` landed 2026-08-19 and works from the tree, with a real two-iteration fine-tune passing; the sprint plan puts the wheel on Day 6.
- Depends on / blocked by: nothing technical; it wants one decision about scope.

## The support gate cannot screen the scales and biases bindings

- What: add a target that scales one quantization group by a power of two, which leaves the integer codes bit-identical while moving the scale and the bias together, and extend the gate's attribution rule to accept a declared two-binding extent.
- Why: `kernelverify/compiler/support.py` derives its device inputs from raw weights through the contract quantizer, and that oracle has no single-binding perturbation of a scale: editing a scale alone leaves the reference bit-identical, so the gate abstains forever.
  Two of the four data bindings of the only kernel in the pack are therefore unscreened, which the policy string names honestly but does not fix.
- Pros: closes the wrong-group-index and stale-scale fault classes on the bindings that currently have no coverage from any gate.
- Cons: it is the first target that perturbs two bindings at once, so the attribution rule that currently reads "every binding byte-identical except one" has to be generalised without becoming a rule that admits an unattributable edit.
- Context: measured and reported by the contract-lens refutation of 2026-08-19; recorded in prereg amendment 2 as the named extension.
- Depends on / blocked by: nothing.

## The serving-reinterpretation guard invalidates its artefact on a comment change

- What: narrow what `bench/reinterpret_serving_adequacy.py`'s `interpretation_identity()` hashes, so that a change which cannot move a derived number does not invalidate the committed artefact.
- Why: it hashes whole files including `kernelverify/schemas/native_ops.py`, so adding an unrelated dict entry and three comment lines turned `tests/test_serving_reinterpretation.py` red on 2026-08-19.
  Regenerating showed exactly one leaf differing, the hash itself, with every derived number bit-identical, which is the guard doing its job at a granularity coarser than the drift it exists to catch.
- Pros: the guard keeps its meaning while stopping unrelated edits from costing a regeneration and a commit.
- Cons: any narrowing is a judgement about which bytes can move a number, and getting it wrong reintroduces exactly the silent drift the header was added to refuse, so a coarse-and-annoying guard is safer than a clever one.
- Context: surfaced while adding the bfloat16 contract entry (ADR 0019); the artefact was regenerated and the invariance check confirmed no number moved.
- Depends on / blocked by: nothing; deliberately not fixed on the way past, per the do-not-touch-unrelated-code rule.

## mlx-lm's `run(args, training_callback=...)` silently ignores its callback

- What: report upstream that `mlx_lm.lora.run` declares a `training_callback` parameter and overwrites it on its own next statement with `get_reporting_callbacks(args.report_to, ...)`, so the argument can never reach the trainer; propose that it fall back to the passed callback when `--report-to` is unset, or chain the two the way the callbacks themselves chain.
- Why: it is a documented parameter whose only possible effect is to look correct.
  Anything driving mlx-lm's trainer from outside and passing a callback gets no reports and no error, which is the failure shape hardest to notice.
  metalrunner hit exactly this and now injects at `train_model` instead, which works and is strictly more invasive than the public parameter would have been.
- Pros: the fix upstream would let metalrunner drop one seam entirely, and the mechanism reduces to passing an argument.
- Cons: it is someone else's release schedule, so the seam has to exist either way; a fix would also change the pinned file hash and therefore the stack pin.
- Context: reproduced against mlx-lm 0.31.3 on 2026-08-19, `mlx_lm/lora.py` `run()`; the seam that works around it is `metalrunner/lora.py` `TRAIN_MODEL`.
- Depends on / blocked by: nothing; it is a PR to write, and the sprint does not wait on it.
