# kernelverify agent instructions

These are the instructions specific to this repository.
Vlad's global instructions still apply; this file adds the project's layout, how it runs, its working rules, and the mistakes already encountered.

## What this project is

- kernelverify is a GPU-kernel correctness verifier.
  It exists because LLM-generated GPU kernels routinely pass the benchmarks that ship with them while being wrong, which the paper "The Correctness Illusion in LLM-Generated GPU Kernels" (arXiv 2606.20128) documented and whose gpuemu corpus we vendor.
- The strategy has two phases.
  First build the verifier, then build the optimiser that generates faster kernels for open-weight models (Llama, Qwen) with the verifier gating every candidate.
  Vlad's Mac (Metal backend) is a first-class optimiser target alongside server GPUs.
- The core reframing: escape is a property of the pair (bug, test case), not of a bug alone.
  The differentiator: test policies are scored by mutation against a synthesised fault population that provably contains the published corpus faults.
- The current battery covers three axes: shape and dtype, input scale, and structured input modes (opposed signs, near zero, constant rows).
  The shipped policy is boundary coverage of single features, then feature pairs, then random exploration; it catches 100% of the viable fault population at 16 evaluations per operator.

## Layout

- `kernelverify/reference/kernels.py` - parameterised kernels, correct by default; keyword seams introduce faults.
  Single source for both the corpus ports and the mutation catalogue, so the synthetic fault space provably contains the published faults.
- `kernelverify/mutation/catalogue.py` - the fault catalogue (45 entries), corpus faults marked `from_corpus`.
- `kernelverify/tolerance/contract.py` - the admissible-implementation contract: the class of kernels the verifier promises never to flag, plus the generator that samples it.
  Read the module docstring before changing any tolerance; the exclusions are what keep precision faults faults.
- `kernelverify/tolerance/floor.py` - the shipped conditioning-aware tolerance.
  Its ensemble is a prefix of the contract population, not a hand-written list.
- `bench/cpu_ports.py` - corpus kernel name to parameterisation mapping, plus each buggy kernel's correct control.
- `bench/measure_escape.py` - escape-rate measurement against the vendored corpus.
  Frozen: reruns must reproduce the ADR 0001 tables exactly.
- `bench/score_oracles.py` - mutation scoring of test policies at equal budget B.
- `bench/probe_bottleneck.py` - one-off probe of where a given fault is detectable; rerun it whenever the catalogue grows.
- `bench/calibrate_quant_bits.py` - per-bits ensemble adequacy for the quantization contract, with its pre-registered rule in the module docstring.
  Prints both the pre-amendment and post-amendment reading of every width, which ADR 0009 relies on; do not remove either.
- `bench/calibrate_quant_device.py` - the device-arithmetic membership calibration: trigger, membership-before-K repair order, K re-derivation, with its pre-registered rule in the module docstring.
  ADR 0012 was the reading of its first run; ADR 0016 is the reading of its 2026-08-15 re-run under the chained `factored-groups` member, and reruns must reproduce ADR 0016's tables (ADR 0012's no longer reproduce, by design).
  Its records are committed at `bench/results/quant_device_adequacy.json`; the harness overwrites its own cache on every run, so copy the records there after any run an ADR reads.
- `bench/calibrate_quant_serving.py` - 3-bit adequacy at the Qwen3-4B E2E serving shapes, with its pre-registered rule in the module docstring: continuity anchor against the ADR 0012 cache, per-shape G0, the batch-regime probe with direct-match coverage, per-cell K demand against the shipped K = 4, and gates under width-pooled fault equivalence.
  ADR 0013 is the reading of its run, and its interpretation follows ADR 0014 and ADR 0016 (the exit-1 refusal ADR 0013 recorded was the pre-ruling reading).
  Reruns reproduce every column bit-identically EXCEPT `factored-groups`, which ADR 0016 deliberately changed: that column moves and everything derived from it moves with it, by design, the same amendment ADR 0012's freeze line carries.
  Amended 2026-08-15 (fourth, ADR 0014): STEP 4/5 read under per-cell held-out eligibility from `kernelverify/schemas/heldout_eligibility.py` - out-of-contract cells are labelled with their numbers, never a DEMAND MISS; the miss branch fires on the admissible-only demand; the shipped-tolerance overshoot is printed beside `k_demand`, both-readings style.
  Amended 2026-08-15 after three SIGKILLs in step 2: per-shape and per-implementation progress lines carrying an RSS self-report, a per-step checkpoint at `bench/.cache/quant_serving_partial.json` (atomic write, `--resume` at step granularity only), and row-chunked dequantization so the lm_head weights stop paying a 3x whole-matrix transient.
  The amendment and its bit-equality proof are in the module docstring.
  Amended again the same night (memory-truthfulness): a phys_footprint budget with a distinct refusal exit (never a shrunk grid), a machine-global single-instance lock plus an available-memory gate, and child-process isolation per measurement iteration; the root leak was per-case Metal buffer allocation, fixed by the buffer pool in `kernelverify/runners/device.py`.
  The three Jetsam kills that forced this (66.7, 69.4, 39.5 GB footprints on the 36 GB machine) are dissected in the module docstring's second amendment; `tests/test_serving_survival.py` is its proof suite.
  Amended a third time (the merge review of that night's work, same docstring): the checkpoint fingerprint now carries a sha256 over the modules a record's value passes through, because the run resumed past STEP 0 - the pipeline's only end-to-end bit-exactness gate - on a checkpoint written by pre-buffer-pool code, and `--continuity-only` runs STEP 0 alone, which is the check to run after any change to the arithmetic path.
  The single-instance lock moved to `machine_state.MeasurementLock`: ONE `fcntl.flock` shared by every heavy harness, since a per-harness lock let the pricing probe run beside the calibration, which is the 03:29 collapse itself, and a pid file cannot be read without a stale-pid judgement that eventually steals a live holder's lock.
  The pinned 3-bit artifact lives at the absolute path `/Users/vlad/kernelverify/bench/.models/qwen3-4b-3bit-g64`; `bench/.models` is gitignored, so it exists in the main worktree only and never arrives via merge.
- `bench/calibrate_k.py` - measures what the admissible-implementation contract demands of K, and how many ensemble members it takes to represent that contract.
  Rerun it whenever the contract, the ensemble or the catalogue changes.
- `bench/roofline.py` and `bench/metal/roofline_probe.mm` - this machine's measured ceilings, GPU and CPU.
  Everything the optimiser claims is scored against these, so they are the denominators of the whole phase 2 story.
- `bench/baseline_llamacpp.py` - llama.cpp at a pinned master commit, placed on the roofline.
- `bench/baseline_kernels.py` - upstream's own per-kernel perf set, each case scored against the roofline at its own arithmetic intensity.
- `bench/gguf_info.py` - GGUF tensor table reader; supplies the flop and byte models those two scripts need.
- `bench/probe_baseline_gaps.py` - one-off probes that closed the three claims ADR 0007 first shipped as inferred; rerun it whenever the baseline moves.
- `bench/measure_baselines.py` - the one command that measures the machine's baselines across both stacks and appends them to `bench/.baselines/<date>.jsonl`.
  It alternates the arms within each workload cell (one sampling group per cell, schema v3), refuses to call a number binding on a busy or unplugged machine, and refuses sub-millisecond samples as absolute claims.
- `bench/machine_state.py` - the idle gate, the timing floor, and the one machine-wide measurement lock (`MeasurementLock`, an `fcntl.flock` on a fixed path) every heavy harness takes, with the reason each exists.
- `bench/detached_run.py` and `bench/start_binding_run.sh` - the detached run path: a one-shot launchd job that waits for a strong-idle window, then runs `measure_baselines.py` with no terminal attached, retrying up to three passes on dispersion (ADR 0010).
- `bench/OPERATOR-CARD.md` - the one-card instruction for starting a binding run and reading its outcome.
- `bench/mlx_info.py` - the MLX safetensors equivalent of `gguf_info`, so both stacks get modelled bytes rather than file size.
- `bench/.baselines/SCHEMA.md` - the row contract the per-chip matrix renderer consumes. The producer validates against it on every write.
- `bench/reinterpret_serving_adequacy.py` - the ADR 0014 reading of the committed ADR 0013 evidence: a pure CPU re-read (sha256-checked) that writes a separate derived artifact and refuses to write if any in-contract quantity drifts from the recorded tables.
- `kernelverify/schemas/heldout_eligibility.py` - which held-out implementations are admissible in which (batch, dtype) cells: versioned contract data, each exclusion hash-guarded against the MLX kernel source it was ruled on, failing loudly on mismatch so an MLX fix is never waved through on a stale label.
- `bench/results/` - the recorded baselines, the committed serving-adequacy evidence plus its derived reinterpretation, the device-grid records, and the qmv boundary pricing recording. ADR 0007 is the reading of the baselines, ADR 0013/0014/0016 of the serving records, ADR 0016 of the device grid, ADR 0015 of the pricing recording.
- `docs/adr/` - decisions with the measurements that forced them.
  Read these before changing any method.
- `vendor/gpuemu-corpus/` - vendored unmodified at the commit pinned in `vendor/PINNED.txt`.
  Never edit anything under `vendor/`.
- Empty packages (`battery`, `detectors`, `runners`, ...) are planned components, not dead code.

## Running

```
cd /Users/vlad/kernelverify
.venv/bin/python bench/measure_escape.py   # ~1 min, must reproduce ADR 0001
.venv/bin/python bench/score_oracles.py    # ~1 min warm, ~10 min after a catalogue or ensemble change
.venv/bin/python bench/calibrate_k.py --n-random 24   # instant warm, ~20 min cold, must reproduce ADR 0005
.venv/bin/python bench/calibrate_quant_bits.py        # ~3 min, must reproduce ADR 0009
.venv/bin/python bench/calibrate_quant_device.py      # ~7 min, needs the Metal GPU, must reproduce ADR 0016
.venv/bin/python bench/calibrate_quant_serving.py     # ~45 min, needs the Metal GPU and the pinned artifact, reproduces ADR 0013 except the factored-groups column (ADR 0016)
.venv/bin/python -u bench/derive_repaired_member_column.py   # ~70 min, CPU only, ~16 GB steady and 24.3 GB peak at lm_head; rewrites ADR 0016's committed column
```

- `derive_repaired_member_column.py` is derived, not measured: it re-reads the committed ADR 0013 records with the repaired `factored-groups` column recomputed, and every ADR 0016 number comes from its output.
  `tests/test_repaired_member_artifact.py` compares the committed header against the live code identity, so ANY edit to `quant_contract.py`, `phase0_contract_k.py`, `calibrate_quant_serving.py` or the generator itself obliges a re-run before that test is green again.
  The hash is deliberately whole-file rather than per-function: it can only over-fire, and over-firing costs an hour of compute and says so, while under-firing is the failure this repo has already had (a committed derived file that named code which had since moved twice, behind a check that compared the file to itself).

- Verdicts are cached at `bench/.cache/verdicts.pkl`, fingerprinted by mutation names, input modes, K and every ensemble member's label; any catalogue, mode or oracle change rebuilds automatically.
- Contract measurements are cached at `bench/.cache/contract_k.pkl` under the same discipline, fingerprinted by the contract version and sample size as well.
- Deleting `bench/.cache/` is the safe full reset.
- The environment needs `torch`, which only `gelu[variant=erf]` uses; a worktree venv created without it fails part-way through a verdict build.
- It also needs `mlx==0.32.0` and `mlx-lm==0.31.3`, pinned across worktrees so binding comparisons stay on one toolchain.
  Without mlx, the mlx-dependent test modules are skipped whole or fail to collect, so the suite under-reports badly.
  Skipped modules hide their contents rather than their count, so quote test counts from a fully equipped venv only; this branch passes 725 in 78 s as of 2026-08-16.
  Fully equipped means `pyobjc` as well as `mlx`: without the Metal bindings the runner-backed pack tests fail rather than skip, and a venv with mlx alone reports a number nobody should quote.

The machine baseline, in this order, because each step writes the denominators the next one divides by:

```
.venv/bin/python bench/roofline.py           # ~1 min, machine ceilings
.venv/bin/python bench/baseline_llamacpp.py  # ~6 min, llama.cpp on the roofline
.venv/bin/python bench/baseline_kernels.py   # ~8 min, per-kernel
```

Or, for the cross-stack matrix rows in one command:

```
.venv/bin/python bench/measure_baselines.py     # refuses unless idle and on AC
```

Or detached, so every interactive session can be closed first; this is the binding path, because the sessions are the contention (ADR 0010):

```
bench/start_binding_run.sh     # arms a launchd job, then quit Terminal
```

- External dependencies, deliberately outside the repo: a llama.cpp checkout at `/Users/vlad/llama.cpp` and GGUF models at `/Users/vlad/models/gguf`.
  These two paths are canonical across lanes as of 2026-08-14; `~/src/llama.cpp` is abandoned.
  The commit, build flags, model files and power state are recorded in the result JSON, so a rerun that disagrees can be diagnosed rather than argued about.
- Re-baseline whenever llama.cpp master moves, and record the commit; a speedup measured against a stale baseline is not a speedup.

## Working rules for this repo

- Every method change needs a measurement behind it and an ADR entry recording what forced it.
  Pre-register the next upgrade and adopt it only when a measured miss demands it, the way ADR 0002 pre-registered pairwise coverage and ADR 0003 adopted it.
- Never cite the corpus's `benchmark_verdict` fields as evidence; they are hardcoded "pass" and were never computed.
- Any 100% claim must be backed by exact miss counts, not by a rounded table cell.
- Grow the fault catalogue faster than the policy adapts; the numbers stay honest only while the population outpaces the tuning.
- Excluding a mutation as equivalent requires a stated reason for why no oracle could ever see it.
- Test policies must never reference a known fault; they may only use the operator schema, dtypes, and input modes.
- Never add AI co-author attribution to commits.
- Every worker-lane task is planned through the gstack head-engineering review (/plan-eng-review) before dispatch, Vlad's standing rule from 2026-08-14.
  Amended 2026-08-15 (final form): at plan time, two skills run with every review - /brainstorming (obra/superpowers: classify the request, widen the option space before converging, hard approval gate) and /grill-me (mattpocock: walk the design tree dependency-first until understanding is shared).
  /to-tickets slices each ruled plan into tracer-bullet tickets with blocking edges before lane dispatch; /tdd runs at every task start, binding on every implementation ticket in every lane brief.
- Task lifecycle (Vlad, 2026-08-15): at task START, after a plan or decision is confirmed, run the superpowers workflow (worktree isolation, executing-plans or subagent-driven-development per the work's shape, test-driven-development, and verification-before-completion before any success claim).
  DURING the work, the /spartan quality gates run between every step - the phase reviewer gates each phase and does not let work skip ahead.
  At task COMPLETION, run the code-simplifier plugin on the changed code.
  Completion is not claimable until the gates passed along the way and the simplifier pass is done.
  The coordinator writes the lane plans into a design doc under ~/.gstack/projects/kernelverify/, the review rules every open decision with Vlad, and a lane brief must trace to a ruled plan with no unresolved decisions, the way W1-W8 traced to the runner-consolidation review.

## Mistakes already encountered

- The shell cwd resets between tool calls, so `cd` into YOUR OWN checkout in every command.
  For the main session that is `/Users/vlad/kernelverify`; a worktree lane uses its own path (`/Users/vlad/kv-*`), never main's.
- The verdict cache stores plain tuples, not dataclasses, because pickled dataclasses remember their defining module and break when loaded from an import context.
- Structured input modes can make a correct fp32 kernel exceed the published tolerance against the fp64 reference.
  That is ill-conditioning, not a port bug; the shipped oracle handles it with the ensemble-floor tolerance (ADR 0004), and a control failing that tolerance on any mode now always means the oracle or the port is broken.
- An input-perturbation probe is not a substitute for the ensemble floor: it misses internal accumulation error by up to 64x, and probing at fp16 epsilon absolves fp16-internal faults.
  The falsification trail is in ADR 0004; the probe survives only as the repaired opaque-reference fallback.
- Table cells rounded to whole percents once overstated a result (99.8% shown as 100%); keep one decimal and verify exact counts for any 100.0% cell.
- A deterministic-only test policy plateaued at 88% while random reached 98%; exploration must survive in any policy.
- Pure pair-greedy coverage cratered to 67% at B=4 by buying pair density before basic diversity; cover single features first, then pairs, then random.
- The corpus fp64 references run as subprocesses over a stdin/stdout protocol; they are the slow part of any rebuild, so batch and cache around them.
- Sequential accumulation is not the worst legitimate summation order, despite what ADR 0004 assumed when it built the ensemble from it.
  Seeded random permutations of the same reduction beat it by up to 8.45x on the attention family, and the magnitude-sorted order that theory nominates as worst is not worst either; ADR 0005 has the measurement.
- A tolerance has two halves and both need fingerprinting.
  ADR 0004's cache tag carried K but not the ensemble membership, so changing who computes the floor would have silently reused stale verdicts.
- Any claim about how much of the admissible class an ensemble covers must be scored against an independently seeded draw of that class.
  Scoring a sample against itself makes every ensemble look complete.
- Charging `2 * model_n_params` flops per token overstates prompt processing badly enough to score a run at 110% of the machine's flop ceiling.
  The embedding table is a gather and the output head runs once per decode call, not once per prompt token; take both from the GGUF tensor table (ADR 0007).
- Score every backend against its own ceiling.
  The CPU path measured 13% when divided by the GPU's flop ceiling and 56% when divided by Accelerate's, and only the second number means anything.
- The CPU probes need best-of-50; best-of-5 was still swinging 39% between invocations, while the GPU probes are stable at best-of-12.
- Measure ceilings on AC power and record the power state; the first bandwidth probe on battery read about 6% low.
- A kernel gap seen at one shape is not a kernel gap until a second shape shows it.
  The q3_K deficit reproduced at large reduction dimensions and nearly vanished at small ones, which changes what it is worth (ADR 0007).
- Generation carries a fixed 0.99 ms/token dispatch cost on this machine, so any small-model bandwidth percentage is depressed by it and is not evidence about the kernels.
- `--pure` when quantizing for a kernel comparison, or the K-quant presets mix types per tensor and measure the wrong thing.
  Requantized files are timing artifacts only and are numerically junk; keep them out of any quality measurement.
- Leave-one-class-out is a necessity diagnostic, never a pass or fail gate.
  No ensemble can cover a class it holds no member of, so as a gate it fails every ensemble whose classes are genuinely distinct, and hardest when they are most distinct; ADR 0009 has the amendment and both readings.
- Count ensemble members by comparing their output, not by counting their names.
  `lut-gather` and `dequant-pairwise` in the quant contract are bit-identical at every bit width, so the six-member ensemble is five.
  `device-dequant-loop` is bit-identical to `dequant-pairwise` too (ADR 0012): the dequant step is exactly representable so it must match, and this Metal compiler's vectorization of the sequential loop happens to round like the CPU matmul, so the nine-name device-joined ensemble is seven.
- A member that stands for a class of kernels must round the way that class rounds, not better.
  `factored-groups` formed its per-group sums with numpy's pairwise reduction, which is EXACT on a constant row (64 identical values halve down a power-of-two tree with no rounding), while every real int-accumulate kernel chains the group and rounds 63 times; the member was therefore most accurate exactly where the device kernels are least accurate, the floor was too tight there, and the shipped tolerance flagged a correct device kernel on 10 of 1,536 serving records (ADR 0016).
  Vectorized numpy reductions are pairwise by default; a member whose realism depends on its rounding sequence has to spell the sequence out.
- A K derived on one harness's shapes is not the shipped K.
  The device harness derives K from three synthetic shapes and prints `shipped K`; the serving grid's six real shapes are not among them, and on 2026-08-15 the two disagreed (2.766 against 3.120 at 2560x9728), so taking the printed line would have shipped a K the next serving run refuses through its own DEMAND MISS branch (ADR 0016).
  Read every committed record set before moving K, and remember that passing gates are not the check: they passed at the K that would have refused.
- A cache keyed on member NAMES cannot see a member's arithmetic change.
  The verdict cache and `contract_k.pkl` fingerprint the ensemble by label; the ADR 0016 repair kept the label and moved the floor, and `score_oracles` would have reported the pairwise member's verdicts as the chained member's; `QUANT_ENSEMBLE_VERSION` in `kernelverify/schemas/quant_contract.py` exists to be bumped for exactly that, the way `CONTRACT_VERSION` already was for the unquantized contract.
- `mx.quantize` packs one contiguous little-endian bit stream per row, not 32 // bits values per word; the two agree only when bits divides 32.
  Reading it the wrong way is silent rather than loud, because `verify_canonical_against_mlx` falls back to treating MLX's output as canonical whenever the comparison fails.
- An A/B timing comparison in separate passes measures the clock, not the kernels: GPU power-state drift moved one fixed shape's time from 131.7 to 93.1 us minutes apart, reversing a comparison's sign completely at every point.
  Batching dispatches past 5 ms did NOT prevent this; interleaving the arms within each round is independently load-bearing.
  Interleaving is necessary and NOT sufficient: it equalizes a clock excursion across arms but cannot detect one, so the reference arm's own spread is the detector - reject any round whose reference samples exceed the class spread limit (1.5x max-to-min for kernel arms, tighter for steadier quantities).
  Interleave every comparative measurement, always, even when each dispatch is ms-scale.
- A `k_demand` reading is a K demand, not an error magnitude.
  At fp16 activations every contract member rounds its output to the storage dtype, so the ensemble floor collapses to the output's own rounding, the shipped tolerance's floor term goes inert (base_tol wins in 768/768 fp16 records), and `e / floor` explodes: a 12.4x overshoot of the actual tolerance was read as a demand of 195 (ADR 0014).
  Report the shipped-tolerance overshoot beside the demand, always, and treat a floor that all implementations saturate identically as measuring nothing about class spread.
- Row-partitioning a matmul is NOT bit-exact, however plainly the arithmetic says each output element's dot product is untouched by it.
  Measured on numpy 2.5.2 over Accelerate: cutting the output-row dimension of `x @ w.T` changes dgemm's blocking and moves fp64 results by up to 2e-14, at chunk sizes 1, 7 and 3276 and at shapes from (5, 128, 96) to (16, 2560, 20000).
  Chunk the dequantization instead, which is elementwise per row and therefore exact by construction, and leave every matmul whole.
  This one is silent rather than loud: it would have moved the fp64 anchor that every error is measured against, and the continuity anchor cannot catch it because the standing measure never takes the chunked path.
- An ensemble member called per record re-dequantizes the whole weight matrix per call.
  `ENSEMBLE[name](x, artefact)` takes the quantized artefact and unpacks it internally, so checking five members across 32 lm_head records is 160 independent 1.5 GB dequantizations; a derived-column generator written that way reached 29 GB and was killed by its own watchdog, while the same work through the harness's hoisted evaluators holds ~16 GB steady with transients to ~23.
  `bench/calibrate_quant_serving.py` already solved this: `eval_pairwise` / `eval_lut` / `eval_serial_chunked` / `eval_factored_serial` / `eval_factored_groups` take the dequantized arrays instead of the artefact, hoisted once per `(shape, draw, seed)` block by `ArtefactHoists`.
  Any new analysis over the serving grid goes through those evaluators, never through `ENSEMBLE` directly; reuse `ArtefactHoists` too unless the analysis needs a strict subset of what it allocates, and say which subset and why if so.
