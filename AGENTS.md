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
  ADR 0012 is the reading of its run; reruns must reproduce its tables.
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
- `bench/machine_state.py` - the idle gate and the timing floor, with the reason each exists.
- `bench/detached_run.py` and `bench/start_binding_run.sh` - the detached run path: a one-shot launchd job that waits for a strong-idle window, then runs `measure_baselines.py` with no terminal attached, retrying up to three passes on dispersion (ADR 0010).
- `bench/OPERATOR-CARD.md` - the one-card instruction for starting a binding run and reading its outcome.
- `bench/mlx_info.py` - the MLX safetensors equivalent of `gguf_info`, so both stacks get modelled bytes rather than file size.
- `bench/.baselines/SCHEMA.md` - the row contract the per-chip matrix renderer consumes. The producer validates against it on every write.
- `bench/results/` - the recorded baselines, committed. ADR 0007 is the reading of them.
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
.venv/bin/python bench/calibrate_quant_device.py      # ~7 min, needs the Metal GPU, must reproduce ADR 0012
```

- Verdicts are cached at `bench/.cache/verdicts.pkl`, fingerprinted by mutation names, input modes, K and every ensemble member's label; any catalogue, mode or oracle change rebuilds automatically.
- Contract measurements are cached at `bench/.cache/contract_k.pkl` under the same discipline, fingerprinted by the contract version and sample size as well.
- Deleting `bench/.cache/` is the safe full reset.
- The environment needs `torch`, which only `gelu[variant=erf]` uses; a worktree venv created without it fails part-way through a verdict build.
- It also needs `mlx==0.32.0` and `mlx-lm==0.31.3`, pinned across worktrees so binding comparisons stay on one toolchain.
  Without mlx, three test modules are skipped whole by a module-level `importorskip`, so the suite reports 181 passed and 3 skipped where an mlx-equipped venv passes 187.
  The skipped modules hide their contents rather than their count, so quote test counts from an mlx-equipped venv only.

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
- `mx.quantize` packs one contiguous little-endian bit stream per row, not 32 // bits values per word; the two agree only when bits divides 32.
  Reading it the wrong way is silent rather than loud, because `verify_canonical_against_mlx` falls back to treating MLX's output as canonical whenever the comparison fails.
- An A/B timing comparison in separate passes measures the clock, not the kernels: GPU power-state drift moved one fixed shape's time from 131.7 to 93.1 us minutes apart, reversing a comparison's sign completely at every point.
  Batching dispatches past 5 ms did NOT prevent this; interleaving the arms within each round is independently load-bearing.
  Interleaving is necessary and NOT sufficient: it equalizes a clock excursion across arms but cannot detect one, so the reference arm's own spread is the detector - reject any round whose reference samples exceed the class spread limit (1.5x max-to-min for kernel arms, tighter for steadier quantities).
  Interleave every comparative measurement, always, even when each dispatch is ms-scale.
