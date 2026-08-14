# ADR 0006: The Metal runner, and what it costs to execute a candidate kernel

Date: 2026-08-14
Status: Accepted

## Context

Every number this project has published so far was produced by numpy.
The reference kernels, the mutation catalogue, the ensemble floor and the policy tables all run on CPU ports, which was the right way to measure a test surface but leaves the verifier with no way to execute the thing it verifies.
Phase 2 needs one: the optimiser generates candidate kernels for a backend and the verifier gates each candidate, and Vlad's Mac is a first-class target, so the first backend is Metal.

The runner is also where the second half of the optimiser's question is answered.
A candidate is only interesting if it is both correct and faster, so the component that executes a kernel is the component that must time it.

## Decision 1: the artifact is raw Metal shading language plus a launch spec

A candidate arrives as a full MSL translation unit, an entry point name, an ordered list of buffer bindings, and the grid it wants.
Shapes and scalar values are deliberately not part of it, because one candidate runs against a whole battery of cases and each case has different dimensions; they arrive per case instead.

The alternative considered was MLX's `fast.metal_kernel`, which is the shorter path to a GPU on this machine.
It was rejected for the contract, not for its quality: it takes a kernel *body* and generates the signature itself, so the buffer layout and the launch stop being things a generator states and become things the framework decides.
A verifier whose input format silently rewrites the candidate cannot report what the candidate actually did.
The runner therefore drives first-party Metal through PyObjC, which adds `pyobjc-framework-Metal` as this repository's only dependency beyond numpy.

Sizes that vary per case are written as extents: an integer, the name of a run parameter, or a list whose parts multiply.
That covers every launch this battery produces, including a grid of one threadgroup per row of a `(B, S, H)` tensor, without inventing an expression language.

## Decision 2: one worker subprocess per candidate, with results streamed

Nothing in Python can interrupt a dispatched Metal command buffer.
A kernel that does not return can only be stopped by killing the process that owns it, so the parent never hosts candidate code: it starts `kernelverify.runners.worker` in its own session, hands over one batch, and reads results off a pipe.

Results stream one line at a time rather than arriving as a single payload, which is what makes a hang diagnosable instead of merely survivable.
The parent holds a rolling deadline that every received result resets, so the case that was in flight when the budget expired is named, and the cases already measured are kept.

The batch is compiled once, because compilation dominates the cost of a small kernel.

| Batch size | Total | Per case |
|---|---|---|
| 1 | 208.9 ms | 208.91 ms |
| 8 | 173.9 ms | 21.73 ms |
| 16 | 190.3 ms | 11.89 ms |
| 64 | 251.0 ms | 3.92 ms |
| 8 separate batches of 1 | 1295.9 ms | 162.0 ms |

Fixed cost is about 180 ms of process start, device open and compile; the marginal cost of a case is about 1.4 ms for seven dispatches.
At the shipped policy's budget of 16 cases per operator, gating one candidate costs about 190 ms.

A fresh process per *candidate* is not amortised away, deliberately.
Reusing a worker across candidates would let one candidate's damage reach the next one's verdict, which is the failure mode a verifier can least afford.

## Decision 3: execution failures are statuses, not exceptions

A generated kernel fails in ways a numerical oracle cannot interpret, and every one of them has to be survivable inside a sweep.

| Status | Cause | Measured |
|---|---|---|
| `compile_error` | MSL rejected, or the entry point does not exist | 0.19 s, carries the compiler diagnostic |
| `launch_error` | dispatch refused, or the command buffer failed | 0.24 s |
| `invalid_spec` | a binding or extent names something the case lacks | 0.00 s, no process started |
| `timeout` | the worker produced nothing inside its budget | 0.36 s against a 0.6 s kernel |
| `crash` | the worker died before reporting | 0.01 s |
| `unsupported` | no Metal device on this machine | not a verdict on the kernel |

Two of these are load-bearing rather than cosmetic.
An oversized threadgroup is a programmer error that Metal answers by aborting the process, so the runner checks the request against the compiled pipeline's own limit first and returns `launch_error`; without that check a bad launch would be indistinguishable from a crash.
An invalid spec is caught before a process is started, so a malformed candidate costs nothing.

## Measurement 1: the timing has a resolution floor, and it is higher than it looks

Comparing candidates needs a number that survives interference.
Interference only ever makes a kernel look slower, never faster, so the reported headline is the minimum over the repeats rather than the mean or the median.

Measured first on an idle M3 Pro, warmup 2, 50 repeats, the picture is reassuring: the body of the distribution is tight at every scale, median over minimum between 1.00 and 1.07 from 2.4 microseconds of GPU work upwards, with occasional 3x outliers in the tail below ten microseconds.
Warmup barely matters at that scale; worst spread over twelve trials was 1.24 at warmup 0 and 1.22 at warmup 2.

That reassurance does not survive a loaded machine, which is the condition the rest of this measurement was taken under, with three sibling agent sessions working on the same Mac.
Reproducibility of the *minimum* itself, best-of-5 repeated over 12 runs:

| GPU time per dispatch | Spread of best-of-5 across runs | Usable for ranking |
|---|---|---|
| 2.5 us | 24.1x | no |
| 5.5 us | 15.4x | no |
| 22 us | 2.75x | no |
| 196 us | 1.32x, and up to 4x when the machine is busier | no |
| 1.5 ms | 1.05x | yes |
| 14.5 ms | 1.05x | yes |

The short-kernel variation is not random jitter, and that matters for what can be done about it.
Within a single 24-repeat run at 200 microseconds the samples move together, drifting from 150 to 200 microseconds, while a different run of the same kernel sat near 600 microseconds throughout with one sample at 329.
Whole runs move together rather than individual samples straying, which is consistent with the GPU changing power state; the mechanism was inferred from the timing shape and not confirmed against a clock counter.
Either way the practical consequence holds, because averaging and extra repeats do not recover a signal that shifts uniformly.
At 14.5 milliseconds the same 24 repeats span 14495 to 14505 microseconds, a spread of 1.001.

The consequence for the optimiser phase is a rule rather than a caveat: rank candidates on kernels that run for milliseconds.
A speed claim made on a shape small enough to finish in microseconds is not a measurement, however many repeats it averages.

Two versions of this component's own test failed in the suite by asserting a spread bound on a short kernel, which is how the floor was found.
The test now asserts the separation between a 200 microsecond kernel and a 58 millisecond one, and asserts tightness only on the long one.

### Interleaving: proposed as an independent rule, did not reproduce here

Interleaving the two arms of a comparison within a round was proposed as load-bearing on top of ms-scale dispatches, after a sibling lane measured a 1.45x win reverse to 0.77-0.94x once interleaved, at dispatches already past 5 milliseconds.

It did not reproduce on this runner.
The same kernel at a known 1.300x iteration ratio, running at 29.0 and 37.7 milliseconds, returned exactly 1.300 in five trials, both blocked (all of A, then all of B) and interleaved (A, B, A, B), with no variation between trials at all.

That does not refute the sibling result, it bounds it.
The likely difference is the window rather than the order: both arms here sit seconds apart inside a single process, while a reversed comparison spanned minutes, and the drift measured above moves whole runs rather than samples within one.
Two other differences are unexamined and could matter more: this probe is pure ALU with tiny buffers, so it is insensitive to memory bandwidth in a way a real quantized matmul is not, and it times GPU timestamps rather than a host-side lazy-evaluation boundary.

So the rule that survives both measurements is about elapsed time, not ordering.
A comparison whose arms are separated by minutes, by separate processes, or by anything else that lets the power state move must interleave.
Arms measured back to back inside one process did not need it here.
Interleaving is cheap, so it stays the default for any published comparison; the correction is that ms-scale dispatch alone is not sufficient, and neither is interleaving alone.

## Measurement 2: the ensemble floor covers a real GPU kernel

ADR 0004 calibrated the shipped tolerance against an ensemble of provably correct implementations, all of them numpy: pairwise accumulation, sequential accumulation, a second flash tile width.
Whether that floor covers an implementation the calibration never saw was an open question, and a real GPU kernel is the first chance to ask it.

A Metal softmax that reduces in a third structure again, a strided tree across 64 threadgroup lanes, was judged against the corpus fp64 reference under `conditioned_tolerance`, at a 1e-5 base tolerance:

| Case | Correct kernel error | Tolerance | Verdict |
|---|---|---|---|
| B=2, S=7, H=256, corpus uniform | 7.45e-09 | 1.00e-05 | pass |
| B=1, S=3, H=1025, corpus uniform | 1.86e-09 | 1.00e-05 | pass |
| B=2, S=7, H=3, near zero | 2.98e-08 | 1.00e-05 | pass |
| B=1, S=7, H=256, constant rows | 0.00e+00 | 1.00e-05 | pass |
| B=8, S=1, H=1025, opposed signs | 1.86e-09 | 1.00e-05 | pass |

The margin is about 300x at worst, so the floor is not close to false-positiving on a genuinely different reduction order.
This is evidence, not proof: one operator, one kernel, fp32 only.

The same cases were run with the `tl.exp2` confusion seeded into the kernel, computing 2^x where the reference computes e^x.
It is caught on four of the five, by margins between 110x and 3300x, and escapes on constant rows with an error of exactly zero.
The reason is legible and is ADR 0001's thesis restated on hardware: when every element of a row is identical the softmax is 1/H under any exponential base, so no oracle looking at that case can see the fault.
The test pins the escape as an escape rather than relaxing the assertion.

## Consequences

The verifier can now execute and time candidate kernels on Apple silicon, and a candidate that fails to compile, refuses to launch, hangs or crashes comes back as a named status that a battery run survives.

Known limits, stated rather than hidden:

- Isolation is process level only.
  A kernel that writes outside its buffers can corrupt GPU memory belonging to that process, and killing the worker is the remedy, not the prevention.
- A correct but slow kernel and a hung one are the same event to the runner.
  The budget is the caller's policy, and calibrating it against the reference implementation's own time is the obvious next step.
- One output tensor per kernel, because that is what the corpus schemas declare.
  Lifted by the consolidation update below: the transport is multi-output as of W1.
- Timing below roughly a millisecond per dispatch is not reproducible under contention, so the optimiser must size its comparisons above that floor rather than trusting a fast number on a small shape.
  Whether a dedicated quiet-machine mode is worth building is a separate question, and it should be answered by measuring how often the floor actually blocks a ranking.

Not done here, and deliberately: the runner is not yet wired into the case space or the scoring harness, so no published table changes.
Doing that is the step where a Metal kernel can be scored against the full battery, and it touches files this component does not own.

## Consolidation update (2026-08-14, dispatch W1-W6)

The runner consolidation kept every decision above and changed four things, each behind a measurement.

### The transport is multi-output, and it carries integers

`RunCase.output_shapes` is a list of `(shape, dtype)` and `RunResult.outputs` a list of arrays, one per output binding in binding order; the single-output restriction is gone.
The integer dtypes were not planned, they were forced: MLX's generated kernel ABI appends `const constant int*` shape buffers and `const constant int64_t*` stride buffers when a kernel body indexes through them, so an extracted kernel cannot be bound without int32 and int64 input tensors, and uint32 followed for packed quantized words.
That is the shape/stride self-correction trail: the first bridge draft guessed the ABI, the guess was replaced by parsing the signature MLX actually generated, and any parameter the parser does not recognise is a loud error rather than a guess.
ADR 0011 (certificates) takes this trail as evidence for its extraction-provenance tier.

### Specs of one candidate share a worker session, and a hang costs one case

A candidate may arrive as several specializations of one template.
They now share one worker process, one compile per spec, because the isolation argument in Decision 2 is about candidates, not about specs: damage by one specialization can only reach its siblings' verdicts, never another candidate's.
Resume-from-index is ported from the retired runner: a hang or crash is charged to the case in flight (or to the spec being compiled, when no case was), and a fresh worker resumes from the next case.
The one deliberate non-resume is a worker that dies before opening the Metal device, which indicts the environment rather than the candidate.

### Extraction: the verbose dump, and what did not reproduce

The MSL a surface actually runs is captured from `mx.fast.metal_kernel`'s call-time verbose dump.
Two mechanics were measured on mlx 0.32.0:

- The dump is written by the C++ layer straight to file descriptor 1; a Python-level `sys.stdout` redirect captures zero bytes, so the capture worker rebinds fd 1 itself before MLX loads.
- The eng review's stated reason for one process per specialization was that the in-process kernel cache suppresses reprints.
  That suppression did not reproduce: a repeat verbose call reprints in full, cache warm or not.
  The rule survives anyway, on attribution grounds: a process whose fd 1 carries exactly one dump cannot mis-attribute source to the wrong specialization, whatever a future MLX does.

The match criterion between the extracted kernel under this runner and the live surface under MLX is the shipped oracle's verdict function across the battery including structured modes, never bitwise; the silu extraction matched on all four modes, and a strided kernel with shape/stride metadata round-tripped bit-exactly on synthetic data.

### Compile options are mirrored and recorded, and divergence has a signature

Metal's own compile default is fast math; MLX's documented default for custom kernels is `math_mode="safe"`.
Until the consolidation this runner compiled candidates under Metal's default, which judged them under a different math regime than the framework they are meant to replace.
The runner now defaults to MLX's documented default, and the worker echoes the options each shader was actually built with into `BatchResult.compile_options`, which certificates record as inputs (ADR 0011).
A validation failure pattern that concentrates on reassociation-sensitive cases while order-insensitive cases pass is raised as the compile-option alarm, because that is what a math-mode divergence between arms looks like.

### compare() (decision D6): the reference arm is the canary

`compare(spec_a, spec_b, cases)` holds both specs compiled in two live workers and dispatches the arms back to back each round, which is the interleaving default the measurement section above already argued.
`spec_a` is the reference arm, and any round in which its own samples spread wider than 1.5x max-to-min is auto-rejected; a comparison with any rejected round is not ok and a gate built on it exits non-zero.
This is the wide-qmv canary made structural: ratios the reference arm cannot back are refused rather than withheld quietly, and the per-round evidence is kept so a refused run is inspectable.

### Cross-reference

The C1-at-fp16 topology split (numeric enforcement for sequential-class accumulators, structural attestation for tree-class, commit b3fc3b8) is the first live instance of the per-clause attestation-mechanism field that ADR 0011 will formalize, and the compile-option record above is the same idea applied to the compiler: every claim a certificate makes names how it was checked.
