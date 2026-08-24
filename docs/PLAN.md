# metalrunner sprint 1: the verified AI kernel compiler, applied to QLoRA on Apple Silicon

**Status:** final plan, 2026-08-19; Days 0-3 and most of Day 5's code executed; the measurement bridge increment is executed and committed; the live increment is "Next increment: the knob profile, two registered widths, and Amendment 5", RE-CUT 2026-08-20 night after steps 0-3 landed (two rulings taken: the dial width and candidate A's ceiling-and-UNRESOLVED semantics), then RE-CUT AGAIN 2026-08-21 by the Amendment 7 section at the end of this file, which closes the box blocker with four rulings and cuts the calibration from 43 hours to 8.6; the Hoid article of 2026-08-20 re-anchors the strategy in its own section below and supersedes the appendix's competitive read.
Supersedes the brainstorming draft; all evidence sections are kept in the appendix.
Lane zero (the 4-bit decode pricing) is separate, partially executed, and sequenced below.

## Sprint status (2026-08-19, night)

Executed and committed on `lane/metalrunner-sprint1` (28 commits, suite green at 1500 with Metal, 1207+95 skipped without):

| Piece | State |
|---|---|
| Pre-registration doc | Committed alone before any code, plus amendment 1 (the gates ruling below) |
| bf16 contract entry | eps 2^-7, uint16 carrier, exact conversions cross-checked against MLX over all 65536 patterns; ADR 0019; Metal flushes bf16 subnormals and MLX's CPU stream does not, band declared out of contract |
| Gradient mechanism | `mx.custom_function` vjp survives value_and_grad + checkpoint + compile in the trainer's own arrangement; pinned by tests |
| Store, funnel, lint, held-out, search space, brief, stages | All built and tested; V1 (nothing timed before verified) enforced at construction and at run time; the sealed held-out draw leak-tested |
| Tolerance-free gates | 5 designs adversarially reviewed, ALL 5 broken (each with a false positive and a miss); ruling: a gate abstains where it would need a new rule about what a kernel may do; only the unwritten-output gate built (two-pass complementary byte fills); guard-rows and determinism gates blocked on a separate ruling; the abstention ruling is in AGENTS.md and prereg amendment 1 |
| Eight-mutant battery | Each mutant caught by the stage that claims it, earlier stages shown to have no objection, counts verified |
| Loop | Turns end to end with the parametric knob-sweep generator on the real GPU; re-proposals counted apart from candidates and failures |
| LLM generator | Built and run live: 15 claude-fable-5 calls, 0 misfires, 5 distinct kernels, all green through compile-lint-unwritten; every prompt and response stored verbatim |
| Input-support gate | Built after two adversarial rounds (amendment 2); both planted faults refused, correct kernel screens clean |
| metalrunner package | Entry point, stack pin by version and seam hash, routing report, receipt, forced-stock control mode, seam installer with call counting, loss-curve recorder; verified on a real two-iteration fine-tune |
| E2E harness | `bench/train_lora_e2e.py` built by Codex against prereg sections 8-9, then fixed under a 54-agent adversarial review (8 confirmed findings applied); amendment 3 moved the floor rule into the comparison and added the reference-spread REJECTED verdict |
| Measurement bridge | Executed: `metalrunner/measurement.py`, harness diff, `routing.eligible`, lora.main wired through the one installer, meeting-point tests, receipt evidence; committed through `5b3b7f2` |
| Amendment 4 | Committed alone: the profile runs under `mx.disable_compile()` because MLX refuses an eval inside a compiled step; the cost is priced by a compiled/uncompiled total ratio with band [0.90, 1.10] that REJECTS the profile if breached |
| Profile decision arithmetic | `bench/profile_rules.py` plus 30 tests: gain, ratio_lo, reconciliation, compile transfer, kill rule, selection with the tie band as a band rather than a sort key; committed `583072e` |

Parked, waiting on Vlad's explicit go: the calibration run, the Day 1 binding stock profile (selects the first operation), and lane zero's 4-bit pricing run.

## Day 3 increment: the LLM generator (Vlad's rulings, 2026-08-19)

- Backend: the claude CLI, model pinned `claude-fable-5`, schema-forced output; the codex preset (`gpt-5.6-sol`, read-only sandbox) is built as a second argv constant but not the default.
- First live session: yes, run it after tests are green; foreground, 3 rounds of 5 candidates, hard limit, no pricing stage wired.
- Sprint-1 generation runs on the dev machine through Vlad's authenticated CLIs; generation on an end user's machine is out of scope (consistent with the NOT-in-scope list).

### Design (validated against the real seams by a design agent, 2026-08-19)

New module `kernelverify/compiler/llm.py`:

- The model seam is any callable `brief -> raw response text` with a `model_id` attribute; tests inject fakes, production uses `CliModel` (frozen dataclass: pinned argv, model_id, timeout, an `unwrap` that maps the CLI's envelope to the assistant text).
- `CliModel` runs the argv via `subprocess.run` with the brief on stdin and a hard timeout; nonzero exit and timeout raise a typed `ModelCallFailed`.
- Structured output forced through the CLI's JSON-schema flag with one pinned schema `{"kernel_source": string}`; extraction is `json.loads` plus three assertions, no fence parsing, no repair.
- Refusal taxonomy for model calls that yield no candidate, each counted and stored, never guessed around: `exit`, `timeout`, `envelope` (stdout is not the CLI's envelope), `schema` (assistant text is not the pinned shape; an explicit textual refusal by the model lands here).
- `generate_session(operation, model, funnel, store, *, context, rounds, per_round, limit)` lives in llm.py so fakes can test the whole round-trip: build brief from store, lazily call the model as the loop pulls, screen through the funnel, rebuild the brief next round; the store is append-only so re-running simply continues, with duplicates refused by hash into the reproposed count.
- Argv presets to pin (verify the two uncertain spellings with one live call each at landing time, then fixture the observed envelopes):
  `claude -p --model claude-fable-5 --output-format json --json-schema <store>/schema.json`
  `codex exec -m gpt-5.6-sol --json --output-schema <store>/schema.json -s read-only -`

Store changes, all additive (old journals must load byte-identically):

- `propose(..., response=None)` stores the raw model stdout verbatim as a content-addressed blob, mirroring the existing prompt blob (R14: prompt and response verbatim).
- `response(candidate)` reader mirroring `prompt()`.
- `no_candidate(kind, *, prompt, response, detail)` event for model calls that produced no source to hash; `census()` counts them as `no-candidate:{kind}` so the session and the next brief's census can never undercount; the detail field is never read by the brief (rule V2 holds).
- `note_generator(model, argv, schema, timeout)` event, written once per session: the R14 audit record proving which binary, flags, model id and schema ran, in the same clock-free journal.
- Two guarded reader edits so the new events load: census's else-branch keys on `event == "stage"`, `get()` filters with `.get("candidate")`.

Loop change: `Proposal` gains `response: str | None = None`, threaded into `store.propose`.

Entry point `bench/generate_candidates.py` (foreground): argparse with `--store`, `--backend {claude,codex}`, `--model` (required, no default), `--rounds`, `--per-round`, `--limit`, `--timeout` (default 300); writes the schema file into the store dir; funnel is compile -> lint -> unwritten only, so pricing is absent rather than disabled; prints each round's session summary plus refusal counts.

Model id auditability: per candidate in the origin string (`llm <model_id> r<round>`), per session in the `generator` journal event.

### Tests (no network, no CLIs, no GPU needed for the llm tests)

In `tests/test_candidate_store.py`: response blob verbatim and content-addressed; old journals still load; a no-candidate event counted in the census with its response readable; the generator event carries model and argv and breaks no reader.

New `tests/test_compiler_llm.py`: prompt and response land byte-for-byte; garbage JSON is a counted `schema` refusal and no candidate; a duplicate source is a re-proposal; the model id appears in the journal twice; a half-precision kernel from the fake dies at the real lint and the NEXT round's brief contains its failure line (the feedback round-trip); a raised `ModelCallFailed` is counted, never guessed; `CliModel` mapped to the taxonomy using `sys.executable -c` one-liners (real subprocess, zero network); extraction refuses missing key, empty source, non-object.

### Sequencing

1. Store additions plus store tests.
2. Proposal.response threading.
3. llm.py plus its test file.
4. bench/generate_candidates.py.
5. One live envelope-verification call per backend, fixture the observed shapes, finalize the two unwrap functions.
6. The first live session: `--backend claude --model claude-fable-5 --rounds 3 --per-round 5`, foreground; report the session census and the store path.
7. Commit per green step, one coherent commit per piece, on the existing branch.

### Verification

- Full suite green before every commit; `KV_FORCE_NO_METAL=1 pytest -m gpu` shows no unprotected test.
- The live session's acceptance: every model response stored verbatim, every refusal counted under its taxonomy kind, the census sums to responses received, and at least one candidate reaches the unwritten gate (whether or not it survives).
- A deliberately wrong model id run once to confirm the `exit` refusal path fires and is counted rather than crashing the session.

## Current increment: the Day 1 profile harness (planned 2026-08-20; eng review same day, 11 decisions ruled)

### Context

- The decision arithmetic (`bench/profile_rules.py`) and Amendment 4 are committed; nothing yet produces the numbers they consume, and the profile gates everything after it, because its selection rule picks the sprint's first operation.
- Probes on the real stack (2026-08-20) settled the instrument question.
  An eval placed inside a `custom_function` vjp is permitted, fires in true backward order, and forces real work: 84 ms of gradient time appeared between two marks of a 195 ms step.
  Hand-chaining the step out of staged vjp calls costs +75% and is rejected.
  Instance-level `__call__` assignment does not dispatch in Python, so that mechanism silently installs nothing.
  Marks on a synthetic six-region chain cost +40.5% against a 2% reconciliation limit, so the check as registered rejects any working instrument.
- Vlad's rulings, in order taken: calibrate the instrument cost then amend; pin a Dolly-15k subset; split the instrumented pass per candidate at the deciding cell only; extend `Seam` to dotted paths; extract the shared runner first; budget memory from the instrumented mode; state the L/Q overlap in Amendment 5; assert the process is stock; interleave the floors; rename the ceiling sweep; test the marks' real failure modes.
- Codex reviewed the plan independently and three of its blockers reproduced against the installed stack. They are folded in below and each is marked OV.

### What already exists, and what this reuses

| Need | Existing | Verdict |
|---|---|---|
| Install a replacement, count calls, refuse a foreign patch, restore in reverse | `metalrunner/seams.py` `Installation` | Reuse, extended to dotted attribute paths (step 1) |
| Child spawn, preflight, model and data hashing, stack record, atomic recording, exit-code mapping | `bench/train_lora_e2e.py`, 2156 lines | Extract into a shared runner both harnesses import (step 2) |
| Two-arm comparison with a drift canary | `bench/interleave.py` `interleaved_samples` | Reuse for every floor |
| Spread, round eligibility, `RunInvalid` | `bench/decode_rules.py`, `bench/machine_state.py` | Reuse unchanged |
| Lock, idle gate, memory budget, detached protocol | `bench/machine_state.py`, `bench/memory_guard.py`, `bench/detached_run.py` | Reuse unchanged |
| Gain, ratio_lo, reconciliation, transfer, kill, selection | `bench/profile_rules.py` | Reuse; extended only by the collapse rule (step 4) |

`bench/roofline.py` is NOT this: it measures the machine's bandwidth and FMA ceiling. The profile's quantized-versus-dense sweep is named `ceiling_sweep` throughout, so one directory does not hold two things called roofline.

### The instrument: pass-through marks, no recompute (`bench/profile_instrument.py`)

A mark is an identity `mx.custom_function` on the tensors flowing between regions.
Forward, it evals its inputs and timestamps.
Its vjp evals the cotangent, timestamps, and returns it unchanged.
Stock operations are untouched and never recomputed: MLX's own backward runs between the marks, and each eval fences everything pending since the previous mark, which is what makes a timestamp a completion time.

```
  forward  ->  [enter]--- attention ---[exit]  ...  [enter]--- head ---[exit]  -> loss
                  |                       |             |                 |
                  t0                      t1            t2                t3
                                                                           |
  backward <- [mark]--- attention.bwd ---[mark] ... [mark]--- head.bwd ---[mark]
                  t7                      t6            t5                t4

  region span = (t1 - t0) + (t6 - t7)        gaps between spans = the remainder
```

Patch points, all class- or module-level so one install covers all 36 layers and the parameter tree is untouched:

| Patch point | Region | Note |
|---|---|---|
| `mlx_lm.models.qwen3.scaled_dot_product_attention` | attn-core | a re-export from `mlx_lm.models.base`, which is exactly the `defined_in` case `Seam` already handles |
| `mlx.nn.QuantizedLinear.__call__` | qmm | LoRA holds the base layer as `self.linear`, so the base matmul is reached through here; adapter matmuls and swiglu fall into the remainder, correctly, since candidate Q replaces only the base kernel |
| `mlx.nn.QuantizedEmbedding.as_linear` | head-matmul | the tied head is a quantized matmul, verified in source, so it counts in both f_L and f_Q |
| `mlx.nn.losses.cross_entropy` | cross-entropy | the trainer resolves it through `nn.losses` at call time |

Shares: f_A = attn-core; f_L = head-matmul + cross-entropy; f_Q = qmm + head-matmul. Forward and backward spans both counted.

**Split passes at the deciding cell.** Every mark costs time, and that cost lands inside the marked region's numerator while the step total it divides by carries every region's cost. Attention is marked 36 times per step, the head once, the quantized matmul about 250 times, so one combined pass inflates f_Q and deflates f_A and f_L, biasing the very comparison the rule makes. Cell B therefore runs one instrumented pass per candidate. Cells A, C and D, which decide nothing, keep one combined pass and their shares are labelled as carrying cross-region bias.

**Completeness (OV).** Compared against the marked pass's own elapsed, "spans plus gaps equal the step" cannot fail, because the remainder is defined as whatever the spans leave on the same timeline. Amendment 4 does not say that: it compares the sum against the step "taken without the interior boundaries", which is the plain pass, and read that way it is the instrument's cost against a 2% limit. So the registered reconciliation and Amendment 5's instrument-cost band are one check at two limits, and today the 2% is what binds and no working instrument meets it.

The check that catches a seam is the count, and the counts are NOT symmetric. Forward, attention fires once per layer and each projection once per layer. Backward, mlx-lm adapts only the last `num_layers` blocks, so attention fires once per ADAPTED layer and the projections `7 * adapted - 3` times, the three being those that consume the lowest adapted block's input, whose gradient nothing below asks for. Measured at 1, 2, 4 and 8 adapted blocks on 2026-08-20 and exact at every one. The head and cross-entropy fire once per step in each direction. A count short of that means a seam never fired, and without it that region's share reads as zero and lands silently in the remainder.

**The marks must be identity (OV).** Equal batches across modes is not sufficient. The plain and instrumented passes must produce the same loss, the same gradients and the same final adapter state, or the custom gradient rule changed the computation and the two modes describe different work.

**The process must be stock.** The profile's whole claim is stock with none of our kernels. Before timing, assert that all four seams hold their original objects and that no metalrunner seam is installed, record that they did, and refuse otherwise. `Installation`'s foreign-patch refusal gives this almost free.

### The harness: `bench/profile_stock.py`

**One child per cell, one fixed batch, modes interleaved (OV).** Amendment 4 requires the step total both ways "in the same run, on the same batch". A child per cell holds one batch and drives the trainer's real step through every mode round-robin, five rounds, rotating mode order so no mode always runs warm. This satisfies the amendment literally, makes rounds genuine repeats so the spread gate reads the machine rather than the workload, and removes most of the run's process-startup cost. A crash loses that cell's whole set, which the detached runner already treats as a retry.

Modes at cell B: `compiled`, `plain`, `instr-A`, `instr-L`, `instr-Q`. Modes at cells A, C, D: `plain`, `instr-all`.

| Ratio | Numerator | Denominator | Band | Effect |
|---|---|---|---|---|
| Compile transfer (Amendment 4) | plain total | compiled total | [0.90, 1.10] | breach REJECTS |
| Instrument cost (Amendment 5) | instrumented total | plain total | set from calibration | breach REJECTS |

Shares take the numerator from the pass that marks that candidate and the denominator from the plain pass in the same child, so each share carries only its own instrument cost.

Memory: the declared budget per cell comes from the instrumented peak, not the plain one. Fencing at the head forces all logits resident at once, which at batch 4 is 2.5 GB on its own. Calibration measures both peaks so the binding run's budget is set from evidence and does not refuse halfway through a granted window.

Recording: exclusive-create JSON with raw samples, derived medians, machine fingerprint, versions, model and data hashes, mark counts, and every labelled assumption.

**A separate decision artifact (OV).** The recording refuses overwrite by design, so the ruling cannot be written back into it. `--decide` reads a recording and writes its own artifact carrying the verdict, the readings it rested on, the recording's hash and the digest of `profile_rules.py` that ruled. The harness never computes a gain.

`--calibrate`: cell B only, instrumented against plain, about 5 minutes, writes a calibration record, binds nothing.

### The floors (`bench/ceiling_sweep.py`, its own child)

Every floor is a two-arm comparison, so every floor runs through `interleave.interleaved_samples` with its canary. The credited ratio is smallest numerator over largest denominator, which is maximally exposed to the machine drifting between two arms timed separately.

| Candidate | Numerator | Denominator | Labelled assumption |
|---|---|---|---|
| Q | stock quantized matmul, both orientations, at every shape | dense fp16 at the same logical shape | dense fp16 is a ceiling because it skips dequantisation; it also moves about four times the weight bytes, so at small token counts this ordering can invert, and the observed ratio is reported whatever it is |
| A | stock attention region | `mx.fast.scaled_dot_product_attention` forward at the same shapes, times 3 | the fused backward MLX does not implement would cost about twice its forward |
| L | stock head matmul plus cross-entropy | the same on only the supervised rows, plus one extra matmul at the head's backward shape | a streamed kernel reaches stock throughput on the reduced problem and pays one matmul for the gradient |

**Shapes (OV).** The registered set S1 to S5 omits k_proj and v_proj, which on the pinned config are (1024, 2560): 8 key-value heads at head dimension 128. Amendment 5 adds it as S6 and restates the kill rule over six shapes, because the share counts every quantized linear and the floor must cover what the share counts.

**The collapse rule (OV).** Q has one credited ratio but six shapes appearing at different frequencies, and spans are measured once per layer per step. The rule that collapses per-shape and per-call numbers into one `ratio_lo` lives in `profile_rules.py` with the rest of the arithmetic, so it is testable without a GPU and cannot be invented by the harness at run time.

**The footprint tie-break (OV).** The first tie-break wants a peak-footprint delta for implementations that do not exist yet. Amendment 5 defines its source as the floor arms' own measured deltas, which is a statement about the floor rather than the kernel, and labels it as such; where no delta can be measured the tie-break falls through to table order, stated rather than silent.

### The dataset: `bench/.data/dolly-1024` (`bench/pin_dolly.py`)

One-time fetch of databricks-dolly-15k, seeded deterministic selection, converted to mlx-lm chat format, committed as `train.jsonl` and `valid.jsonl` with the selection script beside them.

**Filtered to one length bucket (OV).** mlx-lm sorts examples by length and pads each batch only to the next multiple of 32 above the longest row in it (`trainer.py:157`), so batch width is data-dependent and never the registered 2048. The pinned subset is drawn from a single 32-token band, so every batch pads to the same width, that constant is the registered token count, and five rounds are genuine repeats. Real rows, real prompt masks, real supervised fractions. The corpus padding fraction is a different lever and is measured separately on CPU across all 15k rows, needing no GPU.

### Sequencing (commit per green step)

| # | Step | State |
|---|---|---|
| 1 | Extend `Seam` to dotted attribute paths | LANDED `b5f3223`, 30 seam tests |
| 2 | Extract the shared runner out of `train_lora_e2e.py` | LANDED `350f94b`, 2156 lines becomes 1727 plus 487, written by Codex |
| 3 | `bench/profile_instrument.py` plus tests | LANDED `230e039`, 27 tests, no device needed |
| 4 | The collapse rule and the footprint-source rule | LANDED `f165947`, 47 tests |
| 5 | Corpus tooling, band arithmetic, corpus distribution | LANDED `c306141`, 22 tests; the BAND ITSELF IS UNRULED, see below |
| 6a | Prove the marks fire in mlx-lm's real step and change nothing | LANDED `827682a`, 8 live tests |
| 6b | `bench/profile_stock.py` | LANDED `01ca15e`, 74 tests; the band enters as plan data, so it did not wait on the ruling |
| 6c | `bench/ceiling_sweep.py` | NOT STARTED, waits on the attribution ruling, which decides what the floor is compared against |
| 7 | Codex adversarial review of the instrument | RUN, findings folded |
| 8 | Calibration run, 4B cell B, detached | BLOCKED: VLAD GO #1, about 5 minutes |
| 9 | Amendment 5, committed alone | Ten clauses, listed below |
| 10 | Binding profile run | BLOCKED: VLAD GO #2 |
| 11 | `--decide` writes the decision artifact; the ruling lands here | After 10 |

### What Amendment 5 has to carry

Grown from six to ten as the harness and two adversarial passes met the registered text.
Each is a place where the pre-registration as written cannot be implemented, and each has to be settled before the binding run rather than after.

| # | Clause | Why |
|---|---|---|
| 1 | The instrument-cost band | From the calibration run. Section 3.3's 2% cannot be met by any instrument that can put a clock inside an MLX backward. |
| 2 | The reconciliation is the same check | Read as Amendment 4 writes it, section 3.3's reconciliation IS the instrument cost, at a 2% limit. Amendment 5 should say so and set one limit, not leave two names for one quantity. |
| 3 | How a share is attributed | The blocker below. Every share carries its own marks, and the two defensible attributions disagree by 45%. |
| 4 | Shape S6 | k_proj and v_proj at (1024, 2560) exist in the step and are counted in Q's share, and the registered list omits them. |
| 5 | The kill rule over six shapes | `KILL_SHAPES` is 4 and adding S6 silently turns 4-of-5 into 4-of-6, which is a looser rule. It has to be restated rather than inherited. |
| 6 | How the sweep's two directional ratios reduce to the one scalar the kill rule reads | Nothing registers that reduction today. |
| 7 | Candidate A's floor arithmetic | Section 4.2 multiplies the fused forward by 3, which assumes a backward for every forward. Under LoRA there are 36 forwards and 16 backwards, so at the registered 2x-backward assumption the workload is 36 + 2*16 = 68 forward-equivalents, not 3*36 = 108. As written the floor is about 1.6x too large, which credits A with a smaller ratio than its own assumption implies. |
| 8 | The footprint tie-break's reduction and sign | One signed byte delta per candidate, from a sweep that produces peaks across six shapes, two orientations and several rounds. No reduction and no sign convention is registered. |
| 9 | The L/Q overlap | The tied output head is a quantized matmul and counts in both shares, so the two must never be composed. |
| 10 | The sequence length | The corpus blocker below. |

### BLOCKER, measured 2026-08-20: a share carries its own marks, and the two honest attributions disagree

**RULED 2026-08-20: the knob slope replaces the marked span for all three candidates.
See "Next increment" below.
The evidence that forced the ruling is kept here unchanged.**

The split-pass design was meant to make a share safe by taking the denominator from the unmarked pass.
It does stop one candidate's marks from deflating another's share.
It does not stop a candidate's own marks from inflating its own, because a mark evals and then timestamps: the fences sit inside the spans that form the numerator and outside the plain step that forms the denominator.

Measured on the 0.6B model, five rounds, one fixed batch.

| Candidate | Marked calls | Instrument cost | Share as registered |
|---|---|---|---|
| L | 2 | 1.08x | 0.263 |
| A | 32 | 1.54x | 0.193 |
| Q | 197 | 2.86x | 1.401 |

A share of 1.401 is not a fraction of anything, and the overstatement is monotonic in mark count, which is exactly what separates the three candidates.
Two corrections were tried and both failed.
Subtracting the whole excess gave negative shares for A and Q, so the marks' cost is not all inside the spans.
Doubling each mark's fences measured nothing, because a second eval of an already-materialised tensor is free.

The deeper fact is that wall-clock attribution inside a pipelined lazy graph is not uniquely defined.
Marking every block gave attention a share of 0.208.
Marking ONE block and scaling by the block count gave 0.296, a 45% disagreement, at 1.15x the plain step instead of 1.62x.
The blocks are not the explanation: blocks 20, 21, 22 and 27 scaled to 39.4, 40.3, 39.9 and 40.1 ms, agreeing within 2.3%.
What differs is how much neighbouring work each method fences away, and no measurement here says which of the two is the honest answer.

Options, none taken.
Keep section 3.3 as written and accept a bias aligned with the decision.
Mark one block and scale, which cuts the perturbation and rests on an interchangeability assumption that measured true.
Measure both at the deciding cell and register one, which doubles that cell's window.
Replace the wall-time share with a counted one, which is exact and unperturbed but applies Amdahl's relation to a modelled share.

The harness records the excess beside every share, carries the caveat with the number, and blocks a recording whose share is not a fraction.
Section 3.3 stays as written until it is amended.

**Measured after the above, and it settles which of the two is wrong: both are.**

`bench/mlx_probes/probe_attention_ablation.py` measures the same quantity with no instrument inside the step at all.
It replaces attention with something of the same output shape that does almost none of the work, and reads the whole step's time exactly the way stock's own time is read.
Two arms control the two confounds: one keeps the key and value tensors alive through a reduction multiplied by zero, so a lazy graph cannot drop their projections and quietly charge them to attention, and one adds that same reduction beside stock attention so its cost is measured rather than assumed.

| Method | Instrument inside the step | Attention's share at width 97 |
|---|---|---|
| Ablation | none | 0.042 to 0.047 across two runs |
| Marking every block | 44 mark pairs per step | 0.208 |
| Marking one block, scaled | 2 mark pairs per step | 0.296 |

Both marked figures are four to six times the unmarked one.
The size of the gap matches the fence count: at roughly half a millisecond of fence apiece, 44 fences account for most of what the dense spans contain, and the single sparse span carries one fence against one block's worth of real work, which is why scaling it up multiplies the error by the block count.

The ablation is believed because it reproduces a scaling law it has no way of knowing.
Attention's score matrix is quadratic in the sequence length and every other region in the step is linear, so attention's share must be flat at short widths, where attention's own linear part dominates, and then climb.

| Width | Attention ms | Multiplier on doubling | Share |
|---|---|---|---|
| 97 | 5.85 | | 0.042 |
| 193 | 14.17 | 2.42 | 0.053 |
| 385 | 48.94 | 3.45 | 0.087 |
| 769 | 199.87 | 4.08 | 0.161 |

Linear at the short end and quadratic at the long end, with the crossover where the geometry puts it.
Nothing inside an ablation knows what curve it ought to trace.

What the ablation does not settle: removing an operation changes the graph, and a scheduler may behave differently on the smaller one.
It measures what the step stops paying when attention stops happening, which is the quantity `gain = 1/(1 - f*(1 - 1/r))` needs, and that is not the same as what the step spends on attention while everything else is in flight.
Those two differ wherever work overlaps, and MLX does expose `new_stream`, so overlap is possible rather than ruled out.

**Three further probes, and the case is now complete enough to amend on.**

The definition is forced rather than chosen.
`gain = 1/(1 - f*(1 - 1/r))` is algebraically `T(s) = T*[(1 - f) + f*s]` where `s = 1/r`, so at `s = 0` the registered formula itself says `T(0) = T*(1 - f)`.
That is a plain uninstrumented step with the operation's cost driven to zero.
So `f := (T_stock - T_ablated)/T_stock` is the unique f under which the formula's own extreme case is a measured fact rather than a modelling assumption, and it is defined under arbitrary concurrency, carries no instrument in either its numerator or its denominator, and can be measured with compilation ON, which is the step the product runs.

The harness resolves what the tie band demands.
`probe_step_resolution.py` runs two IDENTICAL arms interleaved and finds them 0.155 ms apart, 0.115% of the step, against roughly 1 ms demanded by differentiating the gain formula at section 4.3's two-point tie band.
Installing a seam whose replacement calls the original and returns it unchanged costs nothing measurable.

Attention's share is a unique quantity, measured rather than assumed.
`probe_attention_uniqueness.py` runs attention TWICE with a real data dependency and bit-identical output, and the second serial copy adds one attention's worth of time: kappa 1.11, 1.10, 0.93, 1.07 across two runs and two widths.
So attention was already alone on the critical path here, and removing it and doubling it agree to a few percent while both disagree with the marked instrument by four to six times.

Candidate Q needs a different knob, and that knob works.
A retune keeps the dispatch, the weight traffic and the host graph construction that a stub deletes, so a stub-based share would credit Q with time no kernel could win back at 197 call sites.
Shrinking the arithmetic while holding the launch count fixed and reading the SLOPE of `T(phi) = a + b*phi` measures only what a retune can attack.
`probe_qmm_knob.py` shows the knob is a clean dial at all six of the model's distinct matmul shapes, R-squared 0.995 to 1.0000, worst residual 2.2% of the slope.
Its slope column is a finding in itself: on the smallest projection only 78% of the matmul's time scales with its arithmetic, so 22% is fixed cost, while on the output head essentially all of it is.

Candidate L now has a knob too, and it exposes why the fault was nearly invisible.
`probe_loss_knob.py` shrinks the vocabulary, which cuts the head matmul's arithmetic and the logits the cross-entropy reduces over in one dial, and those two are exactly L's registered region set.
The fit is clean at R-squared 0.9983 and gives f_L = 0.245 by slope against 0.220 by endpoint.

| Candidate | Marks per step | Marked f | Unmarked f | Error |
|---|---|---|---|---|
| L | 2 | 0.263 | 0.245 | +7% |
| A | 32 | 0.193 | 0.045 | +329% |
| Q | 197 | 1.401 | 0.376 | +273% |

The error tracks the mark count, which is the diagnosis predicting itself.
Two marks contaminate almost nothing, so the registered instrument was very nearly right about candidate L and would have been believed without argument.
Thirty-two marks overstate by a factor of four.
A hundred and ninety-seven produce a number above one, which is the only reason the fault was noticed at all.

One bias in the L knob is stated rather than absorbed: its full-width arm is 2.33 ms slower than stock, fifteen times the resolution floor, because it holds a second copy of a 78 MB head weight and reaches it through a Python frame stock does not have.
That is 6.7% of the slope and it biases f_L upward.

Candidate Q was then driven through the real step by the same knob, shrinking all 196 projections at once, at R-squared 0.9962 and with both scaffold prices negligible: applying the slice costs +0.51 ms and installing the seam at all costs -0.19 ms, against a 51 ms slope.
Its seam is `QuantizedLinear.__call__`, which covers the projections and not the tied head, so 0.376 is the projections alone and the registered f_Q would be larger by the head's share.
That also makes the three figures disjoint, which allows the check the marked instrument could never pass.

| Disjoint region | Share |
|---|---|
| Head plus cross-entropy | 0.245 |
| Attention core | 0.045 |
| Quantized projections | 0.376 |
| Sum | 0.666 |
| Unattributed remainder | 0.334 |

Three disjoint regions accounting for two thirds of the step and leaving a third for the adapters, the norms, the elementwise work and the optimizer update is a reading that can be true.
The same three by the registered instrument sum to 1.857, which cannot be, and section 3.3's reconciliation would have had to reject the profile without ever being able to say why.

What is still not built: the knob for the tied head as part of candidate Q rather than only as part of candidate L, and all of this at the registered cell rather than the 0.6B proxy.
What is established is that the registered method cannot be kept.
It reports four to six times the truth for a region with many instances and little work in each, and the three candidates carry 2, 32 and 197 marked instances.

### BLOCKER, measured 2026-08-20: the registered sequence length does not exist in this corpus

**RULED 2026-08-20: two widths are registered and both are measured, the short one from Dolly and the long one from UltraChat 200k, and the selection takes the candidate with the highest MINIMUM gain across the two.
See "Next increment" below.
The evidence that forced the ruling is kept here unchanged.**

Dolly's median row is 116 tokens under the pinned tokenizer, p95 is 569, and 33 of 15011 rows exceed 2048.
mlx-lm pads each batch only to one plus the next multiple of 32 above its own longest row, so a step's width is a property of the data and `max_seq_length` is a cap this corpus never approaches.
Drawing 1024 training rows from a single band, which is what makes a width registrable before a run, is possible only in bands 64, 96, 128 and 160.
The widest is 160, against 2048 registered.

This is not a tidiness problem, and it is no longer an argument.
At 160 tokens and batch 4 a step carries 640 tokens where the target arithmetic was budgeted against 8192.
The unmarked ablation measures attention at 0.042 of the step at width 97 and 0.161 at width 769: a four-fold change in one candidate's share across a width range narrower than the one in dispute.
The mechanism is geometric and reaches all three candidates.
Attention's score matrix is quadratic in the sequence length; the output head, the quantized projections and the loss are all linear in it.
So as the width rises attention's share rises and every other candidate's share falls, and as it shortens the output head's share rises steeply.
Choosing the band chooses which candidate the rule selects, which is the outcome a pre-registered rule exists to prevent.

The same fact reaches the committed end-to-end harness, which also registers 2048 and would also measure short rows.

Options, none taken:
a longer band with too few rows to fill a slice;
packing several rows into one training example, which mlx-lm does not do by default and which changes the mask structure;
a different corpus with naturally longer rows;
registering tokens-per-step rather than sequence length and batch size separately, which preserves the target arithmetic but moves batch size a long way from the registered 1 and 4;
or accepting the short width and rewriting the target arithmetic against it.

The band is left unchosen and no slice is written.
Choosing one is `python bench/pin_dolly.py --band <edge>`.

### Verification

- Full suite green with Metal and under `KV_FORCE_NO_METAL=1` before every commit; the no-device audit stays at 0 passed.
- Steps 1 and 2 are behaviour-neutral and are proven so by the existing tests passing unchanged.
- The 0.6B smoke must show: marks in both directions, counts exactly 36 per layer-region per step and 1 per step-region per step, plain and instrumented agreeing on loss, gradients and final adapter hash, and spans plus remainder within the registered limit of the plain step.
- The instrument tests carry a fixture reproducing the smoke's real mark log, so the span arithmetic is pinned to observed reality.
- The decision artifact names the recording hash and the rule-code digest, so a ruling can be traced to the exact arithmetic that produced it.

### NOT in scope, deliberately

| Deferred | Why |
|---|---|
| Splitting instrumented passes at cells A, C and D | They decide nothing; their shares are labelled as carrying cross-region bias rather than measured clean at roughly double the window |
| Measuring the corpus padding fraction on GPU | It is a deterministic-transform lever, not one of the three candidates, and it reads off the dataset on CPU |
| Building any candidate kernel | The profile chooses; Day 2 builds |
| Reading a Metal GPU trace for per-kernel attribution | Would attribute with zero perturbation, but reading a .gputrace needs Xcode tooling and the kernel names do not map cleanly onto the three candidate regions |
| Re-registering the compile ratio band | Amendment 4 set it deliberately as an uncalibrated judgement and the first run reports the observed value against it |

### Files

- New: `bench/profile_instrument.py`, `bench/profile_stock.py`, `bench/ceiling_sweep.py`, `bench/pin_dolly.py`, `bench/.data/dolly-1024/`, `tests/test_profile_instrument.py`, `tests/test_profile_stock.py`, `tests/test_ceiling_sweep.py`.
- Modified: `metalrunner/seams.py` (dotted paths), `bench/train_lora_e2e.py` and the extracted runner module, `bench/profile_rules.py` (collapse and footprint rules), the prereg (Amendment 5, step 9), `AGENTS.md`, `TODOS.md`, this plan (the ruling at step 11).

---

## Next increment: the knob profile, two registered widths, and Amendment 5 (planned 2026-08-20; nine rulings taken, eng review folded)

This section supersedes step 6c and steps 8 through 11 of the sequencing above, and replaces them with its own sixteen-step sequence.
Steps 1 through 7 of that older sequence stay landed and are reused, with one demotion named below.
Reviewed 2026-08-20 by `/plan-eng-review` and by an independent Codex pass; thirteen decisions are ruled and folded, and the outside pass's fifteen findings are folded or answered in the review report at the end of this file.

### Context: why the increment changed shape

The Day 1 profile exists to pick the sprint's first operation by a rule written before the numbers.
The rule needs one input per candidate: the fraction `f` of the training step that candidate accounts for.
Section 3.3 defines `f` by putting timing marks around a region of the step, and that definition is now measurably wrong.

A mark evals and then timestamps, so its fence sits inside the span that forms the share's numerator and outside the plain step that forms its denominator.
The error therefore grows with how many marks a candidate carries, which is exactly the thing that separates the three candidates.

| Candidate | Marks per step | Share as registered | Share with no instrument in the step | Error |
|---|---|---|---|---|
| L, output head plus cross-entropy | 2 | 0.263 | 0.245 | +7% |
| A, attention core | 32 | 0.193 | 0.045 | +329% |
| Q, quantized projections | 197 | 1.401 | 0.376 | +273% |

The registered method would have selected the operation with the most call sites regardless of where the time actually goes.
The three unmarked figures are disjoint and sum to 0.666, leaving a third of the step for the adapters, the norms, the elementwise work and the optimizer.
The same three by the registered method sum to 1.857, which cannot be true of any step.

Every number in that table was measured with compilation OFF, and this increment turns compilation on.
Step 4 re-establishes each of them compiled, and if any moves, this table moves with it.

### The thirteen rulings, all taken 2026-08-20 before the numbers they govern exist

| # | Question | Ruling | When |
|---|---|---|---|
| 1 | How is a share defined now | The knob slope, for all three candidates, replacing the marked span | Plan |
| 2 | What width does the profile bind at | Two registered widths, both measured | Plan |
| 3 | How much validation before binding | Validate every knob at the real 4B target first, in its own short window | Plan |
| 4 | Which width binds the selection | The candidate with the highest MINIMUM gain across the two widths | Plan |
| 5 | What corpus fills the long width | UltraChat 200k, filtered to one 32-token band | Plan |
| 6 | Do the three reporting cells survive the scope challenge | Yes, kept in full; section 4.3 already promises them and the all-token cell is the only reading that says whether the claim generalises | Eng review |
| 7 | Which dial measures attention | All three candidate dials are built at the proxy and the smallest scaffold price wins, with the most complete dial breaking a tie, registered before any of the three is measured | Eng review |
| 8 | Does the deletion method stay as a cross-check | Yes, and it is re-measured compiled so both halves of the check come from the same regime | Eng review |
| 9 | Is the tie band still resolvable once compiled | Unknown, so it is measured in the validation window and a refusal is registered now against the answer | Eng review |
| 10 | Does one slope rule fit a retune and a rewrite alike | No. A retune candidate is credited with its slope; a rewrite candidate is credited with its slope plus its measured non-scaling residue, and which is which is read off the names section 4.2 already gives them | Codex pass |
| 11 | Where is the floor measured | Inside the real step, at the same seam the knob uses, so the share and the ratio are the same kind of quantity; the loss candidate's floor is not a callable and gets a written exception | Codex pass |
| 12 | One corpus or two | One. Both bands come from UltraChat, so width is the only thing that moves between the two cells, and Dolly leaves the profile | Codex pass |
| 13 | Where do the amendment's numeric limits come from | An instrument-only stage that runs identical arms and scaffold-only arms with no candidate dialled and no share computed, before any knob runs | Codex pass |

### The arithmetic, and why the slope is the only self-consistent input

The registered gain formula is `gain = 1/(1 - f*(1 - 1/r))`.
Rearranged it is `T_new = T - b + b/r` where `b = f*T` is the part of the step the candidate can attack and `r` is how much faster that part could get.
So `f` and `r` must be measured on the SAME quantity, and that quantity is the attackable part, not the whole region.

A retune keeps what a deletion removes.
A retuned kernel still issues one dispatch per call site and still pays the host-side construction of its own operations, both of which a stub deletes outright, 197 times per step.
Measured at the six real matmul shapes, the share of a matmul's time that does NOT move with the dial runs from 22% on the smallest projection down to under 1% on the output head, so the fixed part is real, it is shape-dependent, and a deletion-based definition would hand the sprint some of it as credit.

The knob therefore shrinks the arithmetic while holding the launch count, the output shape and the graph structure fixed, fits `T(phi) = a + b*phi`, and takes

    f := b / T_stock          the attackable fraction of the step
    r := b_stock / b_floor    how much faster the attackable part could get

with both slopes measured by the same dial.
Under that pairing the registered formula is literally correct rather than approximately correct, which is the argument Amendment 5 rests on.

**A retune and a rewrite do not have the same attackable part (ruled 2026-08-20).**
Only one of the three candidates is a retune, and section 4.2 says so in its own name: "quantized matmul retune at training widths".
The other two are rewrites.
The attention candidate fuses the whole region and never materialises the score matrix; the loss candidate streams the head and never builds the full logits tensor.
Both delete cost that lives in the intercept, which a slope cannot see, so one slope rule would under-credit both of them against the one candidate the slope fits exactly.
The credit therefore splits by candidate type, and the type is read off committed text rather than chosen:

| Candidate | Registered as | Credited with |
|---|---|---|
| Q | a retune | its slope |
| A | tiled attention forward and backward, a replacement | its slope plus its measured non-scaling residue |
| L | a streamed mask-aware head and cross-entropy, a replacement | its slope plus its measured non-scaling residue |

The residue is the difference between the fitted intercept and the ablated step, which the cross-check already measures, so this costs no new arm.
It does mean the rejected deletion method supplies one term for two candidates, which is stated plainly rather than hidden: the deletion is rejected as the DEFINITION of a share and retained as the measurement of a residue, and those are different jobs.

**The share and the ratio have to come from the same place (ruled 2026-08-20).**
`f` is measured inside the real training step, where everything else is running.
Measuring `r` on a bench where the operation runs alone assumes the isolated speedup survives the crowd, and a crowded step is exactly where that assumption fails.
So the floor is installed at the same seam the knob uses and dialled inside the same step: the dense comparison and MLX's fused attention are both real callables, so the machinery already exists.
The loss candidate's floor is a streamed kernel that does not exist yet, so it keeps a bench measurement under a written exception naming what that costs.

**What the dial actually moves, stated precisely, because the amendment has to say it.**
Shrinking a matmul's input width cuts its arithmetic AND the weight bytes it reads, and a retuned kernel can improve how it reads those bytes but cannot decide not to read them.
So `f` is the fraction of the step that scales with the operation's SIZE, not the fraction that is arithmetic, and read as "what a retune can win" it is an upper bound.
That is consistent rather than sloppy, because the floor is measured with the same dial: the dense comparison also reads its weights, so whatever traffic is irreducible sits in both slopes and divides out of `r`.
It does mean a floor can come out slower than stock at small token counts, which section 4.2 already anticipates and reports rather than suppresses.

### The instrument: `bench/profile_knobs.py`

One module, one arm runner, three knobs, no timing device anywhere inside the step.

| Candidate | Seam or seams | The dial | What scales with it |
|---|---|---|---|
| L | `QuantizedEmbedding.as_linear` plus `nn.losses.cross_entropy` | vocabulary rows kept | the head matmul's arithmetic and the logits the loss reduces over |
| A | `mlx_lm.models.qwen3.scaled_dot_product_attention` | one of three candidate dials, chosen by measurement, see below | depends on the dial |
| Q | `nn.QuantizedLinear.__call__` plus `QuantizedEmbedding.as_linear` | input width of the activation and the weight | every projection's arithmetic, and the head's, because section 3.3 puts the head in Q's region set |

**Attention has no obvious dial, so the dial is chosen by measurement against a criterion registered first (ruled 2026-08-20).**
Attention is two matmuls with a softmax between them, and no single dial shrinks all three without changing something else.

| Dial | Scales | Cost of the construction |
|---|---|---|
| Head dimension on queries, keys and values, output padded back | both matmuls | the pad grows as the dial shrinks, which flattens the line and understates attention rather than overstating it; at the full setting the pad has zero width and may be optimised away, leaving that arm structurally unlike the rest |
| Head dimension on queries and keys only | the first matmul only | no scaffold at all, and no pad, but roughly half of attention's matmul work never moves |
| Key and value sequence length | both matmuls and the softmax | the causal mask stops being the plain string the training path passes, so every arm needs a hand-built mask array and may leave the fused path stock actually takes |

All three are built at the 0.6B proxy in step 3, and the registered criterion is: the dial with the smallest scaffold price against stock wins, with the most complete dial breaking a tie.
The criterion is written into Amendment 5 before any of the three is measured, which is what stops the choice being made by the number it produces.

**RULED 2026-08-20 night: the selection binds at the LONG registered width, with two readable-price conditions added; Amendment 6 (re-cut step 5) carries the text.
The evidence that forced the ruling is kept below unchanged.**

**BLOCKER, measured 2026-08-20: the criterion names no width, and the width picks the dial.**
All three dials are built and each holds every property a knob must: bit-identical to stock at the full setting in the loss and in all 56 gradients, the call count fixed at 28 across every setting, the scaffold arm computing stock's own value, and four distinct settings on the ladder.
Then the registered price statistic was applied at two widths, with all 28 arms interleaved inside one set of rounds so no drift sits between the three prices being compared.

| Queries | kv-length | head-dim-qkv | head-dim-qk | Chosen |
|---|---|---|---|---|
| 96 | 0.101 | 0.937 | 0.440 | kv-length, on price |
| 384 | 0.221 | 0.027 | 0.029 | head-dim-qkv, tied with head-dim-qk and broken by rank |

The dial is not a presentation choice, it sets the number: at 384 queries the length dial reads a share of 0.060 and both head-dimension dials read 0.023, against an unmarked ablation of 0.068 measured in the same rounds.
So the criterion at that width picks a dial reading a third of what removing attention outright costs, which is what the plan already predicted for those two dials and did not predict the criterion would prefer.

Two things are wrong here and they are different.
The criterion rewards a scaffold that does not move, and a dial that moves less of attention has less scaffold to move, so cleanliness and completeness pull against each other and only the tie-break knows that.
And every scaffold slope measured so far is inside the machine's own jitter, so what separated the three prices at either width is noise: the best knob fit reached R-squared 0.946 and clause 6's gate demands 0.99.

Attention is 2 to 6 percent of the step at these widths, so the dial moves a couple of milliseconds out of 88 or 23 out of 390, which is why nothing fits.
The lever that fixes it is the long registered width, where attention's quadratic term dominates and its share reaches roughly a fifth of the step.
That is the recommendation and it is a ruling rather than a fix, because clause 15 is committed text and the numbers exposing the gap now exist.
`KNOBS` deliberately still has no entry for candidate A.

**Both MLX claims REPRODUCED on this machine, 2026-08-20, and the second one reaches further than the pass that reported it.**
Read off the graph MLX actually built, using `export_to_dot`, which is the only way from Python to see which primitive was chosen.

| Question | Answer | How it was read |
|---|---|---|
| At which head dimensions does the fused path run | 64, 80 and 128, and no others among 32, 48, 72, 96, 160, 256 | one primitive named `ScaledDotProductAttention` where it fuses, about twenty where it does not |
| Does a gradient trace keep the fused path | No, at every one of those three | the fused primitive is absent from both the value and the gradients |
| Is that a property of the operands or of the trace | Of the TRACE | a call on frozen constants inside one trace is composed too, so both calls show a softmax |
| Does it hold inside mlx-lm's own step | Yes | the first layer's attention fuses with no tracing and is composed inside `nn.value_and_grad` |

The first fact binds the FLOOR and not the share, exactly as the outside pass said: a ladder of 128, 96, 64 and 32 would fit one line through two implementations, and a clean fit through a discontinuity is worse than a bad one because it looks fine.

The second and third are stronger than reported and they change what candidate A's floor can be.
Since the decomposition belongs to the trace, stock's attention is composed in every block and both directions, adapted or not, at any head dimension.
So the fused entry point is a genuine floor rather than a description of stock, which is the useful direction.
But a floor arm installed at the attention seam of the real step, which is what clause 19 asks for, would run stock's own computation and report a ratio of 1.0 - a number that looks measured and is an artefact of where it was measured.
Candidate A therefore needs a written bench exception like the loss candidate's, or no credited ratio at all, and that is a ruling rather than a fix.

**RULED 2026-08-20 night, after two Codex BLOCKER rounds on a bench-exception draft: candidate A gets NO floor and NO credited ratio anywhere.**
The bench exception was drafted, reviewed twice, and rejected on substance: it pairs an in-step share with a bench ratio, which breaks clause 18's exact identity (88 against 86 on worked numbers), needs `R` contexts Amendment 5 never defined, and its "fused forward plus stock backward" arm is a mechanism no seam can produce.
What replaces it is a ceiling: A's best possible score is `U(w) = 1/(1 - f_A(w))` whatever its ratio, it is excluded by arithmetic when a valid share puts `U` at or below 1.10 at either width (its proxy shares of 0.042 and 0.087 are both below the `f = 1/11` line, so this probably resolves cleanly), excluded by dominance when the L/Q winner beats `min_w U` resolvably, and otherwise the profile returns the new registered terminal UNRESOLVED rather than guessing.
Amendment 6 (re-cut step 5) carries the text.

Four properties every knob must hold, each checked by a test rather than assumed:
the output shape does not move, so no downstream op changes shape;
the launch count does not move, so no dispatch is deleted;
the dial is applied at EVERY arm including `phi = 1.0`, so the scaffold enters every arm rather than only the small ones;
and the backward keeps its structure, because the vjp of a slice scatters into a zeros tensor of the original shape.

**Compilation goes back on, and that is what removes the last bias.**
The marks forced `mx.disable_compile()` because MLX refuses an eval inside a compiled step, which is the whole of Amendment 4.
A knob needs no eval, so the profile can run compiled, which is the configuration the product actually runs.
It also removes the scaffold: `mx.compile` traces the Python execution and records only the MLX operations, so a knob's Python frames and its hoisted slicing vanish from the timed graph.
That is exactly the 2.33 ms bias the loss knob had to report against stock when it ran uncompiled.

**One compiled step object per arm, built and warmed with its seam installed, then interleaved with no seam installed at all.**
This is the hazard that would otherwise sink the whole design: `mx.compile` caches the traced graph, so swapping a monkeypatched callable between arms would leave the second arm running the first arm's graph.
Building a separate compiled object per arm gives each arm its own captured graph, makes the seam irrelevant once tracing is done, and is why the timed rounds carry no installer at all.

Confirmed on the installed MLX 0.32.0 rather than assumed, 2026-08-20: two distinct compiled closures each trace exactly once and cache separately, and calling one a second time does not re-trace it.
Building and discarding compiled objects in a loop also returned each one's own correct result, so the cache-collision worry did not reproduce; all arms are still held alive together, which is the conservative arrangement and costs nothing.
What this does NOT prove is the ordering: the trace happens at an object's first call, not at its construction, so a build-all-then-install-later mistake would bake the wrong graph, and that is why it gets its own test rather than a comment.

**And the experiment passed for a reason narrower than it looked, which the outside pass caught.**
MLX keys its compile cache on the underlying callable, not on the wrapper `mx.compile` hands back, so the experiment succeeded only because each arm happened to build a fresh closure.
Wrapping one shared step function twice would collide.
Three constraints therefore become registered requirements rather than implementation taste:

| Requirement | The failure it prevents |
|---|---|
| Every (arm, width) pair owns its own freshly built closure, retained for the whole run | Two arms sharing one traced graph, so a dial silently does nothing |
| The optimizer's state is initialised and evaluated BEFORE any arm is built | Adam's first update grows the captured state tree and forces a second trace, which can land after the seam is gone and bake stock into that arm |
| A trace counter refuses the run if any trace happens during the timed rounds | Every remaining version of the same fault, including a width change retracing an arm whose seam has been removed |

The third one is the general guard and it is why the other two do not have to be exhaustive: a retrace during timing is always a fault, whatever caused it.

Two prices, both measured and both registered as bands before the run:
the scaffold price, `T(phi = 1.0)` against stock with no seam, which must clear a registered limit or the reading is refused;
and the linearity gate, an R-squared floor and a maximum residual as a fraction of the slope, which refuses a candidate whose knob is not a dial.

**One price at one setting is not enough, and the launch-count test is weaker than its name.**
A scaffold whose own cost changes with the dial - the slice, the pad, the backward scatter - enters the slope itself, and a single check at the full setting cannot see it.
So the scaffold is priced at every setting and fitted in its own right, and its slope must be small against the knob's slope, not merely its offset small against stock.
Separately, the seam counts PYTHON CALLS, not Metal dispatches, so the property it verifies is "the number of times the operation was invoked does not change", which is what a stub would break and is not the same as proving no kernel was added.
That is written as what it is rather than called a launch count.

One cross-check, and it is the only independent one the amendment has.
The fitted intercept `a` is the step with the candidate's scaling arithmetic driven to zero, and the ablation measures the step with the candidate removed outright.
Their difference is the non-scaling residue, which for attention is the softmax and the mask work a head-dimension dial cannot see.
Reported, not corrected, and it bounds how much the slope understates a fused replacement.

**The cross-check only works if both halves come from the same regime, and today they do not (ruled 2026-08-20).**
Every number this plan cites was measured with compilation OFF, because the shared timing helper the probes all import calls `mx.disable_compile()` on every step.
Turning compilation on is therefore not a free improvement: it re-opens the ablation figure, the 0.155 ms resolution floor the tie band rests on, and the critical-path check that made attention's share a unique quantity.
So the ablation, the resolution probe and the uniqueness probe are re-run compiled before the amendment is written, and if any of their numbers move, the evidence text moves with them.
A slope with no independent check is how the marked instrument survived as long as it did.

### What the validation run has to settle, beyond the knobs themselves

Three quantities the old design had and the new one lost, each measured in the validation window rather than assumed.

**The tie band's resolution floor (ruled 2026-08-20).**
Section 4.3 calls a two-point difference in gain a tie, which demands the harness resolve roughly one millisecond.
Uncompiled, two identical arms sat 0.155 ms apart and cleared it comfortably.
Compilation makes every step faster, so the same jitter is a larger share of a smaller number and two points of gain is fewer absolute milliseconds.
The validation run therefore carries an identical-arms pair, and Amendment 5 registers the refusal now: if the measured floor exceeds what two points of gain demands at that cell, the profile reports the ordering as unresolved rather than declaring a tie.
Without it the tie-break falls through to table order, and the first row of a table picks the sprint's first operation.

**The memory budget's source.**
The previous design budgeted each cell from the fenced pass, because fencing at the head forced every logit resident at once.
There is no fenced pass now, so that rule has no referent.
The knobs bring their own peak: a second copy of the head weight for the vocabulary dial, and one live compiled step object per arm.
The validation run measures the peak with the knobs installed at both widths, and the binding run's declared budget comes from that measurement.

**The window budget, and a refusal attached to it.**
Five cells at roughly thirteen arms and eight passes each is about 520 steps plus a graph compilation per arm.
The validation run measures per-step and per-compile time at both widths, the binding window is computed from it, and if the computed budget exceeds the window granted, the binding run does not start rather than being killed part-way.

### The marks are kept and demoted

`bench/profile_instrument.py` stays, and stops being an attribution instrument.
Its counts are exact and its identity check is real: the count asymmetry it verifies (attention fires `depth` times forward and `adapted` times backward, the projections `7*adapted - 3` times backward) is what caught a seam that never fired and a test that left gradient checkpointing installed for the whole session.
So the harness keeps one structural pass per cell that asserts counts and gradient identity, and takes no timing from it.
AGENTS.md already leads its entry with the warning; the entry gains the demotion.

### The corpus: two widths, one pinner

mlx-lm pads each batch to one plus the next multiple of 32 above its own longest row, so batch width is a property of the data and `max_seq_length` is a cap this corpus never reaches.
Dolly's widest band that can fill 1024 training rows is 160 tokens.

**One corpus, two bands (ruled 2026-08-20).**
The first design drew the short cell from Dolly and the long cell from UltraChat, and that quietly broke the thing two widths exist to establish.
The loss candidate's floor is defined as the same work on only the supervised rows, so the supervised fraction sets that candidate's ceiling directly, and two datasets have two supervised fractions.
The comparison would have measured dataset and width together and reported it as width.

| Cell | Corpus | Band | Why |
|---|---|---|---|
| Short | UltraChat 200k | one 32-token band, the shortest that fills 1024 rows | width is the only thing that separates this cell from the next one |
| Long | UltraChat 200k | one 32-token band at or above 1024 | the width where attention's quadratic term starts to dominate |

Dolly leaves the profile.
Registered with the band rather than left selectable: the dataset revision, the config name, the chat adapter, both band edges, and the measured supervised fraction at each, because every one of those can move a candidate and none of them is width.

`bench/pin_dolly.py` already carries the band arithmetic, the seeded selection, the distribution report and the split writer.
It is generalised into a corpus-agnostic pinner with a per-corpus fetch and row-to-chat adapter, so one band rule serves both bands and no corpus gets its own private arithmetic.

The two-width ruling applies to the DECIDING cell only.
Cells A, C and D decide nothing under section 4.3 and stay at the short width, which is stated in the record rather than left to be noticed.

### The floors: `bench/ceiling_sweep.py`

Each floor is now measured with the same dial as its share, so `r` is a slope over a slope.

| Candidate | Floor | Measured how |
|---|---|---|
| L | the same operations on only the supervised rows, plus one matmul at the head's backward shape | the vocabulary dial, on the bench, under the written exception below |
| A | `mx.fast.scaled_dot_product_attention` installed at the attention seam | the same dial, INSIDE the real step, forward and backward priced separately |
| Q | the dense fp16 matmul installed at the projection seam | the same dial, INSIDE the real step, at six shapes and both orientations |

Two of the three floors are real callables, so they are installed at the same seam the knob uses and dialled inside the same step, which is what makes `f` and `r` the same kind of quantity.
The loss candidate's floor is a streamed kernel that does not exist, so it stays on the bench and carries a written exception saying that its ratio is assumed to transfer into the step while the other two no longer assume it.

The credited ratio stays `ratio_lo`, smallest numerator over largest denominator.

**The existing interleaver cannot do this, and that is a code change rather than a wording change.**
`interleave.interleaved_samples` takes exactly two arms, and a four-point ladder against a four-point floor ladder is eight.
It is generalised to n arms, keeping the drift canary and the per-round structure, so `ratio_lo` is taken over a distribution of per-round slopes rather than over one point estimate.
That generalisation is the shared sampler, so it is made once in `interleave.py` and not copied, which is the ADR 0004 two-halves mistake this repository has already made once.

**The kill rule needs its reduction written down.**
Registered, it removes candidate Q if stock is within 1.10 of the dense ceiling at 4 of the 5 shapes, which is "all but one shape".
Six shapes makes that 5 of 6, restated as "all but one" rather than as a fraction, because 5 of 6 is a tighter fraction than 4 of 5 and inheriting the old number silently would have loosened it instead.
Two reductions the registered text never names and Amendment 5 now supplies: how a shape's forward and backward ratios reduce to the one scalar the rule reads, and how the two widths reduce, both fixed before the numbers exist.

**SETTLED 2026-08-20: candidate A's forward is NOT already at its own floor.**
The dispatch reads as the investigation said: `mlx_lm.models.base.scaled_dot_product_attention` calls `mx.fast.scaled_dot_product_attention` whenever there is no quantized cache, and a training step has no cache at all.
What the reading missed is that a training step is a gradient trace, and MLX composes that call inside one.
Measured inside mlx-lm's own step, the same layer's attention fuses without tracing and is composed into matmuls and a softmax under `nn.value_and_grad`.
So stock's forward is the composed path, A's forward ratio is not 1.0, and section 4.2's floor is a real floor in both directions.
What replaces this investigation is the harder question above: that floor cannot be measured where clause 19 puts it.

### Amendment 5, committed alone, before any code moves

Grown from ten clauses to twenty-five: seven added by the engineering review of 2026-08-20 and eight more by the independent pass that followed it.

Two of them are contradictions inside my own draft rather than gaps in the registered text, and they are called out here so they are not read as external findings.
Clause 2 asked for three disjoint shares summing to at most one while clause 13 deliberately puts the tied output head inside two candidates at once, so the sum check as written would have counted the head twice and produced a remainder that means nothing.
Clause 24 is worse, because it is a bug in committed code rather than in a draft: the shipping floor is tested against the largest gain and the winner is then chosen from inside the tie band, so a candidate below the floor can be selected by the footprint tie-break.
That one is fixed whatever happens to the rest of this increment.

| # | Clause | Why it must be written before the run |
|---|---|---|
| 1 | The share is the knob slope, `f := b/T_stock`, replacing section 3.3's marked span | The registered method is measurably wrong by up to a factor of six and its error tracks mark count |
| 2 | Section 3.3's span-plus-remainder reconciliation is void, replaced by: each share is a fraction, the three are disjoint, and their sum plus the remainder is 1 | With no spans there is nothing to reconcile, and the sum check is the one the marks could never pass |
| 3 | The marks are retained as a structural check on counts and gradient identity, never as a timing instrument | The counts are exact and have already caught two real faults |
| 4 | Amendment 4 is superseded in part: the profile runs COMPILED, and the compile-transfer band no longer gates it | Amendment 4 turned compile off only because the marks needed an eval inside the step, and it separately registers a band that a measured 1.194 ratio would breach |
| 5 | The instrument-cost band becomes the scaffold-price band, one limit per knob | A knob's `phi = 1.0` arm against stock is the honest measure of what the instrument costs |
| 6 | The linearity gate: an R-squared floor and a maximum residual, refusing a candidate whose knob is not a dial | Without it a nonlinear knob produces a slope that means nothing |
| 7 | Shape S6, k_proj and v_proj at (1024, 2560) | They exist in the step and are counted in Q's share, and the registered list omits them |
| 8 | The kill rule restated at 5 of 6 shapes | `KILL_SHAPES` is 4 and adding S6 would silently turn 4-of-5 into a looser 4-of-6; 5-of-6 preserves the registered fraction |
| 9 | The floor is a slope too, and `r := b_stock/b_floor` | This is the pairing under which the registered gain formula is literally correct |
| 10 | Candidate A's floor arithmetic: 36 forwards and 16 backwards, not 3 times 36, and forward and backward priced separately | Under LoRA only the adapted blocks run a backward, and the investigation above may put the forward at ratio 1.0 |
| 11 | The reduction of two directional ratios to the one scalar the kill rule reads | Nothing registers that reduction today |
| 12 | The footprint tie-break's source, reduction and sign | It wants a byte delta for kernels that do not exist; the floor arms' own deltas are the only measurable source and that is a statement about the floor |
| 13 | The L and Q overlap: never composed, and Q's knob dials the head with the projections | The tied output head is a quantized matmul and sits in both registered region sets |
| 14 | Two registered widths, the corpus and band for each, and selection by highest minimum gain across them | Attention's share moves four-fold across the width range in dispute, so choosing one width would choose the winner |
| 15 | The attention dial's selection criterion: smallest scaffold price against stock, most complete dial breaking a tie | Attention admits three dials that give different shares, and the criterion has to exist before the three numbers do |
| 16 | The tie band's resolution floor, measured compiled at the target cell, with a refusal if it exceeds what two points of gain demands | The floor was measured uncompiled and compilation shrinks the step, so nothing currently says the tie band is resolvable at all |
| 17 | The memory budget's source: the peak measured with knobs installed at the validation run | The clause it replaces budgeted from a fenced pass that no longer exists |
| 18 | A retune is credited with its slope, a rewrite with its slope plus its measured residue, and the type is read off section 4.2's own names | One rule for both would under-credit the two rewrite candidates by more than the band that counts as a tie |
| 19 | The floor is measured inside the real step at the same seam as the knob, with the loss candidate's bench measurement as a named exception | `f` is measured in the crowd and `r` was to be measured alone, and nothing proved the ratio survives the crossing |
| 20 | One corpus, two bands, with revision, config, chat adapter, both band edges and the supervised fraction at each all registered | Two datasets differ in the supervised fraction, which IS the loss candidate's floor, so the width comparison would have carried a second variable |
| 21 | The numeric limits come from an instrument-only stage that dials no candidate and computes no share | An amendment with no numbers lets the code or the first result decide what acceptable means |
| 22 | The sum check runs over a disjoint partition of the step, which is NOT the same as the candidates' region sets | Clause 2 demanded disjoint shares while clause 13 puts the tied head in two candidates at once, so as written the check double-counts it |
| 23 | The kill rule reads "all but one shape", with the reduction across direction and across width both written | 5 of 6 is a tighter fraction than 4 of 5, and neither reduction exists in the registered text at all |
| 24 | The shipping floor applies to the SELECTED candidate, not to the largest gain | Reproduced in the committed selector: a candidate below the floor can win on the footprint tie-break because the floor is only ever tested against the leader |
| 25 | Trace discipline: a fresh retained closure per arm and per width, optimizer state initialised first, and a trace counter that refuses any trace during the timed rounds | MLX keys its compile cache on the underlying callable, so a shared closure or a late retrace silently turns a dialled arm back into stock |

### Sequencing (commit per green step)

| # | Step | Verification | Blocked on |
|---|---|---|---|
| 0 | `/plan-eng-review` on this section, plus an independent Codex pass | LANDED 2026-08-20: four decisions ruled by the review, fifteen findings from the outside pass, eight of them P0, four more decisions ruled off them | Nothing |
| 1 | Amendment 5, twenty-five clauses, committed alone with no code in the commit | LANDED `c13445e`, plus `820acab` for TODOS and AGENTS. Five review rounds against an independent model: round three found the arithmetic broken, round four found two regressions the repair introduced, round five found one clause admitting two answers on one input, round six cleared it. Each finding reproduced before it was fixed | Step 0 |
| 2 | `bench/profile_knobs.py`: the arm runner, the fit, the bands, the three knobs, the trace discipline; the six probes rewired to import from it | Device-free: a known-answer line recovers slope, intercept and R-squared; the arm plan; both bands; a knob whose fit passes while its scaffold fails, and the reverse; every refusal. Live: each seam fires, output shape and Python call count hold across phi, `phi = 1.0` is bit-identical to stock in loss AND full gradient, no seam is installed during the timed rounds, every patched class is restored, and the trace counter refuses a deliberately induced retrace | Step 1 |
| 3 | Three attention dials built, the registered criterion applied, and both MLX claims reproduced on this machine | LANDED. Both MLX claims reproduced by reading the built graph, and the second reaches further than reported: the decomposition belongs to the gradient TRACE, so stock is composed in every block and both directions and clause 19's floor arm cannot reach the fused path at all. Three dials built and every knob property asserted. Both rulings taken the same night: the dial selection binds at the long width, and candidate A takes the ceiling-and-UNRESOLVED semantics in place of any floor | Step 2 |
| 4 | LANDED `d919aa5`. RE-CUT 2026-08-20 night: the shipping-floor fix, `select_first_operation` filters below-1.10 candidates BEFORE the tie band and the tie-breaks, per Amendment 5 clause 24 | Two new regressions FAIL first (the footprint tie-break picking a below-floor candidate, and table order doing the same), then pass; below-floor candidates stay in the ranked evidence but never in `selected` or `tied_with`; the all-below branch still reports the top two; a gain of exactly 1.10 stays eligible; TODOS entry retired and AGENTS.md's fault line updated | Step 3 |
| 5 | LANDED `cb4b9cd`, committed ALONE, NOT CLEARED for binding use. Eight clauses 26-33. Eleven Codex review rounds plus three specialist passes (falsification, implementation, propagation). Clause 31 was rewritten FOUR times: one terminal for all sites reversed committed clause 16; a largest-`R` governing context is not an ordering because the demand is a time; a scalar demand assumes uncertainty transfers one-for-one and it does not, since the share takes the median numerator while `ratio_lo` takes smallest-over-largest, amplifying a floor movement by `M/N`, reproduced as exactly five. Version four bounds each rule's ACTION over its measurements' box, writing the credit as a saving `K = M(1 - F/N)`. Clause 32 reversed a process fault by restating Amendment 5's thresholds rather than editing them in place; the diff is a pure insertion. Clause 33 fixes units, ordering, both artifact schemas, and what steps 7-10 must produce. `R`'s reading is resolved to the of-medians one and clause 16's `2R` is replaced by a registered `alpha = 0.05` and a measured critical value, because `2R` has a 17.7% false-separation rate under the ideal null | BLOCKER CLOSED 2026-08-21 by Vlad's ruling, re-scope then reduce; Amendment 7 below carries it, and clause 31 v4's exactness claim was withdrawn in `9f00e06` after a box-design pass reproduced an interior minimum | Step 4 |
| 6 | LANDED `412bc59`. The corpus pinner generalised, UltraChat fetched, BOTH bands pinned from it, Dolly retired from the profile. Both bands DERIVED by the registered rule at revision `8049631c`: short 64 (width 65, supervised median 0.556), long 1056 (lower edge exactly 1024, width 1057, supervised median 0.870), zero rows dropped by the adapter | Both bands hold 1024 rows, both splits hash, the supervised fraction is measured and recorded at each, the distribution report names what was dropped | Step 1 |
| 7 | `bench/profile_stock.py` rewired: knob arms replace marked modes, the marked pass demoted to a structural check, two widths at the deciding cell, `binding_blockers` rewritten clause for clause | The existing 85 tests move with it; every gate the old function held has a named replacement or a written reason it is gone; the 3-candidate by 2-width matrix is required complete and context-equal before any reading is derived, with candidate A's typed absence as a legal completion state per Amendment 6 | Steps 2, 5 and 6 |
| 8 | INTERLEAVER HALF LANDED `4f3d9b5` (n-arm `interleaved_arms` with rotation, two-arm wrapper unchanged in order). `bench/profile_rules.py` (the rest) and `bench/interleave.py`: a width dimension, slope-based `r`, the retune-versus-rewrite credit split, highest minimum gain across two widths, the kill rule's two reductions, the linearity gate, the tie-band refusal, the A-ceiling and UNRESOLVED verdict, absence-typed inputs; the interleaver generalised from two arms to n with canary and per-round structure kept | Device-free tests for every rule, including two widths that disagree, a maximin winner that wins at neither, the ceiling excluding A at `f <= 1/11`, an UNRESOLVED verdict when the winner cannot beat the ceiling, and the interleaver's existing two-arm callers unchanged in behaviour | Step 5 |
| 9 | `bench/ceiling_sweep.py`: candidate Q's dense floor installed at the same seam as its knob and dialled inside the real step, six shapes, both orientations; candidate L's bench exception recorded; candidate A has NO floor arm, per Amendment 6 | Device-free tests for the reductions and the kill rule; live tests that each floor arm runs the operation it claims | Steps 3, 7 and 8 |
| 10 | Validation stage one, instrument only: identical arms and scaffold-only arms at the target, no candidate dialled, no share computed | The resolution floor, the scaffold noise and the peak footprint, all measured with no candidate number in existence | VLAD GO #1 |
| 11 | Amendment 5 addendum: the measured resolution floors, committed alone, derived from step 10 and from nothing else. NO LONGER ONE NUMBER: Amendment 6 registers a third context, candidate L's bench arrangement, so the addendum carries `R(w)` and `R_bench(w)` per cell and width | The addendum carries both floors, keyed by context, and the measurement they came from, and nothing else enters it; a floor of zero or a non-finite floor REFUSES it | Step 10 |
| 12 | Validation stage two: the knobs at the 4B target, both widths, including clause 26's dial selection at the long width | Every surviving knob linear at 4B, every scaffold slope small against its knob slope, the dial named or the no-dial branch taken, per-step and per-compile time measured so the binding window is budgeted from evidence | VLAD GO #2 |
| 13 | Binding profile run, both widths at the deciding cell, three reporting cells at the short width | Blocked on step 12 PASSING and not only on approval; closing idle gate, eligible rounds, every share a fraction, the disjoint partition summing at or below one; refuses to start if its computed budget exceeds the granted window | Step 12 passing, then VLAD GO #3 |
| 14 | Ceiling sweep run | The same discipline; the kill rule reported whatever it says | Step 9, then VLAD GO #4 |
| 15 | `--decide` writes the decision artifact and the ruling lands in this plan | The artifact names the recording hash, the sweep hash and the digest of the rule code that ruled; an UNRESOLVED verdict is a legal ruling and names every absent candidate with its typed reason | Steps 13 and 14 |

The old step 4 (everything re-run at 0.6B compiled) is DEMOTED from the sequence: its numbers are exploratory evidence, not validation, and the pieces that mattered were absorbed into steps 3 and 12.
The re-cut's reasons, in one line each: the shipping-floor fix is the only work with zero open questions; the amendment unblocks every rule that follows; the corpus defines the widths everything is measured at.

### Verification

- Full suite green with Metal and under `KV_FORCE_NO_METAL=1` before every commit, and the no-device audit stays at zero unprotected tests.
- Every knob carries a test that its `phi = 1.0` arm is bit-identical to stock, which is the check that separates a dial from a different computation.
- Every knob carries a test that the launch count does not move across phi, which is the check that separates a retune knob from a deletion.
- The compiled-per-arm design carries a test that two arms at different phi produce different times, which is the check that MLX did not hand the second arm the first arm's cached graph.
- It carries a second test that the seam is installed at each compiled object's FIRST CALL rather than merely at build time, because that ordering is what decides which graph gets baked, and a build-then-install-later mistake would silently bake the wrong one.
- It carries a third test that no seam is installed at all during the timed rounds, because a leaked installation would cost time inside the measurement and be invisible in the result.
- Every patched class is asserted restored at the end of its test, which is a regression: a leaked class-level patch already ran for a whole test session on this branch and produced counts nobody could explain.
- The fit carries a known-answer test: a line of known slope with known noise recovers its slope, its intercept and its R-squared.
- The selection rule carries tests for two widths that disagree on ordering, for a highest-minimum winner that wins outright at neither width, and for the interaction between that rule and the two-point tie band.
- A trace counter refuses the run if any compilation trace happens during the timed rounds, and a test induces one deliberately to prove the counter fires.
- The `phi = 1.0` identity test compares the full gradient tree as well as the loss, because a knob that changes only a gradient would pass a loss-only check.
- The selection rule carries a regression test that no tie-break of any kind can return a candidate below the shipping floor, which is a bug in code that is already committed.
- The reading is derived only from a complete three-candidate by two-width matrix whose entries share one context, so a short-width share can never be paired with a long-width floor.
- Validation stage one's acceptance is stated before it is armed: a resolution floor, a scaffold noise figure and a peak footprint, with no candidate dialled and no share computed.
- Validation stage two's acceptance is stated before it is armed: three linear knobs, three scaffold slopes small against their knob slopes, and measured per-step and per-compile time at both widths.
- No GPU window is armed without Vlad's explicit go for that run, at most one armed job at a time, on AC power.

### NOT in scope, deliberately

| Deferred | Why |
|---|---|
| The end-to-end harness's own registered 2048 | The two-width ruling reaches it and it is a committed TODO, but it belongs to Day 5's increment and changing it now would move two harnesses at once |
| Cells A, C and D at the long width | They decide nothing under section 4.3, and the reduction is recorded rather than hidden |
| Removing `bench/profile_instrument.py` | Its counts are load-bearing and have caught two real faults; it is demoted in writing, not deleted |
| Reading a Metal GPU trace for per-kernel attribution | Zero perturbation, but it needs Xcode tooling and its kernel names do not map onto the three candidate regions |
| Building any candidate kernel | The profile chooses, Day 2 builds |

### Files

- New: `bench/profile_knobs.py`, `bench/ceiling_sweep.py`, `bench/.data/ultrachat-<band>/`, `tests/test_profile_knobs.py`, `tests/test_profile_knobs_live.py`, `tests/test_ceiling_sweep.py`.
- Modified: `bench/profile_stock.py`, `bench/profile_rules.py` (including the shipping-floor bug, which is fixed regardless of this increment), `bench/interleave.py` (two arms to n, in the shared sampler, not a copy), `bench/pin_dolly.py` (generalised to a corpus-agnostic pinner, ONE band arithmetic, not a second script), `bench/profile_instrument.py` (demotion only), all six probes under `bench/mlx_probes/` (they stop owning the shared runner and the shared fitter and import them from `profile_knobs.py`), the prereg (Amendment 5 and its addendum), `AGENTS.md`, `TODOS.md`, this plan.
- Retired from the profile, not deleted: the pinned Dolly slice, which stays in the tree for the end-to-end harness that still references it.
- One vocabulary for the scaffold arm: the probes currently call the same thing `noslice`, `stock+reductions` and `phi=1.00`, and it becomes registered text, so it gets one name before it does.
- Reused, never reinvented: `metalrunner/seams.py` for every installation and count, `bench/interleave.py` for every two-arm comparison, `bench/machine_state.py` and `bench/memory_guard.py` and `bench/detached_run.py` for the run discipline, and the shared child runner extracted at step 2 of the previous increment.

---

## Next increment, re-cut 2026-08-21: Amendment 7, the profile certifies less and the box becomes affordable

This section supersedes steps 9 through 15 of the sequencing above and replaces them with its own order.
Steps 7 and 8 stay as cut and gain the scope changes below; step 5 stays landed and its open blocker is closed here.

### Context: why this re-cut exists

Amendment 6 registered clause 31's action principle, which supports a decision only where it holds everywhere inside the uncertainty box around that decision's own measurements.
A design pass then priced that box and found it both undefined and unaffordable.
The box has to be over per-arm, per-round samples across a 45-arm manifest at two widths, and the recommended schedule was about 43 hours of pure timing, excluding warm-ups, compilation, the floors, the sweep and every idle refusal.
A structural problem sat beside the cost: the calibration stage runs before the attention dial is selected and before the floors and the sweep exist, so it could not produce the authoritative box at any price.

Vlad ruled 2026-08-21: re-scope what the profile decides, then reduce the manifest to the arms the surviving rules read.
Three follow-on rulings were taken the same day.
Every figure below is arithmetic over Amendment 6's own cost model, which prices one arm at one round at one replica at 4.4 seconds, 0.4 at the short width and 4.0 at the long.

### The four rulings

| # | Question | Ruling |
|---|---|---|
| 1 | What does the profile certify | Only what names the first operation: the L versus Q ordering and the shipping floor. Everything else is measured, published with its own uncertainty, and gates nothing |
| 2 | What is candidate A's role | Reported, never a gate. Its dial is named by rule, its ceiling is published, and a ceiling exceeding the winner's score is an open question recorded against the winner rather than a refusal to select |
| 3 | Does the kill rule certify | No. Its per-shape ratios are published and the terminal is decided by the score and the shipping floor alone |
| 4 | How many null blocks | 39, against a registered rate that needs 19 and a design schedule that assumed 79 |

### What happens to each of clause 31's eight sites

| Site | Fate | Why |
|---|---|---|
| Clause 24's shipping floor | CERTIFIED | it decides whether building anything is worth the sprint |
| Clause 16's per-width pair | CERTIFIED | this is the L versus Q choice itself |
| Clause 16's two-point boundary | CERTIFIED | band membership is part of that same choice |
| Clause 27's route 1 | REPORTED | it can only exclude candidate A, and candidate A no longer gates |
| Clause 27's route 2 | REPORTED | same |
| Clause 23's kill rule | REPORTED | ruling 3 |
| Section 4.3's retained order | REPORTED | it sets Day 3's build order, not Day 2's |
| Clause 15's dial price | RETIRED | ruling 2 names the dial by rule, so there is no price comparison left to make |

Three certified margin families remain, all of them over candidates L and Q at two widths.

### Why certifying less also tightens what survives

The calibration is a grouped maximum: each null block's score is the largest standardised leaf contrast over the predeclared vector, and the demand is an order statistic of those block scores.
So every leaf in the certified vector can only raise the per-block maximum, and every leaf removed can only lower it.
Dropping a certification therefore buys twice, once in hours and once in a smaller demand that the surviving decisions have to clear.
Two consequences follow directly and neither is a judgement call.

Candidate A's arms leave the certified vector entirely, because no certified margin reads them.
They are timed once in the binding run and never re-timed across the calibration blocks, which is where their cost used to sit.

Q's ceiling sweep needs ONE resolution context rather than one per shape and per direction, because its only certified consumer is now the collapsed `ratio_lo`, which is a single quantity.
Under the kill rule each shape was placed against 1.10 on its own, and that is what forced a context apiece.

### The reduced manifest, and what it costs

| Arms | Candidate | Widths | In the certified box |
|---|---|---|---|
| 9 | L, four knob settings, four scaffold settings, one ablation | both | yes |
| 8 | Q, four knob settings, four scaffold settings, no ablation because it is a retune | both | yes |
| 1 | stock, no seam | both | yes |
| 9 | A on the kv-length dial, four knob, four scaffold, one ablation | long only, where the dial fits | no, reported only |

| Quantity | Full scope | Re-scoped |
|---|---|---|
| Arms in the calibration manifest | 45, both widths | 18, both widths |
| One null block | about 33 minutes | 13.2 minutes |
| Blocks, pilot plus fresh | 79 | 20 plus 19, so 39 |
| Calibration total, STEP CONTEXT ONLY | 43.45 hours | **8.58 hours** |
| The binding run's own timing pass over all 27 arms | not separately stated | 9.6 minutes |

That is a 5.06x reduction, and it fits one granted overnight window rather than a week of them.

### The dial is named by rule, and that closes the ordering problem

Candidate A's dial becomes kv-length, named by clause 15's own registered completeness tie-break rather than by a measured price.
This is not a new rule invented to save time.
Clause 26 already records that where every scaffold slope sits inside the machine's jitter the clamp records them all as zero, every price ties, and the tie-break names kv-length at both widths; the proxy smoke is exactly that case, with the best knob fit reaching 0.946 against a gate of 0.99.
Clause 28 already records that kv-length is the only dial placing all four ladder settings on one implementation.

The structural problem dissolves with it.
The calibration stage no longer runs before a selection it cannot see, because there is no selection left to make.

### What this gives up, stated plainly

The profile can no longer decline to select because candidate A might have been better.
It selects the L or Q winner and records candidate A's ceiling beside it, so a high ceiling becomes a known risk carried into Day 2 rather than a refusal.
This is the honest trade: candidate A could never be ruled IN under Amendment 6 either, so what is lost is a veto and not a candidate.

A killed Q is no longer removed by its own rule.
A Q with five slow shapes beside one fast one could score well on the collapsed ratio and be selected, while the per-shape evidence in the report says the opposite and no rule reads it.
Measured over the registered six shapes, that band opens at a concentration of 1.910199 when the fast shape's ratio is 4.0, and it WIDENS as that shape gets faster, reaching 1.297 at a ratio of 100.
A candidate Q whose collapsed ratio is at or below the kill line still cannot ship, which is now a proof rather than an example: the gain is increasing in the share with limit the ratio, so a ratio at or below 1.10 forces a score below 1.10 for every valid share.

The dial's bias changes meaning rather than disappearing.
If kv-length's scaffold is genuinely the most expensive of the three at the long width, candidate A's share carries that cost and reads high, which inflates its ceiling.
Under Amendment 6 an inflated ceiling made exclusion harder, which was the safe direction; under ruling 2 it only makes candidate A look better in the report than it is, so the report has to say so where the number appears.

The demand comes from 39 blocks rather than 79, so it estimates the true rate's critical value less precisely.
The registered rate itself is unaffected, because 39 is comfortably above the 19 that rate needs.

Clause 27's terminal table loses UNRESOLVED and keeps three terminals: SELECTED, the no-selection record, and INCOMPLETE.
Totality and disjointness have to be re-verified over the reduced state space rather than assumed to survive.

### Sequencing (commit per green step)

| # | Step | Verification | Blocked on |
|---|---|---|---|
| 5b | LANDED `4a04bdb`, committed ALONE, 368 insertions and no deletions, clauses 34-39. One adversarial Codex round returned 16 defects, every one reproduced by execution before it was admitted and every one applied, plus three found by self-attack first. The four that mattered: clause 36 computed its whole concession through committed code registering five shapes and a four-shape kill while Amendment 5 registers six and five-to-kill, moving the crossing from 1.528 to 1.910199; the terminal enumeration invented a kill flag for candidate L, which has no kill rule, so the legal space is 864 states and not 1728; the precedence reversal moves 8 states from SELECTED to INCOMPLETE, strictly worse and unstated; and clause 15's criterion run on the registered prices returns head-dim-qkv, so naming kv-length is a REVERSAL and not an application. 59 numeric claims audited by execution, all hold | Done |
| 7 | LANDED, five commits: `994c545` candidate L's true ablation; `3477181` clause 26's scaffold cases, the sign precondition on the median slope, the `10R` excursion on both estimators and clause 33's residue fault; `0c1af7c` the pure width reducer; `9bc94ab` Amendment 9; `b896945` the rewire itself - Amendment 8's manifest replacing the mode manifest, the reducer replacing `cell_reading`, `binding_blockers` rewritten gate by gate, schema 2, and a rule-facing `profile_matrix` at the deciding cell | suite 1904 to 1936; every removed gate carries a written reason and a test that no blocker still names it; `--calibrate` gone with its band, `--decide` refusing schema 2 until step 8 | Step 5b |
| 8 | LANDED `5dca9e9`. The ruling rewritten into the SAVING domain: three certified sites and four reported, three terminals with UNRESOLVED retired, candidate A share-only and gating nothing, a killed Q as a recorded flag, six shapes and five-to-kill with both of clause 8's reductions, clause 12's missing second condition, `gain` strict at a share of one, and the three rules Amendment 5 voided removed rather than deprecated | suite 1936 to 1976; clause 38's table verified by ENUMERATION over 432 legal states on the pair of terminal AND payload, with a deliberately reinstated candidate-A gate moving 28 states to show the model is not vacuous | Step 5b |
| 9 | LANDED, five commits: `581688d` Amendment 10; `656d5e3` candidate Q's dense floor IN the step; `3331bf3` Amendment 11; `1836a9c` the kill bench; `9ec6fdf` Amendment 12; `c1450c3` candidate L's bench; `1bb912f` the sweep recording | suite 1976 to 2065; three amendments, every claim audited by execution, three of them withdrawn during the audit for not reproducing | Steps 7 and 8 |
| 9b | Amendment 13, the four step-10 obligations discharged, and the exploratory pass built | LANDED, eight commits `1df1c33` through `57442c8`: the arm price measured at the target, both benches interleaved, the seed guard, Amendment 13's six clauses, and clause 57's exploratory pass proven on the device | Step 9 |
| 10a | The exploratory pass: three passes of the binding manifest plus its pairs, then the margin | Amendment 14 and the reader LANDED `9a024eb` and `1aa6406`; the sweep's entry point and the runner are outstanding | Step 9b |
| 10b | The exploratory run itself | ARMED 2026-08-21, waiting on a strong-idle window; 5.09 hours per Amendment 15 clause 63, sizes the calibration and names no operation | Step 10a |
| 10 | The calibration stage, both widths, at the schedule clause 57's branch rule selects | 43.23 hours at the lean schedule, 76.18 as clause 37 registers it; `R` and the demand per context, no candidate dialled and no share computed | Step 10a |
| 11 | The addendum, committed alone | both floors and the demand, keyed by context, and nothing else | Step 10 |
| 12 | Validation stage two, the knobs at 4B, both widths | every surviving knob linear at 4B, every scaffold slope small against its knob slope; no dial selection to make | VLAD GO |
| 13 | Binding profile run, one 9.6-minute timing pass over 27 arms | blocked on step 12 PASSING, not only on approval | Step 12 passing, then VLAD GO |
| 14 | `--decide` writes the decision artifact and the ruling lands here | the artifact names both recording hashes and the rule-code digest, and carries candidate A's ceiling and the per-shape kill evidence as reported-not-certified | Step 13 |

Step 9 now runs BEFORE step 10, which is the reordering the structural problem forced.
The old step 14, a separate sweep run, folds into step 9's own arms because the sweep's floor is installed in the same step as the knob.

### Verification

- Full suite green with Metal and under `KV_FORCE_NO_METAL=1` before every commit, and the no-device audit stays at zero unprotected tests.
- Amendment 7's terminal table is re-verified by enumeration over the reduced state space, the way Amendment 6's 2304 states were, and the run must show every state landing on exactly one of three terminals with no state reaching SELECTED through an unresolvable comparison.
- The cost arithmetic in this section is reproduced by execution against Amendment 6's own 4.4-second model before the amendment lands, so the 8.58 hours is a computed figure and not an estimate.
- The certified vector is enumerated explicitly in the amendment, leaf by leaf, so a later reader can check that candidate A's arms are absent from it.
- Every certified margin keeps clause 31's requirement unchanged: a certified lower bound over the whole box with shared expressions kept shared, corners alone still known unsound.
- The kill rule's demotion carries a test showing the exact case it now misses, five shapes near the ceiling beside one with headroom, so the concession is pinned by execution rather than described.

### Step 7 landed 2026-08-21, and what step 8 inherits from it

The harness now measures Amendment 8's manifest, and the ruling arithmetic is the only piece of the profile still written against the old scope.

`bench/profile_rules.py` is UNCHANGED by design and is now the odd one out.
It still registers five projection shapes and `KILL_SHAPES = 4` where Amendment 5 registers six and five-to-kill; it still ranks a single share against a single credited ratio at one width; it still has an UNRESOLVED terminal Amendment 7 clause 38 removed.
Step 7 added two things to it and nothing else: `WIDTHS`, the two registered bands with their batch and operation widths, and `WIDTH_ORDER`.
`--decide` refuses a schema-2 recording rather than running that arithmetic over it, so the gap is a refusal rather than a wrong ruling.

What step 8 reads is `record["profile_matrix"]`, built at the deciding cell alone: candidates in `rules.CANDIDATES` order, widths in `rules.WIDTH_ORDER` order, each cell of the table a `reading` or a `missing_share` with a `reason_code`, and never an omission or a null.
Candidate A's short-width entry is a typed absence with `not_measured_at_this_width`, which clause 37 registers is never incompleteness.
Every entry carries `certified`, true for candidates L and Q alone, and a reported entry carries its arms' observed spread labelled as an observed spread and NOT as a bound at any registered rate.

Three readings the code had to make that the document did not supply, all recorded in the artifact rather than assumed:

| Reading | Why the document did not force it |
|---|---|
| A dial's ACTUAL fraction is derived per site and the run refuses if two sites disagree | Amendment 6 supplies no scalar-fit rule for a composite arm whose sites round differently, and inventing one would be a rule made to fit the model in front of it |
| The supervised count comes from mlx-lm's own mask applied to the batch, not from a step's return | Every arm's loss is meaningless at a dialled setting, so a count read off one would be a count of whatever that arm computed |
| Seconds become milliseconds at the single point where a timer's output becomes a sample | Clause 33 puts every rule's input in milliseconds and one artifact carrying two unit conventions is what that discipline exists to prevent |

**Two harness inputs are contracts step 7 wrote and no amendment registers, and both are named here so step 11 can register them rather than discover them.**
The plan's `resolution.floors_ms` per width and `resolution.addendum_sha256`, which the run refuses to start without, are how the committed addendum reaches the harness.
The plan's `bands` mapping, checked against each width's registered slice name, is how the two pinned corpora reach it.
Neither is a rule; both are the shape a rule's output has to take to be read, and Amendment 6 registers no addendum schema at all.

### Step 8 landed 2026-08-21, and what step 9 inherits from it

The ruling is complete and cannot be computed, because it reads one number no measurement in the tree produces yet.

`select_first_operation` consumes a credited SAVING per candidate per width, `K = M * (1 - F/N)`.
`M` and `N` come from the knob arms the binding profile already times.
`F` is the FLOOR's credited denominator, and there is no floor: candidate Q's dense fp16 comparison installed at the same seam as its dial, and candidate L's bench arrangement under clause 19's written exception, are both step 9's to build.
`--decide` therefore refuses on the input rather than on the arithmetic, and a `--decide` that supplied its own `F` would be inventing the one measurement the whole credit rests on.

What step 9 owes, beyond the arms themselves:

| Owed | Why it cannot be deferred again |
|---|---|
| Candidate Q's floor arms at six shapes and both directions | the collapsed `ratio_lo` is the only certified consumer left, so the sweep needs ONE resolution context and not one per shape and direction |
| Candidate L's bench arms | clause 21 forbids reusing a resolution floor across contexts, so the bench is a third context and its arm count is not in the eighteen |
| Both arm counts | clause 37 registers that the 8.58 hours is the STEP context alone and that step 10's window cannot be budgeted until step 9 produces these two |

**Clause 12's tie-break is unreachable and the code carries it anyway.**
Candidates L and Q are the only candidates that ever score, so any band with more than one member contains candidate L, and clause 12's second condition sends every such band to table order.
The rule is written as its own function with both conditions rather than inlined with one, so a later amendment that adds a scoreable candidate inherits the rule rather than rediscovering it.
This is what the Amendment 7 ledger means by clause 12 being "still unreachable", now true of the code as well as of the document.

**One thing the enumeration proved that reading could not.**
Clause 38's table is verified over 432 legal states on the pair of TERMINAL and PAYLOAD, because a table can hold its label invariant while the operation it names moves.
The same enumeration is checked for vacuity: reinstating a candidate-A gate deliberately moves 28 states, so the invariance is a property of the rules and not of an enumeration too coarse to see a difference.

### Step 9 landed 2026-08-21, and what step 10 inherits from it

Step 9 could not start against the registered text, and closing that took three amendments rather than one.
Each was written before the code it governs, each committed alone, and each was audited by execution with its failures named in the text rather than fixed quietly.

| Amendment | What forced it |
|---|---|
| 10 | The credited denominator is an in-step slope, so Amendment 5's cross-shape collapse has no consumer; the kill rule has no arms in the manifest; candidate Q's floor shares the step context; and the calibration is 16.07 hours of timed rounds rather than 8.58 |
| 11 | Amendment 10's own factor-of-two-hundred cotangent claim was a measurement artefact, comparing a subtracted quantity against an unsubtracted one, and it had to be withdrawn a day after it landed |
| 12 | Candidate L's bench needs a no-dial baseline of its own implementation, a ruling on whether the extra matmul is dialled, and a replacement for the round pairing clause 19 breaks |

**Three claims of my own were wrong and are withdrawn in the committed text rather than replaced quietly.**
An identity between two statistics that read 3.000 against 2.500 on exact fits.
An impossibility that is not impossible, because clause 8 consumes a sum and not its terms.
An authority in clause 31 that clause 31 does not grant.

**What step 10 inherits, and none of it is optional.**

| Owed | Why it cannot be deferred |
|---|---|
| A window budget, which 16.07 hours is not | Warm-ups alone add 0.25 hours or 9.6 depending on whether a block rebuilds its arms, and NOTHING registers which. Compilation, idle refusals and the kill bench's own 0.67 minutes are also absent |
| Whether a bench may take a different round count from the step | Candidate L's long-width bench does not resolve its own scaffold offset at the registered five rounds: across four runs the stock offset ranged 49 to 338 ms and the reference offset 4 to 174, and in one run the ordering REVERSED |
| Whether candidate L's bench clears clause 21 at all | Its short-width fit sat on both sides of the 0.99 linearity floor across three runs. A bench that fails either gate leaves candidate L with no credited denominator |
| The freeze condition Amendment 10 registered and nothing has met | The incidental reading has now been seen under three backward constructions. Seeing a pilot three times is not measuring it independently, and the operand generator, its seed, the implementation hashes and the reduction are still to be frozen |

**The certified vector is 27 arms and the binding manifest is 35 short and 44 long.**
It has been corrected twice and each correction added arms nobody had listed: Amendment 8 found candidate P3 missing from Amendment 7's 27, and Amendment 10 found candidate Q's whole floor family missing from Amendment 8's 26.
The count is now a test.

**One machine limit that no clause had ever priced.**
A dense floor weight per call site needs 31.37 GiB against this machine's 33.53, so Amendment 10 clause 45 registers one weight per SHAPE, which costs 2.15 GiB.
That is a choice about what the floor arm COMPUTES, which is why it is registered rather than left to the build.

### Step 9b landed 2026-08-21: the calibration was priced by a model, and the model was wrong

Amendment 10 clause 46 said in its own text that its step context was priced through a MODEL rather than a measurement, and left four items out of its subtotal.
Measuring the model is what this step did, and it moved the number by three.

| | Modelled | Measured at the 4B |
|---|---|---|
| stock step, short width | 400 ms | 789.1 ms |
| stock step, long width | 4000 ms | 14700.8 ms |
| timed rounds over 39 blocks | 16.07 h | 47.43 h |
| the binding pass itself | 15.8 min | 56.2 min |

The long width is 94.9 percent of every block, and no schedule that keeps both widths at five rounds with clause 53's rebuild comes in under 37.11 hours.
Vlad ruled an exploratory pass first, and Amendment 13 clause 57 registers it and the rule that sizes the calibration from it, before the pass runs.

**Two obligations were discharged against the answer their own registered text predicted.**
Clause 49 diagnosed candidate L's flapping scaffold offset as needing more rounds; more rounds made it 6.6x worse at the short width, because the benches timed each arm to completion and the fault was drift between windows.
Clause 46 left the block lifecycle open; no build offset is detectable at the proxy, and a block rebuilds anyway, because a shared build makes the calibration's null conditional on a build the binding pass does not draw.

**One fault in committed code that no clause would have caught.**
A plan seed of 0 leaves mlx-lm's batch permutation unseeded, so the width's fixed batch drew 109 to 130 supervised rows across six calls, and candidate L's floor is defined on those rows.

**What step 10a inherits.**
The exploratory pass is built and device-proven; what it still needs is a runner over three passes and their sweeps, and an aggregator that reports the three margins beside the three window-separated pair contrasts.
Clause 57's branch rule reads the smaller of the two widths' margins against 8 times the largest pair contrast, and that multiplier rests on a simulated ideal null, which is why the clause forbids it substituting for a calibrated demand.

### Step 10a, 2026-08-21: the branch clause 57 registered could not be read, and the sweep could not be run

Writing the reader against clause 57 found two gaps in the clause and three defects in the code beneath it.

**The clause named two quantities and constructed neither.**
Its branch reads "the largest of the three window-separated pair contrasts", and every contrast the manifest produces sits INSIDE one pass, because all four identical-arm arms are arms of a manifest `timed_rounds` interleaves inside every round.
So the branch as committed had no referent at all.
Its other term compares "the smaller of the two widths' margins" against a contrast at no named width, which admits a long-width margin placed against a short-width null, and that is the cross-context comparison clauses 21 and 31 exist to forbid.
Amendment 14 registers both constructions and reverses nothing.

**The pass as priced cannot produce the margin the branch consumes.**
Clause 57's 3.07 hours is the step manifest alone, and the margin needs candidate L's credited denominator, which only the ceiling sweep produces.
Clause 60 gives each pass its own sweep at 3.29 minutes and corrects the pass to 3.23 hours.

**The ceiling sweep has never been run by anything.**
`run_sweep` has no caller, no test and no entry point, and its one `context` argument feeds three consumers that need three different objects: the structural pass's shape tally, a per-width cell context, and the recording's own context.
Reproduced by execution: `shape_counts` raises on a cell context, and both widths would be handed the same supervised count where the two bands differ by 91 against 3699 rows.

**What is parallel.**
Three Codex tasks run in their own worktrees: an adversarial audit of Amendment 13, which landed on self-attack alone where every prior amendment took an outside round; an audit of the code path the three-hour run will execute; and the sweep's entry point, its exploratory mode and pass-indexed recordings on both harnesses.

### Step 10a landed 2026-08-21, and two outside passes found what nine commits of self-attack had not

Three Codex tasks ran in their own worktrees against one HEAD: an adversarial audit of Amendment 13, an audit of the code path the run executes, and the ceiling sweep's entry point.

| Pass | Claims checked | Held | Reproduced failures |
|---|---|---|---|
| Amendment 13's text | 48 | 38 | 6, plus 4 underdetermined by the committed construction |
| The run's code path | 55 | 32 | 21 |

**The profile had never run.**
Its parent's preflight accepts `bands` and records `data: None`; the child's recheck indexed `plan["data"]` unconditionally, so every profile child raised `KeyError` before it loaded a model and the parent read the traceback as an unexplained child death.
No test crossed that boundary with a two-band plan.

**The ceiling sweep had never run either, and every one of its inputs was wrong.**
`run_sweep` had no caller, no entry point and no test; its single `context` argument fed three consumers needing three different objects; both widths were handed one width's supervised count where the bands differ by 91 rows against 3699; and its machine lock was constructed without the owner the lock requires.

**Two measured numbers moved.**
Candidate L's bench built its cotangent inside the timed call at the kept vocabulary, which is the dial, and the fitted floor slope fell 10.5 percent at the short width and 18.7 at the long when the operand was materialised with the others.
That slope is the credited denominator, so the bias sat inside the credit.

**Four findings reached committed text and Amendment 15 settles them.**
Clause 54's same-round-count rule would have refused clause 57's own lean schedule, and a simulation under the ideal null with per-group scales says the rule's wording was the wrong side: the pilot's round count moves the demand's precision and not its rate, and the ten percent it costs is registered rather than left inside the word variance.
The reduction across the three margins is the smallest.
The pass is 5.09 hours, because clause 45's 3372 seconds is five timed rounds and neither clause 57's 3.07 nor Amendment 14's own 3.23 counted a warm-up or a build.
And the exploratory sentinel was not inert: at 0.1 ms candidate L had no margin at either width and at 1.0 ms it had one, so the reader now computes from the fits and applies no gate.

**What is armed.**
The detached runner holds `exploratory_pass.py` waiting for a strong-idle window, three profiles and three sweeps, resumable across windows because a child's idle refusal keeps its own exit code and a pass already recorded is skipped.

### Amendment 9, LANDED `9bc94ab`: two readings the reducer had to make and the document did not supply

Clause 41 resolves clause 26's third scaffold case.
The clause calls its three cases exhaustive and its wording is not: an arm whose excursion clears `R` while its shape limits fail meets one of the second case's conditions, so the second rejects it and the third, worded as "neither of the other two conditions holds", does not take it.
Clause 26's OWN worked example is in that gap, which makes it a contradiction inside the clause rather than a case it forgot: it says of medians 100, 130, 110 and 105.5 at `R = 1` that the fitted excursion is "1.05, which clears `R`", with a coefficient of determination of 0.0012.
Registered: the third case takes everything at or above `R` the second does not.
Load-bearing on the first real data it met, where candidate L's proxy scaffold cleared the excursion condition by more than twice over while its shape limits failed.

Clause 42 names candidate L's ablation stand-in and states its bias.
Clause 33's instruction to build it the way candidate A's is built does not transfer: candidate A's arm relies on the queries carrying a gradient of one, while the loss is the ROOT of the backward, and the head's output IS the full logits tensor whose construction is the cost being removed.
The stand-in is named so its sign cannot be moved later by an unregistered change of construction, and the measured instability is recorded: across three arrangements the residue read -1.115, -0.484 and -0.188 ms, and the alternative stand-in gives +0.192, a spread of 1.307 against an assumed floor of 0.155.
All proxy figures, taken against an `R` measured uncompiled while the arms are compiled, so they bind nothing.

### Amendment 8, LANDED `4da2007`: the partition arms Amendment 7's manifest left out

Found by reading the step 7 design against the amendment step 7 would have been built from.
Amendment 7 lists the certified vector leaf by leaf and prices the binding pass at 576 seconds over 27 arms, and neither names candidate P3, the projections-without-the-head region clause 22's partition consumes, whose ladder is eight arms at every width.
The reasoning error is subtle and worth keeping: Amendment 7 argued that because candidate A is measured at the long width only, the short width falls into clause 22's NOT RUN outcome.
That holds for the PASS branch and not for the REJECT branch, which Amendment 6 registers as running "whether or not a region is absent", so it still fires at the short width reading P1 and P3 alone.
Amendment 7 therefore weakened a committed gate without declaring it.
Codex's step 7 design had already specified the correct behaviour, which is what exposed it.

| | Amendment 7 | Corrected |
|---|---|---|
| Arms, short width | not stated | 26 |
| Arms, long width | 27 total | 35 |
| Binding pass | 576 s | 752 s |

The box does not move: no certified margin reads P3, and what does read it is a fault check that takes no margin, so the certified vector stays at 18 arms and the calibration at 8.58 hours.

### Measured 2026-08-21 after the ablation fix: candidate L's residue is below resolution and its SIGN is set by the stand-in

`_loss_knob().ablate` returned the dial's smallest setting rather than an ablation, so the residue was `a - (a + b * phi_min)`, negative by a quarter of the slope, and clause 18 rejects a residue below `-R`.
Reproduced by a live test that the ablated arm computed the smallest setting's own loss exactly, then fixed with a true ablation, and the test now separates them.
The fix cannot use attention's trick of multiplying operands by zero, because the loss is the ROOT of the backward and a zero derivative there would delete the whole step rather than candidate L's two regions.

Measured on the 0.6B proxy, 9 rounds, slope 30.271 ms and intercept 56.989 ms at R-squared 0.9993:

| Ablation stand-in | Ablated step | Residue |
|---|---|---|
| a one-column slice of the head's input, backward scatters | 57.176 ms | -0.188 ms |
| a reduction over the hidden axis, backward broadcasts | 56.797 ms | +0.192 ms |

The old ablation would have given -7.971 ms, so the fault is gone by a factor of about sixteen.
What remains is worse than a small number: the two defensible stand-ins disagree by 0.380 ms while the quantity they measure is about 0.19, so the measurement cannot determine the residue's SIGN and an implementation choice does.
Clause 18's clamp records a residue below `R` as zero and would dispose of both at any `R` at or above 0.19, and the proxy's own `R` was about 0.155 uncompiled, so this is marginal rather than settled.
BLOCKER for step 12: which stand-in is registered has to be written down before any run credits candidate L, in the same way clause 27 registers candidate A's ablation and its stated bias, because a residue whose sign comes from a construction nobody registered is a rule made after the numbers.

### Still unpriced, and named rather than buried

Candidate L's bench floor and Q's sweep need their own resolution contexts under clause 21's prohibition on reuse, and neither is in the 18-arm figure.
Step 9 has to produce their arm counts before step 10's window can be budgeted, so the 8.58 hours is the step context alone and the total will be larger.
This is the one number in this section that is not yet arithmetic.

---

## Hoid, 2026-08-20: the strategy re-anchored ("Rethinking the AI Compiler", runhoid.com)

Hoid published its architecture the same day this section was written, and the appendix's competitive read ("stealth, inference-only, no public correctness or measurement story") is SUPERSEDED by this section.
Analysed by four independent passes: three lens agents (architecture, verification epistemology, strategy) and a full Codex pass, all reading the article and this repo together.
The article text is archived in the session scratchpad; the four reports agree on every point below.

### What they published

- An agentic compiler for NVIDIA inference: the agent reads a profile, hypothesizes the bottleneck, proposes graph rewrites, and a harness keeps what agrees numerically and measures faster, unattended for hours.
- An IR: the model is a serialized DAG document, rewrites are local region substitutions with declared interfaces, custom kernels are first-class nodes.
- Published results on one B200 across Qwen3-4B, Llama-3.1-8B, Qwen3-30B-A3B serving plus SDXL, Whisper and BGE-M3, with kernels exported as plain torch custom ops, measured with NO Hoid runtime present, in a public repo with pinned envs, against a best-of-stock baseline lattice.
- Forward only. No training, no gradients, no backward anywhere in the post.

### The comparison verdict, agreed by all four passes

| Fact | Consequence |
|---|---|
| Their thesis IS our funnel's rule V1 ("any implementation may be proposed, none allowed in without being checked"), stated publicly with a reference result | "Agentic AI compiler" died today as a headline; differentiation must come from depth and domain |
| They engineered the surface (IR) and ASSERTED the verbs (correct, faster); we engineered the verbs and asserted the surface | The two systems are mirror images; the union is the product |
| Of 12 benchmark-lie failure modes this repo has produced and gated, their loop as described catches 1 cleanly, we catch 11; our only PARTIAL is the baseline lattice, where they are better | The gameability table is both our marketing ammunition and our own checklist |
| Their reward-hacking defense is one unquantified sentence; our mutation battery's measured escape rate is the publishable version of it | Say so explicitly in any public comparison, after R11's battery runs |
| Their own binding numbers come from kernels dropped into unmodified PyTorch, a seam architecture; the IR is their SEARCH surface only | The IR question is narrower than it looks and does not touch how kernels ship or get measured |
| MLX gives serialize (binary), inspect (dot), replay, and NO rewrite path; an MLX IR means writing a training-graph compiler middle-end, and the trace decomposes fused ops, generates the backward, and holds no single static graph | The IR is deferred with a registered trigger: the day the profile shows headroom dominated by inter-seam fusion rather than any single operation, seams are exhausted and the IR earns its build |

### The position (the innovative AI compiler)

**The verified training compiler: an evidence-carrying agentic compiler for training, whose customer outcome is faster QLoRA without silently changing the adapter the user gets.**

| Layer | Role | Why Hoid cannot cheaply copy it |
|---|---|---|
| Verification depth | THE MOAT | Retrofitting fp64 references, calibrated tolerance contracts, mutation populations with measured escape rates, sealed held-out draws, exact-vjp checks and evidence-bound certificates is methodology, not a feature; it is also literally Abloh's company thesis, so it compounds |
| Training and the backward | THE FRONTIER | Their post is forward-only; a training rewrite must carry its own gradient, MLX composes fused attention under any gradient trace (measured here), and gradient verification is machinery only this repo has built |
| Honest instrumentation and pre-registration | THE OPERATING LAYER | This week measured that agent-readable profiles lie (marked shares summing to 1.857); "the agent reads a profile" fails silently without an engineered instrument, and a serialized graph makes a profile readable, not honest |
| Apple Silicon | THE WEDGE, never the moat | Their GTM gravity (B200 buyers, book-a-call) points away from consumer Macs and pip-install distribution; sell it as where we start |
| Agent-facing IR | THE LATER SCALING LAYER | Deferred with the trigger above; copying their frame now means competing on their terms with zero kernels kept |

The window: roughly 2 to 4 quarters before a crude agentic training result appears on the NVIDIA side; the methodology moat outlasts the window but only monetizes if a kept kernel and a shipped end-to-end number exist inside it.
The binding constraint is therefore unchanged and sharpened: SPEED TO ONE HONEST KEPT-KERNEL E2E NUMBER, and the current profile increment is that number's critical path, not a detour.

The 90-day public sentence, every clause of which already has a gate in this plan:
"A real QLoRA fine-tune through stock mlx-lm runs at least 10% faster end to end on an Apple Silicon Mac, and every kernel we swapped in, forward AND its gradients, was verified against an exact reference before it was ever timed, under rules we published before the measurement, with a receipt in every run you can re-check."
Hoid can say none of the four clauses.

### Adopted from Hoid, each at its natural landing spot, none in the current increment's critical path

| Mechanism | Landing spot | When |
|---|---|---|
| Best-of-stock baseline lattice (sweep the stock configurations a user can reach, register best-stock-per-metric as the denominator) | A written amendment to section 8's arms, plus `bench/stock_lattice.py`; select on exploratory samples, remeasure the winner in the binding run | Before the E2E binding run, after the profile increment |
| Clean-export confirmation arm (kept kernels installed as bare `mx.custom_function` swaps in a minimal script with metalrunner absent), WITH the fix Hoid skipped: the exported artifact is re-verified by the existing extraction and certificate machinery | A second confirming arm beside the E2E harness; the metalrunner product arm stays the headline because receipts and routing reports ARE the product | With the E2E harness work, Day 5-6 |
| Stranger-reproducible benchmark package (pinned env, two-command re-run); stronger for us than for them because per-chip claims mean every M-series owner produces their own binding number | The Day 6 wheel-into-fresh-venv item | Day 6 |
| Typed profile handoff (the generator receives the decision artifact's slopes, widths, shapes and hashes rather than prose) | A typed adapter beside `kernelverify/compiler/brief.py` | When the generator next runs against the selected operation |
| Per-metric reporting shape (their TTFT vs throughput split, best stock per metric) | Section 11's reporting and the receipt | With the E2E report |
| Their framing sentence ("correctness stops being a property of the transformation language and becomes a property of the loop the harness runs") | Public positioning, used only after R11's mutation battery has run, citing the measured escape rate | R11 |

### Guardrails on our own claims, from the same analysis

- Never claim the agent found the bottleneck: humans named the three candidates and pre-registered arithmetic selects among them, and that is the honest description.
- Never say "verified" publicly before R11's training mutation battery has a measured escape rate.
- Scope the backward claims to their evidence: Hoid's published loop is forward-only, and MLX has no fused ATTENTION backward and composes fused attention under gradient tracing; neither fact supports "no system anywhere touches backward".
- Never cite Hoid's speedups as production-serving facts: their baseline class is torch.compile, not vLLM or TensorRT.

## Executed increment: the measurement bridge (2026-08-19 night; all five steps committed on the branch; Vlad's ruling: one installer, two callers)

### Context

- The committed E2E harness imports a "measurement module" named by its run plan and demands a protocol nothing implements: `install(candidate_sha256s, force_stock, guard)` returning an object with `uninstall()` and `evidence()`, plus `verify_backward_exact(model_path, candidate_sha256s, guard)`.
- Without it the harness's "ours" arm cannot differ from its control, so the sprint's GO/NO-GO run cannot exist; this is the named integration gap from the harness commit.
- metalrunner already has the ingredients: `seams.Installation` (counting, foreign-object refusal, reverse removal), `routing.CERTIFIED` (empty, filled by the keep stage), `FORCE_STOCK_ENV`.
- No kernel is kept yet and the first operation is unselected (Day 1 profile pending), so the bridge must be complete and tested NOW with zero certified entries; shipping a kernel later is one CERTIFIED entry plus one registry row, with no code change in either caller.
- Vlad ruled to wire `metalrunner.lora.main` to the same installer in this increment, so the harness provably measures the code users run.
- Design validated against the real contract sites by a design agent, 2026-08-19.

### The module: `metalrunner/measurement.py`

- `MeasurementRefusal(RuntimeError)`: a permanent reason this process must not be measured; nothing was changed.
- `Operation` (frozen dataclass): the executable half of a certified kernel, keyed by operation name: `seams` (tuple of `seams.Seam`), `replace(entry, original)` (must verify its loaded source hashes to `entry["candidate_sha256"]` before returning), `backward_cases(entry, model_path)` (data-only case dicts), `backward_check(entry, case, model_path)` (stock gradient via mx vjp vs the candidate's backward, judged by `mx.array_equal` per prereg 8.4).
- `OPERATIONS: dict[str, Operation] = {}`: the code registry, one row per shipped operation, EMPTY today; a registry test pins `keys() ⊆ KNOWN_OPERATIONS` and `== {}` now.
  Code registry over dotted-path strings in data: the operation set is closed at three names, import-time checkable, and callables never enter the data table.
- `entry_problems(entry) -> list[str]`: the keep stage's write contract, every schema fault in plain words.
- `tree_sha256(root)`: the harness preflight's exact algorithm (sorted files, `__pycache__`/`.pyc`/dotfiles excluded, length-prefixed relative path then content).
  The harness's wrapper-hash call sites (preflight ~1418, child ~1735) switch to importing THIS function, so wrapper-hash agreement is by identity, not by test; the harness keeps its own generic `_tree_sha256` for the model and data trees only.
- `install(*, candidate_sha256s, force_stock=None, guard=None, certified=None, operations=None, on_chip=None) -> Installed`: keyword-only injection params are the test seam (harness passes only the three demanded kwargs); `force_stock=None` reads `routing.forced_to_stock()` so `METALRUNNER_FORCE_STOCK` drives the user-path control.
- `Installed.uninstall()` and `Installed.evidence()`; evidence is a pure read that never raises (a raise there would burn a detached-runner retry as EXIT_CHILD_DEATH).
- The bridge NEVER imports mlx; real operation rows import it inside their own callables, which is what makes every refusal path testable under `KV_FORCE_NO_METAL=1`.

### CERTIFIED entry schema (data only, six flat keys, lives in routing.py)

`candidate_sha256` (the store identity), `operation` (a KNOWN_OPERATIONS name), `chip` (written at keep time FROM `routing.chip()` itself, so one reader serves both sides), `bits`, `group_size`, `pricing_recording_sha256` (the evidence pointer for the receipt; the bridge does not act on it).

### install() semantics

- Empty `candidate_sha256s`: install nothing, return honest-zero evidence; this is lora.main's everyday path today and never reads the chip.
- Resolution per candidate, identical for ours and control so a bad entry refuses both arms the same way: missing entry, schema-invalid entry, wrong chip, missing OPERATIONS row, two candidates on one operation, all refuse; a mid-way `SeamRefusal` rolls back everything installed so far, then re-raises as `MeasurementRefusal` (chained).
- The per-call decision point is the wrap closure: every call through a seam increments `routing_decisions`; under force_stock the replacement is NEVER built (control excludes kernel-build cost, which is exactly O4's wrapper-own-cost) and every decision falls through to stock; in the ours arm every decision routes, so `routed_calls == routing_decisions > 0` and `routed_candidates` is exactly the resolved kept set.
- Evidence shape (superset of `_routing_evidence`'s demands): wrapper_installed, forced_stock, wrapper_sha256 (`tree_sha256` of the live metalrunner package, computed once at install), routed_candidates (a real LIST, `_routing_reasons` isinstance-checks list), routed_calls, routing_decisions, plus candidate_sha256s, per-seam counts, foreign_on_removal.
- Guard: one check at install (a kernel whose construction blows the footprint refuses before training), then sampled at exponentially spaced decision counts (1, 2, 4, 8, ...), because a per-call footprint syscall would sit inside the timed region of the primary metric; the harness's own per-step guard is the coarse net in all three arms.
  `guard=None` (the user path) skips all checks.

### verify_backward_exact

- Cases come from the registry row, so a kernel ships with its fixed cases in the same commit or refuses; case dicts are JSON-safe data (shapes, dtypes, seeds).
- `cases_sha256` binds the full spec: sha256 over canonical JSON of `{candidate: {operation, cases}}`; a changed shape, seed, dtype or count changes the committed hash.
- `exact=False` is NEVER a refusal: the bridge reports, `run_binding` rules (a false here IS the measurement); refusals are unknown candidate / schema / chip / missing row / zero cases / non-bool case verdict.
- Guard checked per case; result echoes the resolved candidate set so `_backward_child`'s set-equality gate passes.

### The harness diff (~20 lines in bench/train_lora_e2e.py)

- `_measurement_call(module, function, **kwargs)`: duck-typed on the module's own `MeasurementRefusal` attribute; maps that one type to `PreconditionFailed` (EXIT_PRECONDITION, permanent, not retried) and lets every other exception keep its retryable-traceback semantics; used at the two call sites in `_training_child` and `_backward_child`.
- The two wrapper-hash call sites import `metalrunner.measurement.tree_sha256` (agreement by identity; metalrunner never imports mlx at import time so the parent process stays clean).

### Wiring lora.main (the second caller)

- `routing.py` gains `eligible(fine_tune_type, bits, group_size, *, on_chip=None, certified=None)` returning the matching certified entries; `decide()` is rebuilt on it and now returns `routed=True, "certified and priced here"` on a match (fixing the latent inconsistency where `_why_not` could say routed while `decide` hardcodes False); forced-stock still declines everything with the control reason; behavior-neutral while CERTIFIED is empty.
- `lora.main`, after the routing report and recorder-seam install: derive candidates from `routing.eligible`, `measurement.install(candidate_sha256s=...)`, refusal maps to `EXIT_SEAM_REFUSED` (same condition class: an unverifiable process) and removes the recorder seam; `patch.uninstall()` in the finally alongside the existing removal.
- `receipt.build` gains one `measurement` field carrying `patch.evidence()`, so today's receipt truthfully says routed nothing, zero decisions, in writing.
- The recorder seam stays in lora.py: it is progress recording, not kernel routing, and the harness deliberately installs its own capture at those names.

### Tests (~29, all green under KV_FORCE_NO_METAL=1; style source tests/test_metalrunner_seams.py: purpose-built target module, fake Operation row, fake certified entry, recording guard; every refusal asserts the refusal AND that nothing changed)

- Bridge behavior: ours calls reach the replacement and are counted; control decides without routing; the replacement is never built under force_stock; force_stock defaults to the environment; uninstall restores and counts survive; zero candidates install nothing and say so.
- Refusals: unkept candidate; wrong chip; operation without a row; two entries on one operation; foreign object wrapped as MeasurementRefusal; partial install rolls back; entry_problems names every schema fault.
- Guard: checked at install and at exponential decision counts; a missing guard is not an error.
- Backward: binds cases by hash and counts them; one false case makes exact False, not a refusal; unknown candidate refuses without mlx; a row with no cases refuses; cases_sha256 moves when a shape, seed or dtype moves.
- Meeting-point (real bridge objects through the real harness judges, no kernel, no GPU): live evidence passes `_routing_evidence` against the preflight's hash; ours + control evidence plus the harness's stock literal satisfy `_routing_reasons` end to end; a HOLLOW ours arm (installed, never called) fails `_routing_reasons`, proving the harness catches a bridge that routed nothing; the real `_backward_child` accepts the bridge's backward evidence via monkeypatched tables (it imports no mlx).
- Harness diff: a typed refusal maps to the precondition exit; an untyped crash still escapes.
- lora wiring: main routes kernels only through the shared installer and uninstalls even when run() raises, with the receipt carrying the evidence; nothing certified installs no kernel seam; decide marks a certified operation routed while forced-stock still declines.

### Files

- New: `metalrunner/measurement.py` (~250 lines), `tests/test_metalrunner_measurement.py` (~500 lines).
- Modified: `bench/train_lora_e2e.py` (+~20), `metalrunner/routing.py` (+~25/-10), `metalrunner/lora.py` (+~20), `metalrunner/receipt.py` (+~8), `tests/test_train_lora_e2e.py` (+~45), `tests/test_metalrunner_lora.py` (+~70), `AGENTS.md` module map, `TODOS.md` (retire the integration-gap entry).
- Reuse, never reinvent: `seams.Installation` for all installation and counting; `routing.decide/chip/forced_to_stock`; the harness's `_json_sha256` canonical-JSON discipline.

### Sequencing (commit per green step)

1. `measurement.py` core plus its bridge/refusal/guard/backward tests.
2. Harness diff (`_measurement_call` + wrapper-hash call sites import the bridge's `tree_sha256`) plus its two tests.
3. `routing.eligible` + `decide` routed branch, lora.main wiring, receipt field, plus their tests.
4. Meeting-point tests driving the real harness judges.
5. AGENTS.md and TODOS.md; full suite both ways; the real two-iteration fine-tune re-run to show the receipt's new measurement field on a live run.

### Verification

- Full suite green with Metal and with `KV_FORCE_NO_METAL=1` before every commit.
- The live smoke fine-tune through `metalrunner.lora` passes with the receipt carrying honest-zero measurement evidence.
- The hollow-bridge meeting-point test proves the harness would refuse an ours arm that routed nothing, which is the failure this bridge exists to make impossible to miss.

## Context

The readiness map of 2026-08-19 (114 agents, 71 claims surviving adversarial check) found the repo measures inference decode on one machine and one model, nothing is installable, and no code path accepts a user's model.
Vlad re-aimed the product: metalrunner, a verified AI kernel compiler that makes QLoRA fine-tuning through mlx-lm faster on Apple Silicon Macs.
The compiler is a generate, gate, price, keep loop: a language model writes Metal kernel candidates plus a knob search; the existing contract-anchored verifier gates every candidate before it is timed; the existing pre-registered pricing protocol prices it; survivors are kept with a certificate.
Everything downstream of the generator already exists and is battle-tested; the generator, search space, candidate store, held-out sweep and tolerance-free gates are new.
Four independent design sources were run and reconciled (Codex training pivot, Codex competitive read, a 3-architect 3-judge design panel, Codex 30-50% lever stack); their reports are summarised in the appendix.

## Rulings (Vlad, 2026-08-19)

| # | Question | Ruling |
|---|---|---|
| R1 | First user | Someone LoRA / QLoRA fine-tuning an open-weight model on their Mac |
| R2 | Model families | Qwen 3, Llama 3.x, Mistral / Ministral, Phi-4, DeepSeek R1 distills |
| R3 | Sizes | Under 4B and 7-9B |
| R4 | Machines | 16 GB Macs and up |
| R5 | Headline | Faster fine-tuning, measured end-to-end through the training loop |
| R6 | Must-haves | Verified underneath; runs inside 16 GB |
| R7 | Interface | The user types `metalrunner.lora` instead of `mlx_lm.lora`, every mlx-lm flag and config file unchanged; it runs mlx-lm's own trainer in-process with verified pieces swapped in at the module-level seams; no hidden patching; upstreaming a backend hook is proposed but does not gate v1 (revised after mlx-lm 0.31.3's closed parser was reproduced) |
| R8 | Entry point | `mlx_lm.lora` only, no Python API for custom loops |
| R9 | Base format | 4-bit, group 64 only for v1 |
| R10 | Floor | At least 10% end-to-end, interval clear of 1.0, or the speed claim does not ship |
| R11 | Verified bar | v1 ships on today's gate (verified before timed, fixed cases, honest certificates); the mutation battery with a measured escape rate lands before the word "verified" is used publicly |
| R12 | Decode track | Finish lane zero, then park decode; no new decode lanes until the training product exists |
| R13 | The product | An AI kernel compiler: generate, gate, price, keep, applied to the QLoRA training step; "literally a new thing that is more effective" |
| R14 | Generator | A language model writes Metal kernel bodies from gate and price feedback; a parametric search sweeps knobs on each body |
| R15 | Sprint and ordering | Days: the loop live end to end on ONE training operation ending in a kept kernel and an honest end-to-end measurement; then the pip package and the entry point immediately after, same machine, one lane |
| R16 | Second Mac | 16 GB Apple Silicon; it takes the shipping end-to-end number; the M3 Pro 36 GB stays the dev and pricing machine |
| R17 | Dev model | Qwen3-4B 4-bit group 64 |
| R18 | Design target | 30-50% faster end-to-end, verified throughout, reached as a stack of levers each with its own contract, found and tuned by the compiler per operation and per machine; R10's 10% stays the shipping floor |

## Technical calls made in this plan, listed for veto

- The first operation is chosen by a pre-registered rule applied to a Day 1 measurement, not by argument: the two design voices split (Codex: streamed masked loss; panel: forward quantized matmul), and both converge on "profile stock first", so the profile decides.
- The headline claim is scoped to prompt-masked instruction tuning (the normal QLoRA case); the all-token case is measured and reported alongside at its own honest number (the evidence supports roughly 20-25% there today, not 30-50%).
- Sprint 1 numbers are scoped to the M3 Pro; the 16 GB MacBook takes the shipping number when it is set up (R16); no simulated memory cap is quoted as a 16 GB result.
- The two auxiliary workflows (terrain map, lever hunt) were dropped after two process crashes under load; their remaining value was share estimates, which the Day 1 profile measures on the real machine instead (resume ids in the appendix if ever wanted).
- New code lives in the existing repo: `kernelverify/compiler/` for the loop, `metalrunner/` for the user-facing package; the verifier keeps its name as the library.
- Work happens on a new branch `lane/metalrunner-sprint1` off main; lane zero stays on `lane/4bit-pricing`.

## Machine safety (standing, from today's crashes)

- No GPU measurement is armed without Vlad's explicit go in chat, run by run.
- At most one armed detached job at a time; the disarm command is `launchctl bootout gui/$(id -u)/com.kernelverify.detached-<harness>`.
- The runner already refuses a busy machine and holds `caffeinate` only while armed; arming now additionally waits for Vlad to say the machine is free and on AC power.
- Development smoke tests (seconds of GPU) are fine at any time; anything minutes-long goes through the detached runner.

---

# The design

## The loop, module by module

| Stage | What it does | New or reused |
|---|---|---|
| Brief | Renders the op contract, shapes, dtypes and prior evidence into the generator prompt | New: `kernelverify/compiler/brief.py` |
| Generator | A seed hand-written candidate guarantees the loop never starves; the LLM generator (pinned model id, prompt and response stored verbatim) proposes bodies from gate and price feedback; feedback is built from evidence objects only, never raw stdout | New: `kernelverify/compiler/generator.py` |
| Knob search | Tile sizes, simdgroup layout, unroll, threadgroup memory staging per body, pruned by the 32 KB threadgroup limit | New: `kernelverify/compiler/search_space.py`; reuses `kernelverify/runners/specialize.py` |
| Lint | Source attestation of the working-precision clause: no half or bfloat tiles or accumulators outside load and store idioms; names the offending token | New: `kernelverify/compiler/lint.py` |
| Tolerance-free gates | NaN-sentinel unwritten-output check, guard rows past the output, NaN and infinity propagation, four-way determinism including a fresh worker, shape change at awkward sizes; zero false positives by construction, run before any tolerance | New: `kernelverify/compiler/gates.py`; reuses `MetalRunner`, `GateEvidence` |
| Tolerance gate | The existing contract gate extended for training: bf16 contract entry (new eps and uint16 carrier, its own ADR, pre-registered before any gate runs), gradient cases (input, A, B), mask and padding patterns, tied head; judged against the fp64 reference with the calibrated K | Extends `kernelverify/pack/verify.py`, `kernelverify/schemas/native_ops.py` |
| Price | The existing protocol: interleaved arms, reference canary, interval verdicts, one machine lock, idle gates, memory budget; arms are ours, stock, an in-contract fp32-dequant control fed to the generator, and a dense fp16 ceiling column | Reuses `bench/interleave.py`, `price_qmv_boundary.classify`, `machine_state`, `memory_guard` |
| Funnel | Lint and tolerance-free gates run on every candidate (target tens per hour); the full tolerance gate on survivors; pricing on gate-green survivors only; the sealed held-out sweep only on frozen incumbents | New wiring in `kernelverify/compiler/loop.py` |
| Held-out | Per-candidate seeded shape draw withheld from the generator (M bands, tile-edge shapes, one projection family, dtype, distributions); reaches the generator as PASS or FAIL only; a LOSS inside the claimed window shrinks or voids it; failures never feed back into the lineage | New: `kernelverify/compiler/heldout.py` |
| Store | Append-only JSONL plus content-addressed `.metal` blobs; source, parent, prompt, compiler error, gate evidence, price distribution, rejection reason; refusals kept; re-proposals refused by hash | New: `kernelverify/compiler/store.py` under `bench/.candidates/` |
| Keep | Kept bodies, certificates and the pricing recording committed; a routing table derived at import from the recording, pinned by sha256 of recording, kernel source and launch config, exactly like the decode table | New: `kernelverify/pack/routed_train.py`; reuses `bench/emit_pack_certificates.py` |

Every kept forward is wrapped in `mx.custom_function` with a registered `.vjp`, which is MLX's documented mechanism and the pattern mlx-lm's own losses use; `mx.grad` through a raw `mx.fast.metal_kernel` raises.
The certificate binds both the forward and backward sources.

## The first operation: decided by measurement, rule written first

Day 1 runs one binding profile of stock `mlx_lm.lora` on Qwen3-4B (batch 1 and 4, sequence 2048, rank 8, masked and all-token cells): per-op time shares, memory, padding fraction, cache-clear cost, and a quantized-matmul roofline against a dense fp16 ceiling.
The pre-registered selection rule, committed before the profile runs: for each candidate op, compute the end-to-end gain 1 / (1 - f (1 - 1/r)) from the measured share f and the evidence lower bound of the per-op ratio r, and take the largest.

| Candidate op | Evidence for r before the profile |
|---|---|
| Streamed mask-aware lm_head plus cross-entropy, with hidden-gradient vjp | Stock builds all 152k-vocab logits then masks; eliminated work is source-proven; direct effect 10-20% on masked data |
| Tiled attention forward and backward (grouped-query, causal) | MLX's fast attention falls back to composed ops during training and its fused backward on GPU is not implemented, source-proven in mlx v0.32.0 |
| Quantized matmul retune at training widths | Fixed 32x32x32 tiles both directions; headroom unknown until the roofline; kill rule: within 1.10x of the dense ceiling at 4 of 5 cells means this op is dead |

The two ops not chosen stay in the stack as sprint 2 and 3 of the same loop.
Deterministic transforms (the padding and length-sorting fix, the cache-clearing fix) are not kernels: they enter as exact-equivalence transforms proven by identical per-token loss and gradients on fixed cases, and are offered upstream to mlx-lm.

## Target arithmetic (acceptance budgets from the appendix, to be replaced by Day 1 measurements)

| Case | Budgeted stack | Result |
|---|---|---|
| Prompt-masked, batch 4 | loss 0.83, attention 0.75, qmm 0.71, transforms 0.68 | about 1.47x |
| All-token | attention 0.90, qmm 0.84, transforms 0.81 | about 1.23x, 1.37x with a padding win |

30-50% is honest for the masked case at batch 4; the all-token batch 1 case reads lower and the receipt states which case a run was.

## The interface and the receipt

- `pip install metalrunner`, then `metalrunner.lora` with every mlx-lm flag unchanged; it wraps mlx-lm's module seams in-process, prints a routing report (what was swapped, what was declined and why), and refuses dora and full fine-tune modes with one line.
- Version pins: mlx and mlx-lm versions and file hashes checked at start; on mismatch nothing is patched and the run refuses loudly.
- Every run writes a training-run receipt: model and dataset fingerprints, adapter config, chip and toolchain, forward and backward source hashes, certificate ids, dispatch counts, peak footprint, seeds, loss summaries, final adapter hash.
- The receipt attests certified dispatch plus recorded runtime invariants; it never claims per-step numerical containment without shadow checks.

## The end-to-end measurement

- Harness `bench/train_lora_e2e.py`: three arms (stock, ours, ours-forced-stock as the control), fresh subprocess per arm per round, five interleaved rounds, through the detached runner.
- Primary cell: Qwen3-4B, masked instruction data, the largest batch that fits, sequence 2048, rank 8, pinned dataset and seeds; secondary cell: all-token.
- Fairness, pre-registered: same model and adapter init hashes, same examples and order and truncation and masks, same supervised tokens, same optimizer and schedule and seeds, same memory ceiling; full-job wall time plus warmed steady-state; a bigger-batch result is stated as throughput at the same memory ceiling, never same-batch speed.
- GO: ratio_lo above 1.10 (R10); target 1.30 plus (R18); peak footprint no worse than stock or the memory line does not ship.

## Sprint day-by-day

| Day | Work | Ends with |
|---|---|---|
| 0 | Branch; pre-registration doc (selection rule, cells, GO rules, held-out draw, fairness); bf16 contract ADR on CPU with tests | Rules committed before any measurement |
| 1 | Binding stock profile plus roofline (one detached slot, Vlad's go required); `custom_function` under compile and checkpoint smoke on the 0.6B model | First op selected by the rule; noise floor known |
| 2 | Store, lint, tolerance-free gates, gate harness for the chosen op; seed candidate through the whole gate; eight-mutant mini battery, every mutant caught by its named stage | Candidate zero verified |
| 3 | Generator and knob search; one bounded detached generation session | Store holds 20 plus candidates with pilot prices |
| 4 | Full pricing of the top three (one slot, Vlad's go); sealed held-out; keep, certificate, routing table | One kept kernel or an honest none-kept record |
| 5 | `metalrunner` package, entry point, refusal paths, tests; E2E binding run (one slot, Vlad's go) | GO or NO-GO on the primary cell |
| 6 | Full mutation battery over the training policy; wheel into a fresh venv; stranger test on a Llama-3.2-3B checkpoint | "Verified" earned per R11; pip path green |
| 7 | Buffer; ADR, AGENTS.md, TODOS.md; merge gate; stop for Vlad | |

Each day ends in something verified or measured; a red day stops the sprint rather than sliding.

## Lane zero: the 4-bit decode pricing (R12, in flight)

- Tasks A and B are committed on `lane/4bit-pricing` (`a7f5d4d`, `a0163e7`); the full detailed plan for Tasks D and E is preserved in the appendix.
- The armed run was disarmed today at Vlad's request after it refused once on memory (needed 18.53 GB free, had 17.01) and then waited on battery.
- Sequencing: re-arm in the first quiet window Vlad grants (about 45 minutes, AC, apps closed); sprint Day 1's profile takes the second window; sprint Days 0 and 2 code work does not conflict.
- After its run: Task D (adoption of the 4-bit windows), Task E (ADR and record), merge gate, then decode parks.

## Verification

- `pytest -q` green in the venv before every commit; `KV_FORCE_NO_METAL=1 pytest -m gpu` shows no unprotected test.
- The eight-mutant mini battery on Day 2 proves each gate stage catches its named fault class; the full battery runs Day 6 before "verified" is said.
- The kept kernel's vjp is `mx.array_equal` to the stock gradient in isolation; the E2E arms share identical seeds and data and are compared under the pre-registered fairness rules.
- The wheel installs into a fresh venv and `metalrunner.lora --help` matches `mlx_lm.lora --help`; the version-pin refusal fires against a bumped mlx-lm.
- Every binding number comes from a detached, locked, idle-gated run and lands in a committed recording; a run that goes non-idle quarantines and binds nothing.

## NOT in scope for sprint 1

- The backward-orientation quantized matmul kernel and the two unchosen first-op candidates (sprints 2 and 3 of the same loop).
- 8-bit, fp16 base, group sizes other than 64, dora, full fine-tune (R9).
- The 16 GB shipping number (needs the second Mac set up; sprint 1 numbers are M3 Pro-scoped).
- Per-chip windows for M1, M2, M4 (route-to-stock on uncertified chips instead).
- Upstreaming to mlx-lm (proposed as PRs, never gating).
- Any new decode work (R12).

---

# Appendix: evidence log (2026-08-19)

## Codex outside voice (scratchpad/codex-training-pivot.md)

- BLOCKER, reproduced: the original one-flag interface was impossible in mlx-lm 0.31.3; R7 revised to `metalrunner.lora`.
- Streamed masked loss: stock builds 0.62 GB (batch 1) to 2.49 GB (batch 4) of bf16 logits then masks; Unsloth and Liger avoid the full tensor already, so the novelty is mask-aware plus verified plus receipt.
- Think big: verified memory-bounded QLoRA (preflight refusal against a memory ceiling, budgeted recompute, exact resumability, the training-run receipt); mlx-lm saves only adapter weights today.
- Gate gaps for training: gradient contracts, mask and padding patterns, tied head, forward-plus-backward certificates; sprint work, not a later lane.
- Most valuable asset: the refusal-based claim discipline and the 4-bit group-64 contract; dead weight after lane zero: all decode artefacts, kept as history.

## Competitive read (scratchpad/codex-hoid-research.md)

- Hoid: stealth, inference-only, no public correctness or measurement story; its 100x claim is AMD and SGLang's work. STALE AS OF 2026-08-20: superseded by the "Hoid, 2026-08-20: the strategy re-anchored" section above; still inference-only and still no public correctness or measurement depth, but public, published, and reproducible.
- Stronger players: Yasp, Meta KernelEvolve, Kernel Forge, ATREX, AMD Apex, Alloy (closest on Apple, non-agentic).
- Positioning: "the verified AI kernel compiler for mlx-lm QLoRA on Apple Silicon"; the moat is the accumulated proof trail, which cannot be manufactured retroactively.

## Design panel (tasks/wf6qrcgfc.output; 3 architects, 3 judges)

- First op per the panel: forward quantized matmul (about 60% of step FLOPs, compute-bound, MLX's tile out of contract clause C1); backward orientation sprint 2.
- Facts reproduced: bf16 storage in the pinned artifacts; `mx.grad` raises through `metal_kernel`; tied lm_head is not a `QuantizedLinear` leaf; M is not always a multiple of 32.
- Amdahl formula and the module map adopted into the design; the `.pth` silent-patch idea rejected (R7).
- Confidence: 85% loop live, 65% kernel WIN somewhere, 45% E2E GO in sprint 1.

## Codex 30-50% stack (scratchpad/codex-30-50.md)

- Budgets: masked batch 4 composes to about 1.47x; all-token to 1.23x, 1.37x with a padding win; 30-50% on all-token batch 1 at 16 GB is not presently honest.
- Attention lever, source-proven: MLX fast attention falls back during training; fused GPU backward not implemented.
- Loss corrected to 11-13% of step work; padding 10-35% at batch 4 pending a corpus measurement; length sorting broken in stock; cache cleared every step because a flag is never forwarded; step compile already stock; LoRA fusion alone 0-3%.
- Funnel capacity: 2-3 full candidates per hour today; design target tens per hour through the cheap gates; sealed sweep requirements listed and adopted.
- Fairness rules adopted verbatim into the E2E section.

## Dropped workflows

- Terrain map `wf_b35a74ec-a26` and lever hunt `wf_d5847f61-602`: died twice with process crashes under load; superseded by the Day 1 profile; resumable by runId if ever wanted.

## Lane zero detail (preserved from the eng-reviewed plan)

- Tasks A and B: committed (`a7f5d4d` the pre-registered first-window and red-gate rules; `a0163e7` the width-aware probe and gate with refusals: empty-list refusal, exclusive-create recording, kernel digest, per-width provenance).
- Task C: `bench/start_binding_run.sh bench/price_qmv_boundary.py -- --bits 4`, about 45 minutes; needs 18.6 GB free, AC, quiet.
- Task D: per-width recording collection in `routed_windows` (refuse duplicate width, key-vs-bits mismatch, incomplete grid; check the kernel digest where present, name its absence for 2026-08-15); first window per shape is the single unbroken WIN run; holes stop at the red contiguity test for a written ruling; gate coverage moves with the routed set in one change; the routed-versus-gated cross-check iterates every width.
- Task E: ADR beside 0015 with the verdict census and scope sentence; AGENTS.md and TODOS.md; merge gate; Vlad's word.
- Decisions D1-D9 of the eng review stand as written in git history and the probe docstring.

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | NOT RUN | - |
| Codex Review | `/codex review` | Independent 2nd opinion | 7 | ACTIVE | plan 13 / 11 folded; code 9 / 8 folded; knob plan 15 / 15 folded or answered; Amendment 6 draft: 2 rounds BLOCKER (8 then 10 findings), draft to be replaced per re-cut step 5; next-step evaluation ruled the re-cut; Hoid analysis pass folded into the strategy section |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 2 | CLEAR (PLAN) | round 1: 20 issues; round 2: 9 issues, 4 ruled decisions |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | N/A | no user interface in scope |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | N/A | internal measurement harness |

Scope of the latest round: the "Next increment: the knob profile, two registered widths, and Amendment 5" section, 2026-08-20, branch `lane/metalrunner-sprint1`.

**ENG REVIEW, round 2.**
Scope challenge tripped on file count and the scope held: two of three knobs already exist, the shared runner is being moved rather than built, and the rest is forced by rulings already taken.
Nine issues, four of which were decisions and were ruled: the attention dial is chosen by a criterion registered before the three candidate dials are measured; the rejected deletion method stays as the only independent cross-check and is re-measured compiled so both halves share a regime; the tie band's resolution floor is measured at the target with a refusal registered against the answer; the three reporting cells stay in full.
Five were fixes and were applied: the memory budget clause had lost its referent, `binding_blockers` gated on four things that stop existing, the rule code had no width dimension, a probe named for one candidate was everyone's runner, and the scaffold arm had three names.
Eight test gaps were added to the plan, one of them a regression that had already cost a whole test session.

**CODEX, third pass, on the knob plan.**
Fifteen findings, eight P0, verdict BLOCKER, and it was right.
Four were decisions and were ruled: a retune and a rewrite cannot share one credit rule, the floor has to be measured inside the same step as the share, one corpus has to supply both widths, and the amendment's numeric limits come from an instrument-only stage that computes no share.
Two were faults in the committed tree rather than the plan: the shared interleaver takes exactly two arms and cannot express a four-point ladder, and the selection rule tests the shipping floor against the largest gain and then lets a tie-break pick a candidate below it.
Both reproduced by reading the code, and the second is a real bug that is fixed regardless of what happens to this increment.
Two were contradictions inside my own draft: the sum check demanded disjoint shares while another clause deliberately puts the tied head in two candidates, and the launch-count test counts Python calls rather than Metal dispatches.
Three were MLX facts I had assumed: the compile cache keys on the underlying callable rather than the returned wrapper, the optimizer's first update grows the captured state and forces a second trace, and fused attention is selected only at certain head dimensions so a four-point ladder would fit a line through two implementations.
The rest tightened the kill rule's reductions, the corpus registration, the scaffold pricing and the ordering between validation and binding.

**CROSS-MODEL:** one direct tension, resolved in Codex's favour.
I ran a live experiment showing two compiled closures cache separately and called the per-arm design confirmed.
Codex read MLX's source and showed the cache keys on the underlying callable, so my experiment passed only because each arm happened to build a fresh closure.
Both statements are true and Codex's is the load-bearing one: the design was correct by accident of construction rather than by guarantee, so it is now three registered requirements with a trace counter as the general guard.
Across three Codex passes and two eng rounds the two voices have still never found the same thing twice.

**VERDICT:** ENG CLEARED at plan level, with thirteen decisions ruled and all fifteen outside findings folded or answered.
The plan now runs to sixteen steps, Amendment 5 to twenty-five clauses, and the validation window splits into an instrument-only stage that registers the limits and a knob stage that uses them.
Codex asked for a repeat pass before the amendment is committed, and step 1 should not land without it.

**UNRESOLVED DECISIONS:**
- Whether to run the repeat Codex pass on Amendment 5's actual text before committing it, as the outside voice asked. Recommended, and cheap, but it is a call about how many rounds this gets.
- The four GPU windows at steps 10, 12, 13 and 14, each needing its own explicit go, and step 13 additionally blocked on step 12 passing rather than on approval alone.
- The committed end-to-end harness still registers the 2048 that the profile has now stopped registering, and it is a TODO outside this increment.

---

# THE ANY-MAC PIVOT (planned 2026-08-23, supersedes every open increment above)

## Context

Vlad re-aimed the product: metalrunner must deliver speedups on ANY Mac a user downloads it onto, not just the dev machine.
The long-term vision is an autonomous performance engineer for Apple Silicon: it profiles, tunes, and applies verified optimizations for the user's model on the user's chip, with no manual tuning.
Four rulings taken 2026-08-23, all by AskUserQuestion:

| # | Question | Ruling |
|---|---|---|
| 1 | v1 scope | QLoRA fine-tuning via mlx-lm only; inference is v2 |
| 2 | Where the AI runs | Kernel generation on our side; the user's Mac autonomously tunes, verifies, and keeps or routes to stock; no API key on the user's machine |
| 3 | First-run tune budget | Minutes; small pre-vetted config space, cached per chip forever after |
| 4 | The calibration lane | DROPPED; the 40-GPU-hour statistical profile dies, its machinery is kept; the first operation is the quantized matmul (share 0.376, treated as settled) |

## Design decisions

**D1. The workload is two orientations of one operation.**
Every projection call in the trainer is `QuantizedLinear.__call__`, which is `mx.quantized_matmul(x, w, scales, biases, transpose=True)` at the six registered shapes (S1-S6 including the head and k/v proj).
The backward needs dX via the transpose=False orientation at the same shapes.
No gradient flows to the quantized weight (frozen in LoRA), so the vjp needs the x-cotangent only, the arrangement the gradient-path tests already pin.

**D2. Forward swap ships first; the backward swap is a second certified entry later.**
Prereg 8.4 demands the kept kernel's backward be bit-equal to stock's gradient, and a swapped backward kernel cannot be (different accumulation order).
So the v1 keep unit is: our forward kernel + a vjp that calls stock `mx.quantized_matmul(transpose=False)` for dX, keeping 8.4 exactly satisfiable.
The backward-orientation swap comes later behind its own amendment (tolerance verdict on the gradient replacing bit-equality for that entry).

**D3. Verification split.**
Certified once centrally (chip-independent, properties of the source and format): lint, input-support, the full tolerance battery over shapes x widths x the whole pre-vetted knob grid, gradient cases, held-out draw.
Re-run on the user's device (properties of that chip's compiler and arithmetic): compile, the unwritten-output sentinel gate, and the tolerance gate against a freshly computed fp64 reference at shipped seeds.
The reference is recomputed on-device (Accelerate is not bit-identical across chips) but computed ONCE per (shape, M-bucket, orientation) and reused across all configs, so reference cost does not scale with config count.
K is the shipped K_QUANT, never re-derived on-device.

**D4. Tuner budget arithmetic.**
At most 8 pre-vetted configs per orientation x 6 shapes x 2 M-buckets read from the user's own batches: roughly 100-200 cells.
Each cell: verify (sentinel + one tolerance case against the cached reference), then time interleaved against stock via `interleaved_arms`, ~5 rounds.
Estimated 5-12 minutes end to end; per-cell atomic checkpoints make interruption resumable.

**D5. One local entry per operation carrying a winner map.**
`measurement._resolve` enforces one kernel per operation, so a locally tuned entry is ONE entry whose payload maps (shape, orientation, M-bucket) to a knob config or "stock", with per-call fallthrough to the original where the map says stock.

**D6. Layering lift.**
The tuner is product code and must not import bench/.
`bench/interleave.py`, BudgetGuard, and the AC-power probe lift into a new `kernelverify/timing/` package, with the bench modules left as re-exports so existing harnesses and tests do not change.

## The 5-day parallel schedule (ruled 2026-08-23: full product scope, compressed by parallelism, not by cuts)

The structural fact the schedule is built on: only the dev machine has a Metal device, so the kernel and every measurement are Claude's critical path, and Codex (deviceless, in its own worktrees) takes every line of CPU-testable code off that path.
Deliverables are unchanged from the step table this section replaces; only ownership and ordering change.
Per the standing rule, this plan runs through /plan-eng-review before dispatch, first thing Day 1.

### Ownership rules (how we work with Codex, binding for the whole sprint)

- Codex tasks run in separate worktrees off the lane branch, one task per worktree, dispatched with: an exact file-ownership list (zero overlap with Claude's files or each other's), the interface contract it must meet (function signatures, entry schema), the test command, and the statement that no Metal device exists there.
- Codex never writes GPU-marked tests' bodies beyond skips; every GPU assertion is Claude's to run and land.
- Merge points are fixed at the end of each day: Claude reviews and lands Codex diffs daily so integration debt never exceeds one day.
- Codex's second job is continuous adversarial review: each evening it reviews that day's landed commits (read-only), so review happens in parallel instead of as one big gate at the end.
- Every merge into the lane still runs /code-review on the merge diff per the standing rule.

### Day-by-day

| Day | Claude (GPU, critical path) | Codex (deviceless, parallel) | Night (GPU, detached) |
|---|---|---|---|
| 1 | /plan-eng-review on this plan; step 1 contract groundwork folded into kernel start; MSL seed kernel, tt orientation, first funnel-green compile and correctness at one shape | Task A: verify.py nt reference path + train_qmm skeleton tests. Task B: `metalrunner/local_store.py` + schema-2 `entry_problems`/`routing.eligible` changes with tests | none |
| 2 | Kernel bring-up: both orientations correct at all six shapes; knob axes exported; first timings vs stock | Task C: `metalrunner/tuner.py` decision logic, checkpoint/resume, cache invalidation, battery deferral, BudgetGuard wiring, all against hardware fakes. Evening: adversarial review of Day 1 landings | none |
| 3 | Step 3 vjp + OPERATIONS row: the one GPU test through value_and_grad + checkpoint + compile; wire tuner's GPU path; tuner smoke on device | Task D: `lora.py` + receipt tuning section + tests. Task E: certificate emission + `kernelverify/timing/` lift with re-exports. Evening: review of Day 2 landings | LLM generation session 1 via `bench/generate_candidates.py`, detached, caffeinated |
| 4 | Fold session 1 winners (if any) through the full gates; run the dev machine's own tuner for real; full suite both ways; fix fallout | Evening: adversarial review of the whole branch, merge-gate style. If session 1 was weak: retune the generation brief for session 2 | The binding 3-arm E2E run, detached; LLM session 2 only if the E2E window allows both |
| 5 | Read the binding result; if >= 10% with the interval clear, the number ships; AGENTS.md, one-paragraph prereg amendment (lane drop, tuner protocol, D3 split, D2 scoping of 8.4); buffer for red | Final /code-review on the merge diff | re-run slot if Day 4's E2E was refused by the idle gate |

### What moves and what is deferred (none of it product scope)

- Step 10 (the 16GB Mac's number) is deferred past the 5 days: it needs the second machine set up, blocks nothing, and accrues whenever it exists. The dev machine number ships first, labelled as such, consistent with R16.
- LLM generation compresses from "a few sessions" to one guaranteed night plus one conditional night. The hand-written kernel is the floor; the sessions are upside on top of it.
- The single hard dependency chain is: kernel correct (Day 2) -> vjp wired (Day 3) -> tuner smoke (Day 3) -> dev tune (Day 4) -> binding run (Night 4) -> number (Day 5). Everything Codex builds sits off this chain and can land any day.

### Schedule risks, named

- Kernel bring-up in 2 days is the tight link. Mitigations: start from the proven wide_qmv idiom, tt orientation first (it alone ships per D2, since the backward stays on stock's op), nt orientation degrades to a Day 3 stretch goal rather than a blocker.
- Nights 3 and 4 assume granted overnight windows on AC power with caffeinate, per the standing machine-safety rules; a lost night pushes the binding number into Day 5's re-run slot and the buffer absorbs it.
- If Day 5's number lands below 10%: the honest outcome is recorded, session 2's candidates become the recovery move, and the claim waits; the tuner, package, and receipt still ship as working product.

## Task breakdown (writing-plans format; every step carries its own verification)

> **For agentic workers:** execute task-by-task with checkbox tracking. Codex tasks are dispatched verbatim as worktree briefs. Claude tasks run on the dev machine (the only machine with a Metal device).

**Goal:** metalrunner v1 that tunes, verifies, and speeds up QLoRA on any Mac, with the binding >= 10% number, in 5 days.

**Architecture:** one hand-written knob-parameterized Metal kernel for the transpose=True quantized matmul (the forward), swapped in via the existing seam installer; the vjp calls stock `mx.quantized_matmul(transpose=False)` so prereg 8.4's exact-gradient rule holds; a first-run tuner verifies-then-times pre-vetted configs per chip and writes one schema-2 local entry; routing merges local entries with the shipped table.

**Tech stack:** Python, MLX 0.32.0, Metal MSL via `mx.fast.metal_kernel`, pytest (`pytest -q`, and `KV_FORCE_NO_METAL=1 pytest -m gpu` must show no unprotected test).

**Spec:** the ANY-MAC PIVOT section above (context, rulings, D1-D6).

### Global constraints

- Suite green with Metal AND under `KV_FORCE_NO_METAL=1` before every commit; commit per green step on `lane/metalrunner-sprint1`.
- Codex has NO Metal device: it writes deviceless code and deviceless tests only; every GPU assertion is Claude's.
- Codex tasks run in separate worktrees with zero file overlap; merge points at end of each day; every merge runs /code-review on the merge diff.
- fp32 accumulators in every kernel (lint clause C1); MLX's contiguous little-endian code stream, group 64 always a whole block (the wide_qmv layout, verified bit-identical to mx.quantize).
- No GPU window armed without Vlad's go; nights need AC power plus caffeinate.

### The shared interface contract (all tasks build against this, defined here once)

```python
# kernelverify/pack/train_qmm.py (data surface)
TRAIN_SHAPES = {"S1": (4096, 2560), "S2": (2560, 4096), "S3": (9728, 2560),
                "S4": (2560, 9728), "S5": (151936, 2560), "S6": (1024, 2560)}  # (d_out, d_in)
ORIENTATION = "tt"          # v1 ships transpose=True only; nt is the later second entry per D2
def m_bucket(m: int) -> int  # next power of two, floor 64, cap 2048
def axes() -> dict[str, list]                    # knob axes for search_space.enumerate_knobs
def threads(knobs: dict) -> int                  # for search_space.enumerate_knobs
def threadgroup_bytes(knobs: dict) -> int
TRAIN_QMM_TT_MSL: str                            # the door-neutral kernel body
PREVETTED_KNOBS: tuple[dict, ...]                # <= 8 configs, chosen on the dev machine, NOT chip-filtered
VERIFY_SEEDS: dict[str, int]                     # shape key -> seed for on-device tolerance cases

# metalrunner/local_store.py
LOCAL_ENTRY_KEYS = ENTRY_KEYS + ("schema", "provenance", "mlx_version",
                                 "knob_map", "tuned_at", "evidence_sha256")
# knob_map: {"S5:M128": {...knobs...} | "stock", ...}; schema == 2; provenance == "local-tune"
def load(*, chip, mlx_version, candidate_sha256s, path=None) -> tuple   # () or (entry,)
def save(entry: dict, path=None) -> Path                                # atomic write + fsync
def local_entry_problems(entry) -> list[str]

# metalrunner/tuner.py
@dataclass(frozen=True)
class TuneOutcome:
    entry: dict | None      # the schema-2 entry written, or None
    report: dict            # receipt-ready: cells, census, verdicts, margins, deferral reason
    cache_hit: bool
def ensure_tuned(*, model_path, bits, group_size, chip, mlx_version,
                 batches=None, store_path=None, hardware=None) -> TuneOutcome
# hardware is the injectable seam bundling: limits probe, verifier, timer, power probe, guard
```

### Task 0 (Claude, Day 1, before any dispatch): root CLAUDE.md with skill routing rules

The repo has no root `CLAUDE.md` (only `.claude/CLAUDE.md`, the Spartan toolkit file), so the gstack preamble reports `HAS_ROUTING: no` on every skill run.

**Files:**
- Create: `CLAUDE.md` (repo root)

- [ ] **Step 1:** create root `CLAUDE.md` with exactly this content:

```markdown
# kernelverify

Repository-specific instructions live in AGENTS.md. Read it before doing
anything substantial here; this file only routes skills.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill
tool. When in doubt, invoke the skill.

MANDATORY: whenever planning anything or assigning a task to any agent, use
the superpowers skills. Planning goes through superpowers:brainstorming then
superpowers:writing-plans; task assignment and dispatch go through
superpowers:subagent-driven-development (or superpowers:executing-plans for
inline execution). This applies every time, with no exceptions.

Key routing rules:
- Planning any work or writing any plan → superpowers:brainstorming then superpowers:writing-plans
- Assigning or dispatching a task to any agent → superpowers:subagent-driven-development
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate (with superpowers:systematic-debugging)
- QA/testing behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec
```

- [ ] **Step 2:** verify the check passes: `grep -q "## Skill routing" CLAUDE.md`.
- [ ] **Step 3:** commit alone on the lane branch: `chore: add skill routing rules to CLAUDE.md`.
- [ ] **Step 4:** save the standing rule to agent memory (superpowers on every planning and task-assignment action) so it applies beyond this repo.

### Task 1 (Codex worktree A, Day 1): the nt reference path and the train_qmm data surface

**Files:**
- Create: `kernelverify/pack/train_qmm.py` (data surface only: TRAIN_SHAPES, ORIENTATION, m_bucket; the MSL lands in Task 2 by Claude, do not write kernel code)
- Modify: `kernelverify/pack/verify.py` (add `train_qmm_inputs(x, w, bits, orientation)`)
- Modify: `kernelverify/schemas/native_ops.py` (the quantized_matmul reference must accept orientation "nt": reference is `x @ dequant(w)` instead of `x @ dequant(w).T`, both at fp64)
- Test: `tests/test_pack_train_qmm.py`

**Interfaces:** produces `train_qmm_inputs` and the nt reference; Task 6 consumes `reference_and_tolerance("quantized_matmul", train_qmm_inputs(...))` on-device.

- [ ] **Step 1: failing test** that `m_bucket(65) == 128`, `m_bucket(4224) == 2048`, `m_bucket(64) == 64`, and that `train_qmm_inputs` round-trips orientation into the inputs dict.
- [ ] **Step 2: failing test** that the nt reference equals numpy `x.astype(f64) @ dequantize(artefact, f64)` on a seeded 8x128 case, using the existing `dequantize` from `kernelverify/schemas/quant_contract.py`.
- [ ] **Step 3:** implement; run `pytest tests/test_pack_train_qmm.py -q` to green.
- [ ] **Step 4:** full suite under `KV_FORCE_NO_METAL=1`; commit `feat: train-qmm data surface and nt reference path`.

Brief line for dispatch: "You have no Metal device. Touch only the four files above. The existing tt reference and tolerance machinery in native_ops must be unchanged for existing callers; add, never edit, and prove it by the existing tests staying green."

### Task 2 (Claude, Days 1-2, GPU bursts): the seed kernel, tt orientation, knob axes

**Files:**
- Modify: `kernelverify/pack/train_qmm.py` (TRAIN_QMM_TT_MSL, axes, threads, threadgroup_bytes, PREVETTED_KNOBS, VERIFY_SEEDS)
- Create: `bench/pack_train_qmm.py` (the dev-machine funnel harness, pattern of the wide_qmv bench)
- Test: `tests/test_pack_train_qmm_live.py` (gpu-marked)

**Interfaces:** consumes Task 1's reference path; produces the kernel source whose sha256 becomes `candidate_sha256`, and the knob surface `search_space.enumerate_knobs(axes(), limits, threads=threads, threadgroup_bytes=threadgroup_bytes)` consumes.

- [ ] **Step 1: failing GPU test** at S6 (1024, 2560), M=64: run the kernel via `mx.fast.metal_kernel`, judge with `verify.verify_output("quantized_matmul", train_qmm_inputs(x, w, 4, "tt"), out)`.
- [ ] **Step 2:** minimal correct kernel: threadgroup-staged X tiles, per-block dequant in registers (the wide_qmv 8-code block idiom), fp32 accumulators, knobs BM/BN/BK as function constants. Make Step 1 pass.
- [ ] **Step 3: failing test extension** to all six shapes at M in {64, 128, 512, 2048} plus non-bucket M values (65, 1057) to prove edge handling; make it pass.
- [ ] **Step 4:** export `axes()` (BM, BN, BK, simdgroups, unroll, staging) and verify `enumerate_knobs` returns a non-empty launchable set on this device with every rejection reasoned.
- [ ] **Step 5:** `bench/pack_train_qmm.py` runs the full funnel (compile, lint, unwritten, input-support, tolerance) over the launchable set at the six shapes; every config funnel-green or its failure named.
- [ ] **Step 6:** first interleaved timing vs stock via `interleaved_arms({"stock": ..., "ours": ...}, rounds=5)` at S3 and S5 (the two biggest time-eaters), informational only; pick PREVETTED_KNOBS as the <= 8 configs that win or come closest across shapes.
- [ ] **Step 7:** commit per green step; final commit `feat: train-qmm tt kernel, knob axes, pre-vetted configs`.

### Task 3 (Codex worktree B, Day 1): local store and schema-2 entries

**Files:**
- Create: `metalrunner/local_store.py`
- Modify: `metalrunner/measurement.py` (`entry_problems` gains a schema-2 branch validating LOCAL_ENTRY_KEYS; schema-1 validation byte-identical to today)
- Test: `tests/test_metalrunner_local_store.py`, extend `tests/test_metalrunner_measurement.py`

**Interfaces:** produces `load`/`save`/`local_entry_problems` per the contract block; Task 5 consumes them. `_resolve` must accept a schema-2 entry unchanged (it only reads the six base keys plus validation).

- [ ] **Step 1: failing tests:** save-then-load round-trips; load returns `()` on chip mismatch, mlx_version mismatch, or candidate_sha256 not in the passed set; a corrupt JSON file loads as `()` with a warning, never raises into the entry path.
- [ ] **Step 2: failing tests:** `entry_problems` accepts a valid schema-2 entry, names every missing or malformed schema-2 key, and still validates schema-1 exactly as the existing tests demand.
- [ ] **Step 3:** implement with atomic write (`tempfile` + `os.replace`); green; `KV_FORCE_NO_METAL=1` full suite; commit `feat: local certified store, schema-2 entries`.

Brief line for dispatch: "No Metal device. Touch only these files. Existing measurement tests must pass unmodified except where you extend them additively."

### Task 4 (Codex worktree C, Day 2): the tuner core against hardware fakes

**Files:**
- Create: `metalrunner/tuner.py`
- Test: `tests/test_metalrunner_tuner.py`

**Interfaces:** consumes Task 3's store and the contract block's `hardware` seam (a dataclass of callables: `limits() -> Limits`, `verify_cell(shape, m, knobs) -> bool`, `time_cell(shape, m, knobs) -> dict[str, list[float]]`, `on_battery() -> bool`, `guard: BudgetGuard`); produces `ensure_tuned` and TuneOutcome. Claude wires the real callables in Task 6; Codex writes fakes only.

- [ ] **Step 1: failing tests, decision arithmetic:** a cell is kept only when `min(stock_samples) / max(ours_samples) >= 1.03` after the canary spread check (reference arm max/min <= 1.5, reusing MAX_CANARY_SPREAD); a noisy tie routes "stock"; a withheld cell routes "stock" with reason "canary".
- [ ] **Step 2: failing tests, flow:** cache hit runs zero cells; on_battery defers with reason and touches nothing; a BudgetExceeded from the guard skips that cell with reason and continues; verify_cell False means the cell is never timed (assert the timer fake was not called: rule V1).
- [ ] **Step 3: failing tests, resume:** per-cell checkpoints fingerprinted by (chip, mlx_version, candidate_sha256); a killed tune resumes past completed cells; a partial checkpoint never becomes an entry.
- [ ] **Step 4:** implement; green; commit `feat: on-device tuner core`.

Brief line for dispatch: "No Metal device, no mlx import at module top (mirror measurement.py's discipline). The verify-before-time ordering is rule V1: encode it so the timer cannot be reached for an unverified cell, and test that."

### Task 5 (Codex worktree D, Day 3): entry point and receipt wiring

**Files:**
- Modify: `metalrunner/lora.py` (after `read_quantization`, before `routing.decide`: call `tuner.ensure_tuned`; build `certified = routing.CERTIFIED + local_store.load(...)`; pass `certified=certified` into `decide` and derive candidates from `eligible(..., certified=certified)`)
- Modify: `metalrunner/receipt.py` (a `tuning` section carrying TuneOutcome.report verbatim)
- Modify: `metalrunner/routing.py` only if `render` needs the tuning summary line
- Test: extend `tests/test_metalrunner_lora.py`

**Interfaces:** consumes Tasks 3 and 4. The forced-stock control path must never tune (assert `ensure_tuned` not called when `forced_to_stock()`).

- [ ] **Step 1: failing tests:** first run calls ensure_tuned and the receipt carries the tuning report; second run cache-hits; control run never tunes; a tuner exception is caught, reported as a deferral in the receipt, and the run proceeds stock (a broken tuner must not break training).
- [ ] **Step 2:** implement; green both ways; commit `feat: tuner wired into the entry point and receipt`.

### Task 6 (Claude, Day 3, one GPU window): vjp wrapper, OPERATIONS row, real tuner arms, smoke

**Files:**
- Create: `metalrunner/operations/__init__.py`, `metalrunner/operations/train_qmm.py`
- Modify: `metalrunner/measurement.py` (OPERATIONS gains the one row; registry test updated)
- Modify: `metalrunner/tuner.py` (the real `hardware` bundle: `search_space.Limits` probe, verify via the funnel with `reference_and_tolerance` cached per (shape, bucket), time via `interleaved_arms`, power probe, BudgetGuard from `mx.device_info()["memory_size"]`)
- Test: `tests/test_metalrunner_operations_live.py` (gpu-marked)

**Interfaces:** the Operation row per measurement.py's contract: seams `(Seam("mlx.nn", "QuantizedLinear.__call__"), Seam("mlx.nn", "QuantizedEmbedding.as_linear"))`; `replace(entry, original)` hash-checks TRAIN_QMM_TT_MSL against `entry["candidate_sha256"]`, consults `entry["knob_map"]` per call site shape, falls through to `original` where the map says "stock"; the vjp computes dX via stock `mx.quantized_matmul(..., transpose=False)`; `backward_cases` returns seeded shape dicts; `backward_check` compares by `mx.array_equal`.

- [ ] **Step 1: failing GPU test:** through the real seams, `value_and_grad` + checkpoint + compile in the trainer's arrangement produces loss and gradients bit-equal to stock when knob_map routes everything to "stock", and correct-within-tolerance loss with bit-equal dX gradients when a real config routes (the vjp IS stock, so gradients given equal upstream cotangents match exactly on the fixed backward cases).
- [ ] **Step 2:** implement the row; green.
- [ ] **Step 3: GPU smoke:** `ensure_tuned` end-to-end on one shape (S6) with the real hardware bundle; assert an entry lands with at least one non-stock or all-stock map and the report says which.
- [ ] **Step 4:** commit `feat: train-qmm operation row and live tuner arms`.

### Task 7 (Codex worktree E, Days 2-3): certificates and the timing lift

**Files:**
- Modify: `bench/emit_pack_certificates.py` (a certificate for (TRAIN_QMM_TT_MSL sha256, "tt"): source hash, axes, PREVETTED_KNOBS, VERIFY_SEEDS, funnel outcomes read from the Task 2 recording)
- Create: `kernelverify/timing/__init__.py`, `kernelverify/timing/interleave.py` (moved), `kernelverify/timing/guards.py` (BudgetGuard, available-memory, power probe lifted)
- Modify: `bench/interleave.py`, `bench/memory_guard.py` (become re-exports; every existing import keeps working)
- Test: existing suites prove the lift is behavior-neutral; new certificate-fact tests

- [ ] **Step 1:** the lift, proven by the existing tests passing untouched; commit.
- [ ] **Step 2:** certificate emission with failing-first fact tests; commit.

### Task 8 (Codex worktree F, Day 2): the LLM generation brief for train-qmm

**Files:**
- Modify: `bench/generate_candidates.py` (a `--operation train-qmm` mode: brief built from the train_qmm contract, funnel compile/lint/unwritten/support, store-backed)
- Test: extend `tests/test_compiler_llm.py` with a fake-model round trip for the new brief

- [ ] **Step 1:** failing test that the brief carries the six shapes, the knob axes, and the incumbent's funnel record; implement; green; commit.

### Task 9 (Claude, Night 3, detached GPU): LLM generation session 1

Runbook, not code: `caffeinate -i python bench/generate_candidates.py --operation train-qmm --backend claude --model claude-fable-5 --rounds 3 --per-round 5`, detached, resumable. Morning: read the census; candidates joining PREVETTED must pass the full Task 2 funnel first.

### Task 10 (Claude, Day 4): fold winners, dev-machine tune, full suite

- [ ] **Step 1:** any session-1 winner re-runs `bench/pack_train_qmm.py` gates; only funnel-green sources join the shipped family (at most 2 sources).
- [ ] **Step 2:** run the real tuner on this machine; inspect the entry and receipt by hand.
- [ ] **Step 3:** full suite both ways; fix fallout; land every Codex merge with /code-review per merge; commit.

### Task 11 (Claude, Night 4, detached GPU): the binding E2E run

Runbook: dev tuner entry in place, then `bench/start_binding_run.sh bench/train_lora_e2e.py` (3 arms, 5 interleaved rounds, idle-gated, caffeinated). The harness already demands measurement.py's protocol; nothing new to build.

### Task 12 (Day 5): the number, docs, and the final gate

- [ ] **Step 1 (Claude):** read the binding recording; >= 10% with the interval clear ships the claim; below it, record honestly and queue session 2 as recovery.
- [ ] **Step 2 (Claude):** AGENTS.md module map; one-paragraph prereg amendment (lane drop, tuner protocol, D3 split, D2 scoping of 8.4), committed alone.
- [ ] **Step 3 (Codex):** final /code-review over the whole branch diff vs the pre-pivot baseline; findings folded or answered.

## Top risks

| Risk | Mitigation |
|---|---|
| MLX quantized byte layout misread (silent) | Copy the proven wide_qmv idiom; input-support gate per config; tolerance gate against MLX's own artifact bytes |
| vjp coverage / 8.4 bit-equality | v1 vjp calls stock nt; composition pinned through the real seams; nt swap gated separately later |
| Tuner timing noise on user machines | Interleaved arms, spread canary withholds cells, interval + 3% margin, no keep on battery, env-forced re-tune escape hatch |
| 16GB machines | BudgetGuard from device memory, chunked fp64 references, head cells at reduced M first, unfittable cells skipped with recorded reason |
| Chips where nothing beats stock | Route-to-stock is a first-class outcome stated in the receipt; the claim is per-chip via receipts, never blanket |

## NOT in v1

Inference/decode, quant formats beyond 4-bit group-64, DoRA and full fine-tune, an on-device LLM agent (roadmap: opt-in deep-tune mode with the user's API key), attention and loss-head kernels (operations 2 and 3 of the same loop), the statistical calibration lane, and any new pre-registration statistics machinery: the receipt plus verify-before-time gates are the evidence discipline now.

---

# PIVOT, measured 2026-08-23: the first operation is attention, not the quantized matmul

Day 1 of the 5-day any-Mac sprint was meant to build candidate Q's kernel. The
kernel was built, is correct at all six shapes, and is knob-tunable. Then it was
measured against the machine rather than against expectation, and the operation
choice did not survive.

## What was measured, and what it killed

**Candidate Q has no headroom.** Interleaved, 7 rounds, canary clean:

| shape | M | dense fp16 | MLX quantized | dense/quant |
|---|---|---|---|---|
| S3 gate/up | 260 | 3.116 ms | 3.012 ms | 1.034 |
| S3 gate/up | 2114 | 18.367 ms | 19.213 ms | 0.956 |
| S4 down | 260 | 2.751 ms | 2.617 ms | 1.051 |
| S4 down | 2114 | 18.861 ms | 19.275 ms | 0.979 |
| S1 q_proj | 260 | 1.171 ms | 1.117 ms | 1.049 |
| S1 q_proj | 2114 | 7.730 ms | 8.094 ms | 0.955 |

The dense matmul does strictly less work, because it never dequantizes. MLX
matches it within 4.5 percent everywhere and BEATS it at the short width, where
4-bit weights win on bandwidth. MLX is at the practical ceiling of the machine.
Best case for Q is r <= 1.15, which at f = 0.376 is 5.2 percent end to end,
against a 10 percent floor.

**Both proxy shares were artifacts.** The profile's shares came from a 0.6B model
at width 97, and neither transfers to Qwen3-4B at the registered bands.

| | 0.6B | 1.7B | 4B target |
|---|---|---|---|
| head share of matmul work | 26.1% | 18.1% | 9.7% |

So candidate L's 0.245 was really the 0.6B head being large. At the target f_L is
about 0.10, and even the measured 1.798x work elimination gives 4.6 percent.

Candidate A's 0.045 was measured at width 97, and attention is the one region
quadratic in width while every other is linear.

## What it found instead

Measured at Qwen3-4B attention geometry, fused against composed:

| B | T | fused | composed | ratio | x36 layers |
|---|---|---|---|---|---|
| 4 | 65 | 0.423 ms | 1.370 ms | 3.24x | 49 ms |
| 2 | 1057 | 4.499 ms | 26.123 ms | 5.81x | 940 ms |
| 1 | 2048 | 6.729 ms | 44.892 ms | 6.67x | 1616 ms |

Composed is what training runs, confirmed by reading the graph MLX builds:
the fused primitive appears once in a plain call and zero times under `mx.grad`,
where a Softmax appears instead. MLX also implements no fused attention backward
at all, so the backward is composed too.

At the long band attention forward is roughly a quarter of the step and runs
5.8x off its own fused reference. At a conservative 3x on the region,
`1/(1 - 0.25*(1 - 1/3)) = 1.20`, so 20 percent end to end.

The second prize is memory. The composed path materializes a
B x heads x T x T score matrix: 286 MB PER LAYER at B=2, T=1057, fp32. Not
building it is what puts the 16 GB machine (R4, R6) in reach, and no amount of
matmul tuning would have delivered that.

## The revised Lane A

1. Tiled causal grouped-query attention, forward, with online softmax so the
   score matrix is never materialized. `mx.fast.scaled_dot_product_attention` is
   the correctness AND speed reference for this half, which is why the hardest
   candidate is still tractable in the window.
2. Its backward, hand-written, since MLX has none to compare against. This is
   the genuine risk in the sprint and it is named rather than buried.
3. The same knob surface and the same on-device tuner, unchanged: the tuner
   tunes whichever kernel ships, so Lanes B and C are unaffected by this pivot.

Kept, not discarded: the tt quantized-matmul kernel is committed, correct at all
six shapes, and reaches 1.457x at S6/M=260 over 138 launchable knob settings. It
becomes a later lever at the shapes where it wins rather than the headline.

---

# THE ATTENTION INCREMENT (final plan, 2026-08-23; supersedes the Lane A tasks of the 5-day breakdown above)

## Context

The pivot section above holds the evidence.
Two rulings taken 2026-08-23 by AskUserQuestion, both recommended options:

| # | Question | Ruling |
|---|---|---|
| 1 | Backward verification | Amendment to prereg 8.4: bit-exact gradient equality stays for retune-class kernels whose vjp delegates to stock; rewrite-class kernels (this one) are verified by tolerance against an fp64 ANALYTIC gradient reference through the existing native-op battery, plus the E2E harness's existing loss-curve gate |
| 2 | Timeline | The 5-day window holds; overnight LLM generation sessions are the first cut |

## Lane status at planning time

| Lane | State | Commits |
|---|---|---|
| A (mine, GPU) | Tasks 0-2 landed: CLAUDE.md `20155b9`, QMM data surface `ca1628e`, QMM kernel `515afed`. Revised below to attention | on lane/metalrunner-sprint1 |
| B (tuner, worktree kv-anymac-mr) | DONE_WITH_CONCERNS: store + schema-2 `68b83e7`, tuner core `bcff9c3`, lora wiring `4c0fd8b`. Two confirmed findings, fixes WRITTEN but blocked by plan mode (fix plan committed at `docs/plans/2026-08-23-lane-b-tuner-fixes.md`) | lane/anymac-metalrunner |
| C (timing, worktree kv-anymac-timing) | DONE: lift `a9dd688`, certificates `4d6a17e`, generation brief `b73fd44`. Behavior-neutral, proven by byte-identical failure lists against pristine code, zero existing tests edited | lane/anymac-timing |

Open lane items to execute (Task F below): Lane B fix 1, tuner.py cells_considered/knob_census must count deduplicated verdicts (len(verdicts)) not raw cells (len(cells)); Lane B fix 2, one new lora test driving a real unmonkeypatched cache hit; Lane C concern, the train-qmm certificate's WRAPPER_ASSEMBLED provenance means emit() never writes it (named refusal by design in report/certificate.py), needs a ruling when the kernel's certificate actually ships; two re-reviews died on session limit (resets 11:20pm London) and re-run after the fixes.

## Geometry and interfaces (fixed, all tasks build against these)

```python
# kernelverify/pack/train_attention.py
N_Q_HEADS = 32; N_KV_HEADS = 8; GQA_GROUP = 4; HEAD_DIM = 128
SCALE = 0.08838834764831845          # 128 ** -0.5, the value qwen3 computes
BATCHES = (1, 2, 4)                  # anything else routes to stock
def t_bucket(t) -> int               # powers of two, rounds UP, floor 128 cap 2048
def cell_key(batch, t) -> str        # f"B{batch}:T{t_bucket(t)}"
CELLS = ("B4:T128", "B2:T2048")      # the two registered bands
# three knob surfaces, each the (axes, threads, threadgroup_bytes) triple
# search_space.enumerate_knobs consumes:
FWD_KNOB_AXES  = {"RQ": [4,8,16], "SIMDGROUPS": [2,4], "BK": [16,32,64], "PIPE": [1,2]}
# dq_axes / dkv_axes share names where shapes allow; dkv adds BKV, BQS
def inputs_for_cell(mx, cell, seed)  # seeded fp16 (q,k,v,do) at the TRUE band width (65 or 1057)
def verify_cell(mx, knobs_triple, cell, seed) -> bool   # verify_output + verify_grads + arms_agree smoke
def time_cell(mx, knobs_triple, cell, rounds) -> dict   # interleaved_arms over ours/composed, fwd and grads-alone
```

The seam: `Seam("mlx_lm.models.qwen3", "scaled_dot_product_attention", defined_in="mlx_lm.models.base")`.
Training reaches it with cache=None, mask="causal" (the literal string), sinks=None; the routed callable declines to stock on ANY other combination, on dtype != fp16, and on any geometry other than 32/8/128.

The door (settled design, spike-tested before any Metal):

```python
@mx.custom_function
def inner(q, k, v):            # returns (O, L); L is fp32 logsumexp (B, 32, T)
    ...
@inner.vjp
def _(primals, cotangents, outputs):
    q, k, v = primals; dO, _dL = cotangents; O, L = outputs
    D = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1)
    return run_dq(...), *run_dkv(...)
def seam_callable(q, k, v):    # the seam returns only O
    O, _L = inner(q, k, v); return O
```

L as a second output is the recommended mechanism because the vjp receives outputs, checkpoint re-executes the forward so L is never held across the layer stack, and it costs one 270 KB store.
Plan B if the two-output composition breaks under compile: a small LSE kernel pass inside the vjp, adopted the same day the spike fails.

Backward math (the fp64 analytic reference implements exactly this):
P = softmax(SCALE * Q K^T + causal); O = P V; D = rowsum(dO * O);
dV = P^T dO; dS = P * (dO V^T - D); dQ = SCALE * dS K; dK = SCALE * dS^T Q;
dK and dV sum over each GQA group's 4 query heads.

## Tasks (remaining 4 days; commit per green step; suite green both ways before every commit)

### Day 1

- [ ] **A1. The prereg amendment, committed ALONE and FIRST.** One doc-only commit: 8.4 scoped to retune-class (train_qmm's tt-forward-stock-backward is the worked example); rewrite-class verified by fp64-analytic tolerance through the battery plus the existing loss-curve gate; the F6/F7 measurements recorded with their numbers and the statement that the amendment postdates them. No GPU.
- [ ] **A2. The two-output custom_function spike.** Extend the tests/test_custom_function_gradient_path.py arrangement (value_and_grad + checkpoint + compile) with a toy two-output kernel BEFORE any Metal exists. GPU, minutes. If it fails, adopt plan B the same day and record it in the ledger.
- [ ] **A3. Cell surface + references.** train_attention.py geometry/cell surface with CPU tests mirroring test_pack_train_qmm.py; native_ops.py gains "train_attention": attn_reference (fp64 causal GQA), attn_lse_reference, attn_grad_reference (the analytic math above), ensemble members (standard-softmax-fp32, online-softmax-fp32, reversed-keys-fp32; two orders for grads), attn_tolerance via floored_tolerance at borrowed K=1.5 with the kv_tolerance-style borrowed-K docstring, augment rescaling q and k so softmax is neither uniform nor saturated. Battery dims exercise GQA ratios 2, 4, 8 and T=65 as the edge-tile case. Cross-check tests: attn_reference vs mx.fast.sdpa outside a trace; attn_grad_reference vs mx.grad of composed fp32 MLX attention. GPU for cross-checks only.
- [ ] **A4. ATTN_FWD_MSL at one hand-picked setting.** Flash-style online softmax, causal bound on the key loop, GQA by hq/4, K/V tiles staged as half (storage), every accumulator fp32 (lint C1 attested), out-of-range keys masked to -INF, out-of-range query rows store nothing, epilogue writes O and L. fp64 + sdpa parity at edges T in {1, 63, 64, 65, 127, 128, 1057} and both cells. GPU.

### Day 2

- [ ] **A5. Forward knobs + pack gate.** The FWD knob triple through enumerate_knobs; bench/pack_train_attention.py forward half in pack_wide_qmv's verify-then-time order; interleaved arms {ours-fwd, sdpa-fwd, composed-fwd} at both cells, canary withholding. GATE: forward beats composed at both cells and is within striking distance of 4.5 ms at (2,1057). GPU.
- [ ] **A6. verify_grads.** verify.py gains attn_inputs and verify_grads (dq, dk, dv each judged through the battery, same judge(), no gate states its own tolerance). CPU.
- [ ] **F. Lane fixes + merge wave 1.** Apply Lane B's two written fixes in kv-anymac-mr; re-run the two re-reviews that died on the session limit; merge lane/anymac-timing then lane/anymac-metalrunner into lane/metalrunner-sprint1 with /code-review on each merge diff per the standing rule; retarget the tuner's lazy cell import from train_qmm shapes to the train_attention cell surface (the brief anticipated this with the injectable hardware seam). CPU.

### Day 3 (GPU-heavy)

- [ ] **A7. The backward kernels.** ATTN_DQ_MSL (query-tiled, recomputes P from saved L, no second softmax pass) and ATTN_DKV_MSL (key-tiled per kv head, iterates the 4 group members inside the kernel so GQA accumulation needs no atomics; the pre-planned fallback if that 4x Q/dO re-read loses on the clock is a grid.z split plus mx.sum). D = rowsum(dO*O) stays plain MLX. Verify vs the analytic reference at the edge ladder + both cells, including the distinct-constant-per-kv-head GQA test. GPU.
- [ ] **A8. The door + backward timing.** vjp wiring per the settled design; trainer-composition test with real kernels at small geometry; gradients-ALONE timing (clause 48 discipline) interleaved vs composed backward at both cells. This is the first read of the backward-traffic risk and decides short-band routing. GPU.

### Day 4

- [ ] **A9. Operation row.** metalrunner/attention_op.py (mlx only inside callables): replace() hash-checks source, routes only on the exact training signature, declines to stock otherwise; backward_cases as data-only seeded dicts; backward_check judges through verify_grads (the amended rule, NOT array_equal). OPERATIONS gains the row under the already-registered name "attention forward and backward"; measurement.py docstrings state the class split; seam-call asymmetry test (36 forward, adapted-layers backward) via Installed.evidence(). CPU + one GPU smoke.
- [ ] **A10. Dev tune + binding E2E.** Run the tuner end to end on this machine against the attention cells; then the 3-arm bench/train_lora_e2e.py binding run, detached, idle-gated, caffeinated, overnight. GPU, hours.

### Day 5

- [ ] **A11. The number + docs.** Read the binding recording; >= 10% with the interval clear ships; below it, record honestly, the cut-order retreats are the recovery. AGENTS.md module map; final whole-branch /code-review; ledger rulings surfaced.

## Cut order under pressure (first cut first)

1. Overnight LLM generation sessions (ruled).
2. The PIPE=2 double-buffer axis and any simdgroup-matrix path: ship the scalar-fma forward.
3. Independent dq/dkv knob surfaces: collapse to one shared backward dict.
4. Fused backward at the short band only: per-cell routing keeps composed there; the long band carries the claim.
5. The whole fused backward: degraded mode is the fused forward with a plain-MLX vjp from saved L, still rewrite-class, still tolerance-verified; stock is composed in BOTH directions under a trace, so a forward-only win at all 36 layers is still real and claimable (~8-12%).

## Top risks

| Risk | Mitigation |
|---|---|
| fp16-in/fp32-accum numerics vs a naive tolerance | The oracle is the fp64 reference with the ensemble floor; the flash order itself is a floor member; sdpa comparison is a 5e-3 smoke check, never the oracle; running-max subtraction before every exp |
| Backward memory traffic slower than composed | Measured FIRST THING Day 3, gradients alone, before any tuning; per-cell routing and the plain-MLX vjp are pre-planned retreats; backward only fires at the 16 adapted layers while the forward wins at all 36 |
| GQA indexing bugs (silent when heads look alike) | Battery enumerates ratios 2/4/8; distinct-constant-per-kv-head test; the analytic reference contains the group sum so a missing sum cannot pass verify_grads |
| Two-output custom_function under compile/checkpoint | The Day 1 spike pins the arrangement before any Metal; plan B designed and same-day adoptable |
| T=65 tiles wasting the GPU | Grid is tiles x 32 heads x batch >= 384 threadgroups; knobs reach down to RQ=4/BK=16; the bar at the short band is the 3.24x composed path, not fused sdpa; per-cell decline if it still loses |

## Verification

- Suite green with Metal and under KV_FORCE_NO_METAL=1 before every commit; no em dashes anywhere; never git add -A.
- Every kernel gate judges through verify.py and the battery; no gate states its own tolerance.
- Verify-then-time enforced structurally by the funnel; timing through interleaved_arms with the canary; gradients timed alone.
- The binding claim comes only from the 3-arm E2E run's pre-registered rules; the receipt and routing report say exactly what routed.
- The tuner tunes whichever kernel ships; Lanes B and C are unchanged by the operation pivot beyond Task F's cell-surface retarget.
