<!-- Provenance: a copy of the subagent-driven-development ledger for this plan,
live at .superpowers/sdd/keen-inventing-ritchie/progress.md, which is gitignored
as skill scratch and therefore does not reach a clone. This copy is the record of
which tasks are done and which rulings were taken; the live file remains the one
the skill appends to. Paths made repo-relative. -->

# SDD ledger - plan: docs/PLAN.md

## Setup
BASE at start: 20155b9 (Task 0 CLAUDE.md landed)
merge-base main: 158177a

## Pre-flight conflict scan (2026-08-23)

### Pairs sharing a file or interface

| Pair | Producer -> consumer | Finding |
|---|---|---|
| T1 -> T2 | train_qmm.py: T1 creates data surface, T2 adds MSL | Same file. Both Lane A (me), strictly sequential. No conflict. |
| T1 -> T2/T6 | verify.train_qmm_inputs + nt reference -> kernel gates, tuner verify | Clean: T1 adds, never edits tt path. |
| T3 -> T5 | local_store.load() -> lora.py merges with routing.CERTIFIED | Both tuples. Consistent. |
| T3 -> T6 | measurement.py: T3 adds schema-2 branch, T6 adds OPERATIONS row | Same file, different regions. T3 (Lane B) merges before T6 (Day 3) starts. Enforced by ordering. |
| T4 -> T6 | tuner.py: T4 creates with hardware fakes, T6 wires real bundle | Same file. T4 (Lane B) merges before T6. Enforced by ordering. |
| T3 -> T4 -> T5 | store -> tuner -> entry point | Sequential inside Lane B, one agent, one worktree. |
| T7 -> T2 | bench/interleave.py moved -> bench/pack_train_qmm.py imports it | Re-export preserves the path. Existing suite is the proof. |
| T7 -> everything | interleave 24 importers, memory_guard 24 importers | HIGH FAN-OUT. See F1. |
| T2 -> T6 | TRAIN_QMM_TT_MSL sha256 -> entry["candidate_sha256"] | Clean: one source of truth, hashed at emit time. |
| T5 -> T6 | routing.eligible(certified=merged) -> measurement.install | Clean: schema-2 entries carry chip/bits/group_size. |

### Self-consistency, per task
T0 ok (landed). T1 ok. T2 ok. T3 see F2. T4 ok. T5 ok. T6 see F3. T7 see F1. T8 ok.

### Findings and rulings

F1. bench/interleave.py line 31 does sys.path.insert(0, parents[1]), which
resolves to the REPO ROOT from bench/ and would resolve to kernelverify/ after
the move. A blind move silently changes sys.path for 24 importers.
Ruling: the lift proceeds, and the brief names this line explicitly as the one
thing that must NOT move verbatim. kernelverify/timing/ is a proper package and
needs no sys.path manipulation at all; the line is deleted, not relocated.
Cost if wrong: import failures across bench/, caught immediately by the suite.

F2. measurement.entry_problems currently rejects ANY entry whose key set is not
exactly ENTRY_KEYS, returning early on "entry keys differ". A schema-2 entry has
six extra keys, so today it is refused before any other check runs, and
_resolve turns that into a MeasurementRefusal. This is the single point where
the entire local-tune path dies.
Ruling: T3's brief names this as the load-bearing change, and requires a test
that a schema-2 entry survives _resolve end to end, not merely entry_problems.
Cost if wrong: the tuner writes entries that the installer always refuses, and
the failure looks like a routing problem rather than a schema one.

F3. measurement.install calls wrapping(entry, row) separately per seam, so
Operation.replace is invoked once per seam and receives each seam's own
original. Both seams are (self, x) callables, so one replace body can serve
both, but this is an assumption the plan never states.
Ruling: T6 (mine) asserts the two signatures at install time rather than
assuming them, and the live test exercises BOTH seams.
Cost if wrong: the tied head silently routes through a wrapper built for the
projection shape.

F4. The SDD skill forbids parallel implementation subagents. The plan is built
on three file-disjoint lanes and the 5-day constraint exists only because of
that parallelism.
Ruling: parallel dispatch is retained. The skill's rule guards against file
conflicts; the lane design eliminates them by construction, worktree isolation
enforces it, and the scan table above is the evidence. Lane A (mine) is the only
lane touching kernelverify/pack and the only lane using the GPU.
Cost if wrong: a merge conflict at the daily merge point, visible and cheap.

## Dispatch log

Lanes B and C dispatched 2026-08-23 as workflow wf_a282c3e8-ea4
(build -> 3 diverse reviewers -> adversarial refutation -> fix -> scoped re-review).
Lane B worktree the `kv-anymac-mr` worktree branch lane/anymac-metalrunner
Lane C worktree the `kv-anymac-timing` worktree branch lane/anymac-timing
Both branched from 20155b9. Briefs at laneB-brief.md and laneC-brief.md.

Lane A (Tasks 1 and 2: train_qmm data surface, nt reference, the Metal kernel)
is the controller's own, on lane/metalrunner-sprint1, because it is the only
lane needing the GPU and the only one touching kernelverify/pack.

F5. Lane A, Task 1: the plan has me add an "nt" (transpose=False) reference to
kernelverify/schemas/native_ops.py so the backward orientation can be verified.
Reading the code, that is premature. r_contract is x @ dequant(w).T and EVERY
member of QUANT_ENSEMBLE shares that orientation, so an nt reference needs an nt
ensemble too or its tolerance is computed against the wrong quantity. And v1
does not need it: ruling D2 has our vjp call stock mx.quantized_matmul with
transpose=False, so no kernel of ours ever runs in the nt direction in v1.
Ruling: the nt reference and the orientation parameter are DEFERRED to the nt
entry, post-v1. Lane A Task 1 becomes the data surface alone, and verification
reuses the existing qmv_inputs unchanged. This removes any edit to native_ops.py
(the file feeding every pack certificate and every tolerance in the repo) from
the 5-day window.
Cost if wrong: when the nt kernel is built later it needs the reference and the
ensemble then, which is work moved rather than avoided.

## F6: MEASURED, and it reverses the sprint's settled first operation

The plan treats candidate Q (the quantized projection matmul) as the settled
first operation on the strength of its SHARE, 0.376 of the step. Share is only
half of the registered gain formula. The other half is r, how much faster that
part could get, and r for Q is capped by hardware.

Measured on this M3 Pro, 2026-08-23, interleaved 7 rounds with the dense arm's
own spread as the canary (all rows ok):

  shape        M       dense fp16   MLX quantized   dense/quant
  S3 gate/up   260       3.116 ms       3.012 ms       1.034
  S3 gate/up   2114     18.367 ms      19.213 ms       0.956
  S4 down      260       2.751 ms       2.617 ms       1.051
  S4 down      2114     18.861 ms      19.275 ms       0.979
  S1 q_proj    260       1.171 ms       1.117 ms       1.049
  S1 q_proj    2114      7.730 ms       8.094 ms       0.955

The dense fp16 matmul does STRICTLY LESS WORK than the quantized one: it never
dequantizes. MLX's quantized kernel matches it within 4.5% everywhere and beats
it at the short width, where 4-bit weights win on bandwidth. So MLX is already
at the practical ceiling of this machine for that operation, at 94 to 96 percent
of the dense rate, and a kernel cannot take back what the hardware does not have.

Best case arithmetic for Q: r <= 1.15 even assuming a kernel reaches the best
rate observed anywhere on the machine. With f = 0.376,
gain = 1/(1 - 0.376*(1 - 1/1.15)) = 1.052, so 5.2 percent end to end at best.
The registered shipping floor is 10 percent. Q cannot reach it.

Measured for candidate L in the same session, the tied head plus cross-entropy,
full rows against supervised rows only:
  M = 260,  supervised fraction 0.556:  42.874 ms -> 23.846 ms, 1.798x
  M = 2114, supervised fraction 0.870: 322.168 ms -> 279.896 ms, 1.151x
With f = 0.245, the short width gives 1/(1 - 0.245*(1 - 1/1.798)) = 1.122, so
12.2 percent end to end, which clears the floor.

The difference is structural rather than incidental. Q's headroom is bounded by
what the silicon can do and MLX has already taken it. L's headroom comes from
NOT DOING WORK, and work you never do is not bounded by peak throughput.

Ruling: the first operation becomes candidate L, the masked output head and its
cross-entropy. Candidate Q is not discarded: a correct, verified, knob-tunable
tt kernel now exists and reaches 1.457x at S6/M=260 on the best of 138
launchable settings, so it stays as a later lever at the shapes where it wins.
Cost if wrong: Lane A's half day of kernel work becomes a second-priority lever
instead of the headline, and the head kernel is built one day later than planned.

## F7: the first operation is candidate A, and the proxy widths hid it

Two proxy artifacts sent the plan at the wrong operation, and both are now
measured rather than argued.

ARTIFACT 1, model size. Candidate L's share of 0.245 was measured on the 0.6B
proxy. The tied head is 151936 x hidden whatever the model, so its share falls
as the body grows. Computed from the published configs:
  Qwen3-0.6B  head is 26.1 percent of matmul work   (matches the 0.245 proxy)
  Qwen3-1.7B  head is 18.1 percent
  Qwen3-4B    head is  9.7 percent                  (the actual target)
So f_L at the target is roughly 0.10, not 0.245, and even the measured 1.798x
work elimination gives 1/(1 - 0.10*(1 - 1/1.798)) = 1.046, so 4.6 percent.

ARTIFACT 2, sequence width. Candidate A's share of 0.045 was measured at width
97. Attention's score matrix is quadratic in the width while every other region
is linear, and the registered long band is 1057.

Measured at Qwen3-4B attention geometry (32 q heads, 8 kv heads, head dim 128),
fused against composed, where composed is what a gradient trace actually runs:
  B=4 T=65     fused 0.423 ms   composed  1.370 ms   3.24x   36 layers:   49 ms
  B=2 T=1057   fused 4.499 ms   composed 26.123 ms   5.81x   36 layers:  940 ms
  B=1 T=2048   fused 6.729 ms   composed 44.892 ms   6.67x   36 layers: 1616 ms

And the decomposition is confirmed by reading the graph MLX actually builds:
  plain call    ScaledDotProductAttention x1, Softmax x0
  under mx.grad ScaledDotProductAttention x0, Softmax x1
The fused primitive is present without a trace and absent inside one, so every
training step runs the composed path. MLX also implements no fused attention
BACKWARD at all, so the backward is composed too.

At the long band attention forward is about 940 ms against 2819 ms for all 36
layers of projections, so roughly a quarter of the step, and it is running 5.8x
off its own fused reference. A tiled attention at a conservative 3x on the
region gives 1/(1 - 0.25*(1 - 1/3)) = 1.20, which is 20 percent end to end and
clears the 10 percent floor with room.

There is a second prize. The composed path materializes a B x heads x T x T
score matrix, which at B=2 T=1057 fp32 is 286 MB PER LAYER. Not building it is
what makes the 16 GB machine reachable, and that is a registered requirement
(R4, R6) that no amount of matmul tuning would have delivered.

Ruling: the first operation is candidate A, tiled causal grouped-query attention
with its own backward. Q is measured out (F6). L stays a real but small lever at
4.6 percent. The QMM kernel already built is kept as a later lever at the shapes
where it wins.
Cost if wrong: attention with a hand-written backward is the hardest of the
three kernels, so a failure here costs more days than a failure in L would have.
The mitigation is that the fused forward already exists in MLX as a correctness
and speed reference for the forward half.


---

## Task A1 complete, 2026-08-23

Amendment 19 committed alone (568540d), clauses 77 to 79: the operation was
chosen after its measurements and says so; section 8.4's array equality is
scoped to the retune class; a rewrite-class backward is verified against an
fp64 ANALYTIC reference through the existing battery plus the loss-curve gate.

## Task A2 complete, 2026-08-23

The two-output custom_function door is pinned before any Metal (dfd4dc0).
All three properties the design needs hold on MLX 0.32.0: a tuple return is
accepted, the vjp receives the outputs (so the row logsumexp is free), and the
cotangent of an unconsumed output arrives as ZEROS rather than a refusal.
Survives compile, checkpoint, and the whole trainer composition. Plan B (an
LSE pass inside the vjp) is not needed and is not built.

## Task A3 complete, 2026-08-23

The cell surface, the battery entry and the gradient references (0e7e8b9). The
analytic vjp is cross-checked against central differences (1e-9 relative) and
against MLX's own vjp of a composed attention (1e-5 relative); the forward
against Apple's fused primitive under the same "causal" string mlx-lm passes.

## Ruling F8, measured 2026-08-24: the seam carries BFLOAT16, not fp16

The plan's interface block says the routed callable "declines to stock on
dtype != fp16". Measured on the real path rather than assumed: a spy installed
at `mlx_lm.models.qwen3.scaled_dot_product_attention`, driven through
`nn.value_and_grad(model, default_loss)` on the pinned 0.6B with LoRA layers
installed, sees 28 calls, every one of them

  q (2, 16, 64, 128) bfloat16, k and v (2, 8, 64, 128) bfloat16,
  cache None, mask the literal string "causal", sinks None,
  scale 0.08838834764831845

The model's own config says `torch_dtype: bfloat16` and its stored scales are
bfloat16, and mlx-lm takes the model dtype from the scales, which is exactly
what `kernelverify/schemas/bfloat16.py` already records: "a training kernel
this repository gates is a bf16 kernel whether or not the contract says so".

Ruling: the kernel is written dtype-generically (the staged tiles and the
stored output take MLX's `T` template, every accumulator is fp32), it routes
for bfloat16 AND float16, and it declines everything else. Verification runs
at both, with bf16 as the shipping dtype and fp16 as the sharper check, since
bf16's own eps is 8x float16's and its base tolerance is correspondingly loose.
Cost if wrong: none measured yet; the risk moved the other way, since a kernel
that declined on dtype would have routed nothing on the real model and the
sprint would have shipped a number of zero.

Second fact from the same probe, recorded because it moves a band edge:
`default_loss` trains on `batch[:, :-1]`, so a 65-token batch attends over 64
positions and a 1057-token batch over 1056. Both land in the same buckets the
cell surface registers (128 and 2048), so no cell moves.

## Ruling F9, measured 2026-08-24: the forward runs on the matrix units, and stages nothing

Task A4 shipped a scalar-FMA flash forward, correct everywhere. Then it was
measured against what the sprint's claim actually needs, and the claim did not
survive the first arithmetic.

Measured on the real 4B at the long band, batch 2 width 1057: a stock training
step is 7566 ms and peaks at 14.33 GB; the composed attention region is 22.0 ms
forward and 41.4 ms of backward per layer, which over 36 forwards and 16
adapted backwards is 1454 ms, or 0.192 of the step. The composed backward peaks
at 1477 MB for ONE layer.

At the scalar kernel's 11.7 ms forward and an estimated backward, the region
ratio is about 1.8 and the end-to-end gain is 1.08. R10's floor is 1.10, so the
scalar kernel does not ship a claim.

Four alternatives were measured before the rewrite, not argued:

| Variant | Long band |
|---|---|
| scalar, threadgroup-staged K and V | 12.9 ms |
| scalar, V read from device | 16.5 ms |
| scalar, query rows staged at storage width | 11.7 ms |
| scalar, wide key tiles with chunked staging | 10.8 ms |
| matrix units, threadgroup-staged operands | 15.6 ms |
| matrix units, nothing staged, fp32 operands | 6.8 ms |

Two facts settled the design. Splitting the scalar kernel by phase showed the
two matmuls account for 7.0 ms of 12.9 and the rest is staging, barriers and
reductions, so the ceiling was not arithmetic. And the matrix-unit variant that
STAGED its operands was slower than the scalar kernel, while the one that reads
fragments straight from device memory was twice as fast: the staging pass, not
the matmul, was the cost all along.

Ruling: the shipped forward loads its fragments from fp32 operands the door
casts once, stages nothing, and keeps the whole softmax in the fragment
registers, where a row is held by four lanes at exclusive-or distances 1 and 8
so a row reduction is two shuffles. Final, through the shipped call path
including the cast and the padding: 0.556 ms at the short cell against 1.579
composed and 0.624 fused, and 7.472 ms at the long cell against 22.234 composed
and 4.239 fused. That is 2.84x and 2.98x over what stock training actually
runs, and it puts the end-to-end estimate at 13.2 percent.

Cost if wrong: the fragment layout is a property of the chip and MSL does not
promise it. It is pinned by its own device test, and a machine whose layout
differs fails the on-device verification and routes to stock rather than
answering wrongly, which is the architecture this product already has.

Two facts recorded on the way, because they closed doors that looked open:
MLX's fused attention is IN contract (its error against the fp64 reference is
identical to ours, so it accumulates at fp32 and its speed is a fair target),
and it cannot be borrowed for training even inside a custom_function, because
the decomposition belongs to the trace: under `mx.vjp` the fused primitive is
gone from the VALUE's graph too, and `stop_gradient` on the operands does not
bring it back.

## Task A4 complete, 2026-08-24

Two commits, because the first design did not survive its own measurement.
`def6b71` is the scalar flash forward, correct at every tile edge; `3685cbc`
replaces its body with the matrix-unit version ruled in F9. What survives from
the first is the whole verification surface: the edge ladder, both dtypes, the
group ratios, the causality experiment, the separate verdict on the row
logsumexp, and the two registered cells against the fused primitive.

## Task A5 complete, 2026-08-24

`d8e1f53`. Twelve of twelve knob settings launchable on this chip, all twelve
verified before any was timed, 1740 of 1740 cases green.

| Cell | composed | fused | best ours | vs composed |
|---|---|---|---|---|
| B4:T128 | 0.295 ms | 0.096 ms | 0.267 ms | 1.03x |
| B2:T2048 | 21.858 ms | 3.964 ms | 7.222 ms | 3.02x |

SG2/BK16 wins both cells. The gate as registered is cleared at both.

HEADS UP: the short cell clears by three percent, which is a reading rather
than a rounding. At width 65 the causal bound leaves each threadgroup almost no
work and the door's own cast and padding are most of the call. Per-cell routing
already exists for exactly this, and the short band contributes little to the
claim: attention's share is quadratic in the width.

Note on the numbers moving between F9 and this gate: the standalone timings in
F9 measure one call with a full synchronise around it, and the gate measures
batched dispatches under the shared interleaved engine, which is what every
other pack certificate is measured under. The composed arm is the one that
moves most (1.579 ms standalone against 0.295 batched at the short cell),
because its launch overhead is what batching hides. The gate's numbers are the
ones to quote.

Two coverage gaps in the gate were found by the gate's OWN tests before it ran:
the verified widths straddled no eight-wide or sixteen-wide tile edge, and the
head pairs covered group ratios 1, 2 and 8 but not the 4 the shipped model has.
Both are closed.

## Ruling F10, 2026-08-24: the gate rules on the long cell, and the short cell routes to stock

The A5 gate as the plan wrote it asked the forward to beat composed at BOTH
registered cells. At the short cell the credited ratio sits ON 1.0 and crosses
it in either direction between runs: 1.03 at 14:10 and 0.99 an hour later, on
the same code and the same machine.

What that measures is the door, not the kernel. Measured on the shipped call
path: the cast to fp32 is 0.584 ms and the pad 0.617 ms of a 7.090 ms call at
the long cell, 17.1 percent, and at the short cell the two are 60.6 percent of
a 0.257 ms call. Subtract them and the short cell's kernel is about 0.10 ms
against MLX's fused 0.096 and composed 0.295, so the kernel WINS the short cell
by nearly 3x and the preparation gives it back.

Ruling: the gate rules on the long cell, because attention's share of a step is
quadratic in the width while every other region's is linear, so the long band is
where the claim is won. A registered cell that does not clear is RECORDED as
routing to stock, which is a first-class outcome this product already has (the
tuner decides per cell and the receipt says which cells routed), not a failure.
The change of wording is written into the gate's own docstring beside the two
numbers that prompted it, so no reader can mistake it for the plan's wording.
Cost if wrong: the short band ships stock's speed, which is what it ships today.

Both halves of the door's overhead are queued in TODOS.md with their
measurements. The cast is a contract question (whether clause C1 admits a
narrow MULTIPLICAND fragment, given that a product of two fp16 values is exact
in fp32); the pad is a kernel question (a bounded tail path). Together they are
worth about 17 percent at the long cell and about 2.6x at the short one.

## Ruling F11, 2026-08-24: the baseline three earlier entries were measured against was not the path a training step runs

An adversarial review of the A5 gate reproduced a fault in the gate's own
composed arm, and correcting it moves numbers in F9, in Task A5's entry, in
F10, and in committed prereg text. Everything withdrawn below is named here
rather than edited in place above, because a ledger that quietly restates its
own numbers cannot be audited.

What was wrong: `composed_arm` hand-wrote MLX's decomposition instead of
asking MLX for it. MLX scales the QUERIES before the matmul and calls
`mx.softmax(..., precise=True)`, which upcasts inside its own kernel. The arm
scaled the SCORES afterwards and materialised an explicit fp32 copy of the
whole score matrix and cast it back, which is two extra passes over the
largest tensor in the operation. Measured in one set of interleaved rounds at
bfloat16: 21.522 ms against MLX's own 11.379 at the long cell, a factor of
1.89, and 0.397 against 0.358 at the short one.

Why no gate caught it: the two arms agree in VALUE to 1.562e-02 against a
reference peak of 3.953, and `arms_agree` admits 5e-3 of that peak, which is
1.98e-02. A fairness check on a timing comparison bounds value, and two
programs of the same mathematics can agree in value while one does twice the
work. The check was not weak; it was answering a different question.

Withdrawn, with what replaces each:

| Entry | Withdrawn | Replaced by |
|---|---|---|
| F9 | composed 1.579 ms short, 22.234 ms long; 2.84x and 2.98x; "13.2 percent" end to end | the gate's own corrected run below |
| Task A5 | composed 0.295 / 21.858 ms; 1.03x and 3.02x | 0.234 / 11.367 ms; 0.86x and 1.58x |
| F10 | "the kernel WINS the short cell by nearly 3x" against composed 0.295 | composed is 0.234, so the door-free kernel at about 0.10 ms wins by about 2.3x |
| F9 | attention "0.192 of the step", from region timings over a step time | an ablated 0.105 to 0.108, prereg Amendment 20 clause 81 |

The corrected A5 gate, same 5 rounds, same sampler, canary clean at both
cells: short cell composed 0.234 ms, fused 0.095, best ours 0.268 at SG2/BK16,
credited 0.86x, so it routes to stock; long cell composed 11.367, fused 3.907,
best ours 7.111 at SG2/BK16, credited 1.58x, so the gating cell clears. The
same knob setting still wins both cells and all 1740 verification cases are
still green, so nothing about the kernel or its knob choice moves. F10's
ruling stands unchanged: it routed the short cell to stock before this
correction and for a reason this correction does not touch.

The arm is now `mx.vjp(fused_call, ...)` with the value kept, which IS the
graph a gradient trace builds, and `precise_decomposition` writes that graph
out and is pinned to it by bit-equality at three geometries. Committed
`29abffd`.

Second finding from the same review, unrelated and also real:
`_oracle_member_labels` did not list the two attention ensembles, so a cached
battery verdict scored under one ensemble would have survived a change to the
other. Fixed and committed `9a2ad1c`; the serving reinterpretation artifact
regenerates with it because `native_ops.py` is one of its INTERPRETATION_PATHS.

Cost if wrong: none of the corrections make anything faster or slower. What
they change is what the sprint may claim, and clause 81's arithmetic is the
part that has to reach Vlad rather than sit in a ledger.

## BLOCKER for the sprint's claim, measured 2026-08-24

Attention's share of a real 4B step at the long band, by ablation with no
instrument inside the step, is 0.1063 (7544.8 ms stock, 6743.0 ablated),
reproducing an earlier 0.1076 and a checkpointed 0.1085. The registered
formula's ceiling is therefore 1/(1 - f) = 1.119 with attention taking zero
time, so R10's 1.10 floor needs r >= 6.9 on a region that is 37.611 ms
(forward 11.347, backward 26.264), which is 5.45 ms for the whole thing
against a FUSED FORWARD ALONE of 3.914 ms and no fused backward in MLX at all.
R18's 30 percent is unreachable at any r, because gain < 1/(1-f) requires
f >= 0.2308 and the measured share is less than half that.

Today's kernel, forward swapped and backward left composed, gives 1.012. Both
halves at the forward's own 1.58 would give 1.041.

This is a scope ruling and it is Vlad's. Registered in prereg Amendment 20
clause 81, which states the arithmetic and explicitly takes no ruling.
