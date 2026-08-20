# metalrunner sprint 1, pre-registered

Status: written and committed BEFORE the profile harness exists, before any candidate kernel exists, and before any measurement.
Sections 1 to 10 are the rules.
Section 11 is written after the sprint and records what the rules produced.

Lane: `lane/metalrunner-sprint1`.
Plan: the approved sprint plan of 2026-08-19, rulings R1 to R18.
Parent evidence: `docs/research/2026-08-18-batch-decode-e2e.md` for the pre-registration form this follows, ADR 0010 for detached binding runs, ADR 0015 for how a recording becomes a routing table.

## 1. The question

For someone running a QLoRA fine-tune of a 4-bit Qwen3-4B on an Apple Silicon Mac through `mlx_lm.lora`, does a kernel produced by the generate-gate-price-keep loop make the whole training job measurably faster, end to end, with every substituted kernel verified before it was ever timed?

The comparison is `mlx_lm.lora` run by itself against the same `mlx_lm.lora` run with verified kernels swapped in at mlx-lm's module seams, on the same model, the same data, the same order, the same seeds and the same memory ceiling.
It is spelled that way wherever it is quoted.

Why this question and not a kernel question.
Every number this repository has published so far is a kernel number: a matmul measured against another matmul in a loop this repository wrote.
A kernel ratio is not a user-visible fact, because no user runs that loop.
The code a user runs is mlx-lm's trainer, and until a number comes out of that trainer nothing here is a claim about training being faster.

## 2. What is frozen by this document

The following are decided here and may not be chosen after seeing a number.

- The Day 1 profile's cells, its instrument, and what it records (section 3).
- The rule that selects the first operation from the profile (section 4).
- The kill rule that can remove a candidate operation from contention (section 5).
- What a candidate must survive before it may be timed, and the order the funnel runs in (section 6).
- The held-out draw and its commitment (section 7).
- The end-to-end arms, the fairness conditions, the metrics and the noise floors (section 8).
- The pre-registered outcomes and the GO condition (section 9).

Anything in sections 1 to 10 that turns out to be wrong is amended in writing, in this file, with the date and the reason, BEFORE the code that depends on it moves.
That is the same amendment discipline the batch-decode pre-registration carries, and it exists because a rule edited after a number is not a rule.

## 3. The Day 1 profile

### 3.1 What is measured

One binding profile of stock `mlx_lm.lora` on the pinned artifact, with no kernels of ours installed anywhere.
It answers two things and nothing else: where the training step's wall time goes, and how far each candidate operation is from a measured floor.

### 3.2 Cells

Target: `bench/.models/qwen3-4b-4bit-g64`, the pinned 4-bit group-64 Qwen3-4B, whose geometry is hidden 2560, intermediate 9728, 32 query heads, 8 key-value heads, head dimension 128, 36 layers, vocabulary 151936, tied input and output embeddings.

Fixed: sequence length 2048, LoRA rank 8, the adapter layer count mlx-lm defaults to, one optimizer, one schedule, one seed.

| Cell | Batch | Supervision |
|---|---|---|
| A | 1 | prompt-masked instruction data |
| B | 4 | prompt-masked instruction data |
| C | 1 | all tokens supervised |
| D | 4 | all tokens supervised |

Cell B is the primary cell, because it is the case R18's target arithmetic was budgeted against and the case a normal instruction fine-tune is.
Cell C is the honest secondary, because it is the case where the loss lever does not exist and where the achievable number is lower.

The roofline sweep runs at the five distinct projection shapes the model actually contains, at the token counts the cells produce, so M is 2048 at batch 1 and 8192 at batch 4.

| Shape | (d_out, d_in) | Where it appears |
|---|---|---|
| S1 | (4096, 2560) | q_proj |
| S2 | (2560, 4096) | o_proj |
| S3 | (9728, 2560) | gate_proj and up_proj |
| S4 | (2560, 9728) | down_proj |
| S5 | (151936, 2560) | the tied output head |

### 3.3 Instrument

Per-operation wall time comes from MLX's own evaluation boundaries around named regions of the step, taken in a run whose only instrumentation is those boundaries, and the sum of the named regions plus one explicit unattributed remainder must equal the measured step time.
A profile whose named regions do not reconcile to the step time within 2% is rejected, because a share computed from an incomplete decomposition is a plausible wrong number and this repository refuses those.

The share f of an operation is its wall time divided by the full step wall time, both from the same run, both medians over the eligible rounds.

### 3.4 Run discipline

The profile is a binding run: detached through `bench/start_binding_run.sh`, holding the single measurement lock, idle-gated before and after, under a declared memory budget, on AC power, with Vlad's explicit go for that run.
A run that goes non-idle at the closing gate quarantines its samples and binds nothing, exactly as the pricing probe does.
Five rounds; per-cell medians and spreads; a spread past the class limit rejects the round as the machine rather than the workload.

## 4. The selection rule for the first operation

Three operations are in contention.
The rule below is computed from the Day 1 profile and picks one.
No argument, no preference and no prior report may substitute for it.

### 4.1 The gain formula

For an operation with measured share f and per-operation ratio r, the end-to-end gain if that operation alone is replaced is

    gain = 1 / (1 - f * (1 - 1/r))

This is Amdahl's relation written for one accelerated part; f and r come from the profile and nothing else enters it.

### 4.2 The ratio each candidate is credited with

Each candidate is credited with `r_lo`, the lower endpoint of a measured interval, computed the way the pricing probe computes `ratio_lo`: the smallest numerator sample over the largest denominator sample, so the credited ratio is the worst pairing the samples permit.
The numerator is stock's measured cost of that operation.
The denominator is a measured floor, defined per candidate below, and each floor's assumption is named here rather than discovered later.

| Candidate | Numerator | Denominator (the floor) | The assumption in the floor |
|---|---|---|---|
| L, streamed mask-aware output head and cross-entropy | stock's output-head matmul plus cross-entropy, forward and backward, at S5 | the same stock operations timed on only the supervised rows of the same cell, plus one measured extra matmul at S5's backward shape standing for the hidden-gradient path | that a streamed kernel can reach stock's throughput on the reduced problem and pays one extra matmul for the gradient it must produce |
| A, tiled attention forward and backward | stock's attention region, forward and backward, at the cell's shapes | MLX's own fused attention forward timed at the same shapes on the inference path, multiplied by 3 to stand for forward plus backward | that the fused backward, which MLX does not implement on the GPU, would cost about twice its forward, the standard relation between attention's backward and forward matmul work |
| Q, quantized matmul retune at training widths | stock's quantized matmul time at S1 to S5 | the dense fp16 matmul at the same logical shape on the same machine | that the dense fp16 matmul is a true ceiling for a quantized kernel of the same shape, which it is because it does strictly less work per output |

Both multipliers in the table, the extra matmul for L and the factor 3 for A, are pre-registered assumptions and are reported as assumptions beside any number they produce.
A floor is a measurement of what the operation could become, not a promise that a kernel will reach it; the credited gain is therefore an upper bound on that candidate's end-to-end effect and is labelled as one everywhere it appears.

### 4.3 The rule

Compute `gain` for each of L, A and Q at the primary cell B.
The largest gain selects the first operation.

Tie-breaking, in order: a difference of less than 2 percentage points in end-to-end gain is a tie; a tie goes to the candidate with the smaller measured peak-footprint delta in the profile; a remaining tie goes to the earlier row in the table above.

If the largest gain at cell B is below 1.10, no single operation can reach R10's shipping floor by itself.
In that case the sprint does not pick one operation; it records that reading, keeps the top two, and Day 2 onward builds them in gain order.
That branch is registered here so that a weak profile produces an honest plan rather than a hopeful pick.

The same computation is reported at cells A, C and D beside the decision, and it decides nothing.
If cell C's ordering differs from cell B's, that disagreement is reported in the record, because it is exactly the fact a reader needs when the headline is scoped to masked data.

## 5. The kill rule

Candidate Q is removed from contention if, at 4 of the 5 roofline shapes at the primary cell's token count, the stock quantized matmul is already within 1.10 of the dense fp16 ceiling.
An operation that close to a ceiling that does strictly less work has no headroom worth a sprint, and saying so before the measurement is what makes the reading binding.

No other candidate has a kill rule, because neither L nor A has a comparable ceiling that can be measured directly.

## 6. The funnel: what a candidate must survive before it may be timed

The order is fixed and no stage may be skipped for a candidate that failed an earlier one.

1. Compile.
   A candidate that does not compile is stored with its compiler error and goes no further.
2. Source lint.
   The working-precision clause of the contract is attested from the source: no half or bfloat tile or accumulator outside the load and store idioms.
   A violation names the offending token and stops the candidate.
3. Tolerance-free gates.
   Unwritten-output detection by NaN sentinel, guard rows past the declared output, NaN and infinity propagation, four-way determinism including one fresh worker process, and a shape change at awkward sizes.
   These have no tolerance in them and therefore no false positives by construction, which is why they run before anything that does.
4. Tolerance gate.
   The contract gate, judged against the fp64 reference at the calibrated K, extended for training with gradient cases, mask and padding patterns, and the tied head.
5. Pricing.
   Only a candidate that is green at stages 1 to 4 may be timed, ever, by anything.
6. Held-out sweep.
   Only a frozen incumbent enters the sealed draw of section 7.

Rule V1: no candidate is timed before it is verified.
A timing produced for an ungated candidate is not evidence and is not recorded as a number; it is recorded as a protocol violation.

Rule V2: the generator sees evidence objects, never raw process output.
Feedback is built from the gate's own structured result, so a candidate cannot be advanced by anything that was merely printed.

## 7. The held-out draw

Every frozen incumbent is checked on a set of cases it was never optimised against.

The draw is per candidate and seeded by `sha256(candidate_source_sha256 + salt)`, where `salt` is a 64-hex-character secret generated on 2026-08-19, stored at `bench/.heldout_salt`, which is gitignored and never enters the generator's context.

The salt's commitment, published here before any draw:

    sha256(bench/.heldout_salt) = d5a9492c6466660e0d39c90beaae27a3bdaf1402ce00c8697432d129cd94d389

The salt itself is published in section 11 when the sprint closes, so any reader can recompute every draw and confirm it was not chosen to flatter a candidate.

The draw covers, for each candidate: a token-count band, a tile-edge shape whose dimensions are deliberately not multiples of the candidate's tile, one projection family, the dtype, and the input distribution.
The result reaches the generator as PASS or FAIL and nothing else, so a failing held-out case cannot leak its content into the next candidate.
A held-out failure never enters the generator's lineage as feedback.

A LOSS or a gate failure inside a candidate's claimed window shrinks that window to the region that survived, or voids the claim if nothing survives.

## 8. The end-to-end measurement

### 8.1 Arms

Three arms, all driven through the same `mlx_lm.lora` entry point, differing only in what is installed.

| Arm | What it is |
|---|---|
| 1, ours | verified kernels installed at the module seams and routing |
| 2, stock | mlx-lm untouched |
| 3, control | our wrapper installed, every routing decision forced to stock |

Arm 3 against arm 2 is the wrapper's own host cost and is reported at every cell.
Arm 1 against arm 3 is the kernel's contribution net of that cost.
Arm 1 against arm 2 is the number a user would feel, and it is the one quoted.

Each arm runs in a fresh subprocess, one arm per round, five rounds, with the starting arm rotated each round so no arm always takes the coolest slot.

### 8.2 Fairness conditions, all mandatory

A round is invalid unless every one of these holds.

- Same base model file, same sha256.
- Same adapter initialisation, compared by hash of the initial adapter weights.
- Same examples, same order, same truncation, same masks, and therefore the same count of supervised tokens per step.
- Same optimizer, same schedule, same seeds, same number of steps.
- Same memory ceiling declared to every arm.
- Same mlx and mlx-lm versions, pinned at 0.32.0 and 0.31.3 for this sprint, checked by hash at process start.

A speed number produced from arms that differ in supervised token count is not a speed number, and this is stated here because it is the easiest way for a training benchmark to lie.

If arm 1 fits a larger batch than arm 2 at the same memory ceiling, that is reported as throughput at equal memory, never as same-batch speed, and it is reported in its own row.

### 8.3 Metrics and floors

Primary metric: full-job wall time for a fixed step count, from process start to adapter written.
Secondary metric: warmed steady-state time per step, medians over rounds after the first.
Reported beside both: peak memory footprint per arm, and the loss curve summary per arm.

The noise floor for a comparison is the larger of the two arms' spreads over the eligible rounds.
A delta equal to its floor does not clear it.

The verdict per comparison is an interval verdict, computed the way the pricing probe computes one: WIN only if the whole ratio interval sits above 1.0, LOSS only if the whole interval sits below, REFUSED otherwise.

### 8.4 Correctness alongside speed

The kept kernel's backward is compared to stock's gradient in isolation by exact array equality on fixed cases before the end-to-end run, and a failure there stops the run.

The end-to-end run reports the loss curve of every arm.
A divergence in the loss curve beyond what the same-seed stock arm reproduces against itself is a finding that voids the speed claim for that cell, because a faster run that trains a different model is not a faster run.

## 9. Pre-registered outcomes

**O1, the shipping question**, at the primary cell B, arm 1 against arm 2 on full-job wall time.
GO if the ratio interval's lower endpoint is strictly above 1.10, which is R10's floor, AND arm 1's peak footprint is no worse than arm 2's.
Anything else is NO-GO.

**O2, the target question**, same comparison, reported beside O1 and never substituted for it.
`AT-TARGET` if the interval's lower endpoint is at or above 1.30, which is R18's design target.
`BELOW-TARGET` otherwise, with the measured value stated.

**O3, the honest secondary**, at cell C, all-token supervision, same comparison and same floor rule.
Its verdict is reported with equal prominence to O1 and is never omitted from a quotation of O1.
A GO at cell B with a REFUSED at cell C ships as a claim scoped to prompt-masked instruction tuning, in those words.

**O4, the wrapper's cost**, arm 3 against arm 2 at every cell, reported before O1 is read.
A wrapper cost whose interval sits below 1.0 by more than the floor is a finding against the interface and is reported whatever O1 says.

**O5, the loop's own outcome**, reported whatever the kernel did.
The record states how many candidates were generated, how many died at each funnel stage, how many were priced, how many were kept, and the wall time the loop consumed.
A sprint that keeps no kernel still reports O5, because the loop's yield is the thing being built and a zero is a measurement of it.

A NO-GO on O1 closes the sprint-1 speed claim on this operation, on this machine, on this model.
It does not close the loop, and it does not license a second attempt at the same claim with a rule changed after the fact.

## 10. What this does not claim

Nothing about a 16 GB machine.
Every number in sprint 1 is measured on the M3 Pro 36 GB and is scoped to it in writing; the 16 GB shipping number waits for the second Mac, per R16.

Nothing about models other than the pinned Qwen3-4B 4-bit group-64 artifact, and nothing about 8-bit, fp16 bases, or group sizes other than 64, per R9.

Nothing about full fine-tuning or DoRA, which the wrapper refuses.

Nothing about sequence lengths other than 2048, or LoRA ranks other than 8, unless a cell is added here in writing first.

Nothing about chips other than the one measured; uncertified chips route to stock rather than inheriting a window.

Nothing about the word "verified" in public copy until the mutation battery has run over the training policy with a measured escape rate, per R11.
Inside this document "verified" means what the funnel of section 6 enforces, and nothing more.

Nothing about mlx or mlx-lm versions other than 0.32.0 and 0.31.3; on any other version the wrapper installs nothing and refuses loudly.

## 11. What the rules produced

Written after the sprint.
This section records the profile's shares and floors, which operation the section 4 rule selected, the funnel census, the held-out draws with the salt revealed, and the end-to-end verdicts O1 to O5 as measured.

---

## Amendment 1, 2026-08-19: the tolerance-free gates are not free of false positives, and three of five need a rule that does not exist

Section 6 item 3 named five tolerance-free gates and asserted that "these have no tolerance in them and therefore no false positives by construction".
That assertion was wrong, and the reason it was wrong is worth more than the gates it cost.

Five independent designs were built against the real runner API and each was then handed to an independent reviewer whose instruction was to refuse it.
All five were broken, and every one of the five was broken twice: a correct kernel it would refuse, and a kernel carrying its own named fault class that it would pass.

### What the assertion missed

Having no tolerance in it stops a gate flagging a correct kernel for being numerically different.
It does not stop a gate flagging a correct kernel for being structurally different, and structure is exactly what these five gates read.
Three of the five refuse kernels the tolerance contract admits:

| Gate | What it assumes | What the contract grants |
|---|---|---|
| Unwritten output | a kernel does not read its own output buffer | C2 and C3 grant a reduction blocked at any width, and on Metal a reduction blocked across threadgroups can only be assembled through the output buffer |
| Guard rows | every store lands inside the declared extent | C3 explicitly grants padded-block formulations, and Metal's simdgroup store has no partial-store variant, so a correct tiled matmul spills past a ragged edge |
| Determinism | the same case produces the same bytes twice | C2 grants any evaluation order, so two threadgroups each computing an admissible order and both storing is admissible; `kernelverify/report/certificate.py` already states in its own protocol block that "output values are NOT asserted to be identical, which is not achievable on a GPU" |

Verified directly rather than accepted: the contract module contains no determinism requirement of any kind, and the certificate's protocol assertion reads as quoted.

### The ruling

Where a gate would need a new rule about what a kernel may do, it abstains instead of refusing.

An abstention says the gate has nothing to report about this candidate, records why, and counts as no coverage.
It is not a pass: a case that was abstained is excluded from the screened count, so a session can never present abstentions as things it checked.

This is what makes the unwritten-output gate buildable without a contract change.
Rather than forbidding kernels that read their own output, it detects them: a kernel that does not read its output writes identical bytes under two different fills, so any cell that differs between the fills without being unwritten is proof the fill was read, and the case is not screened.
Abstaining costs coverage; refusing would have cost a promise this repository has kept.

Guard rows and determinism have no equivalent escape, because abstaining on the very property they test would leave them testing nothing.
They are therefore NOT built in sprint 1 and are recorded here as blocked on a decision that belongs to a separate ruling: whether kernelverify's compiler loop declares, as its own policy and not as a contract amendment, that it will only keep kernels that are deterministic and stay inside their declared extent.
That decision is deliberately left out of this sprint.

### Mechanism change to the unwritten-output gate

Section 6 said "NaN sentinel". The gate as built uses two dispatches per case with complementary byte fills, 0xA5 and 0x5A, refusing only cells that came back holding both.

Two reasons, neither of them preference.
A shipped and already-verified kernel in this repository writes NAN into every declared output cell on its capacity-overflow path, and the next gate in the funnel exists to run kernels whose correct output is NaN, so "a NaN left behind means unwritten" refuses correct kernels.
And half of the runner's tensor dtypes are integer carriers where no value is out of range, so no single sentinel value can be safe for them; a byte fill is dtype-blind.

### Scope correction to the same gate

A clean verdict attests that a store reached every declared cell.
It does NOT attest that the kernel produced each cell's value.
A kernel that clears its own output and then applies a wrong bound stores into every cell, passes this gate, and computes half of them wrongly; no choice of fill pattern changes that, because the mechanism can only observe whether a store arrived.
That sentence is now in the gate's policy string and travels with every verdict it issues, so a clean result can never be read as coverage of the wrong-bounds class.

One residual hole is named rather than hidden: a kernel that reads its output AND whose every declared cell is absorbed by the fill pattern shows no differing cell, so the abstention does not trigger.
The abstention fires the moment any one cell is not absorbed.

### Correction to a claim made during the review

The reviewer of the awkward-shapes gate argued it has no work to do, on the grounds that every shape the operator dispatches at is a whole number of 256-wide tiles.
That is false at one shape, and it is the shape that matters most here.
The output head is 151936 wide, which is 593.5 tiles of 256, a remainder of 128; every other dispatched dimension is exact.
So exactly one awkward shape exists in the operator's own domain today, and it is the output head, which section 4 lists as candidate operation L.
The reviewer's structural point stands and is adopted: the shape domain must come from the operator's own producer, never from an incumbent kernel's docstring, and outside that domain the gate abstains rather than refusing.

### Effect on the sprint

Section 6's funnel is unchanged in order.
Stage 3 ships with one gate rather than five in sprint 1: unwritten output, built as described above.
NaN and infinity propagation and awkward shapes remain buildable without a ruling and are narrowed by their reviews; guard rows and determinism are blocked as recorded.
This amendment is written after the unwritten-output gate's code landed rather than before, which is late by this document's own rule, and is recorded as such.

---

## Amendment 2, 2026-08-19: a sixth gate, input support, built after two adversarial rounds

The NaN-dye review measured a blind spot no planned gate covers: for a quantized matmul the packed integer codes carry essentially all the weight data, no out-of-range value exists in an integer carrier for a dye to use, and a real support fault on the codes was measured passing the shipped tolerance gate on this repository's own near-zero input mode.
A kernel can ignore a quarter of its weights and leave every gate green.

The support gate closes that class: perturb one weight code through the contract quantizer, and every output cell whose fp64 reference moves by more than both runs' error envelopes plus two output ulps must change its bits.
The witness condition is an inequality, not a threshold, so no tolerance enters; judgement is one-directional, so formulation freedom under C2 and C3 cannot produce a false positive.

The design was refuted twice before being built, and both attacks changed it.

- The contract attack measured a correct kernel being refused when the builder's packing disagreed with the oracle's by one code position.
  The gate now derives every device input itself from the raw weights through the contract-anchored quantize-and-pack path, and its API has no parameter through which pre-packed device bytes can arrive.
- The same attack proved the scales and biases bindings unscreenable through the raw-weight oracle: no single-binding perturbation of them exists.
  They are named as unscreened in the gate's policy string rather than left as a silent gap; the two-binding power-of-two group rescale is recorded here as the future extension.
- The coverage attack planted two one-line support faults in the shipped kernel, a dead guard on the two-word read and a wrong lane stride, and both passed the first design at 49x to 443x margin because its probe positions came from the operator's index space, which provably never reaches a word-straddling code.
  Probe positions now come from the binding's address arithmetic: every alignment residue a code can have inside a word is covered completely, and a lane-stride band of block indices is covered so a wrong stride in (32, 64] is caught at its first skipped block; beyond that band block positions are sampled, and the policy says which is which.
- One further change came from re-deriving the dead-guard case: a straddling code's perturbation flips exactly the bits living in its second word, leaving the first word identical, so a broken two-word read sees no change at all and the fault is caught for every original value rather than only when the low bits happen to differ.

Both planted faults are acceptance tests now and both are refused; the correct kernel screens clean at every judged target.
Nondeterminism abstains rather than refusing, per amendment 1, and abstentions count as no coverage.

## Amendment 3, 2026-08-19: where the noise floor lives, and what detects a clock that moved

Sections 8.3 and 9 were written before the harness existed and they turn out to say two things that cannot both be implemented literally.
The harness was reviewed against them by six independent reviewers, each finding adversarially refuted; two of the surviving findings are about this document rather than about the code, so they are settled here before the code moves.

### The noise floor applies to a comparison, not to an outcome

Section 8.3 states the floor rule for "a comparison": the floor is the larger of the two arms' spreads, and a delta equal to its floor does not clear it.
Section 9 gives O1 exactly two conjuncts, the interval's lower endpoint above 1.10 and the peak footprint no worse, and gives O3 "the same comparison and same floor rule" as O1.

Implemented literally these disagree, and the harness reproduced the disagreement: O1 computed whether the delta cleared its floor and then dropped the answer, while O3 used it to override the interval verdict.
The same samples therefore read GO at one cell and REFUSED at another, which is not two readings of one rule.

The ruling is that the floor belongs to the comparison, in one place, and every outcome inherits it.
A comparison may claim WIN or LOSS only if its interval clears 1.0 on one side AND its delta clears its own noise floor; otherwise it is REFUSED, which means undecided and never a direction.
O1 then keeps the two conjuncts section 9 gave it, reading a verdict that already respects the floor, and O3 reports that verdict directly.
This is the conservative direction: it can only make a GO harder to obtain, never easier, so it cannot manufacture a claim that the original text would have refused.

### A reference arm whose spread says the clock moved rejects the round

Section 8.3 named the noise floor as this measurement's whole defence against timing noise.
That is not sufficient, and AGENTS.md already says why in general terms: equalising a clock excursion across arms is not detecting one, and the reference arm's own spread is the detector.

The gap is concrete rather than theoretical.
With ours at [100, 101, 102, 101, 100] seconds and stock at [130, 190, 200, 195, 205], stock's delta of 93.1% clears a floor of 38.5%, the interval's lower endpoint is 1.27, and the run reads GO, while the uncontaminated truth is about 1.02, which is NO-GO.
The floor rule does not catch this because a contaminated reference arm widens the floor more slowly than it moves the ratio.

So a comparison whose REFERENCE arm's spread exceeds the class limit is REJECTED, and a rejected comparison makes its cell non-binding rather than contributing a verdict.
The reference arm is the second of the two, which is stock for the two comparisons against stock and control for ours-against-control.

The limit is 1.5x max-to-min, inherited from the decode lane's kernel arms and NOT calibrated for full-job wall time.
That is stated plainly because it is a weakness: AGENTS.md says steadier quantities warrant a tighter limit, and a five-round wall time on an idle machine is a far steadier quantity than a microsecond dispatch, so 1.5x is a loose upper bound that catches gross excursions and will miss small ones.
It is not tightened here because nothing has yet measured what round-to-round variation is normal for this quantity; the Day 1 profile measures it, and tightening afterwards is a narrowing of what may be claimed and needs no further authority.

Two limits of this rule are recorded so nobody reads more from a passing canary than it says.
It cannot see a contamination that lands evenly on every reference round, because such a contamination leaves the spread small while still moving the ratio; the opening and closing idle gates remain the only defence there.
And it is a per-comparison rule, so a cell may be rejected on one comparison and readable on another; a cell is non-binding if any of its comparisons is rejected, because the arms it is reading came from the same rounds.

## Amendment 4, 2026-08-20: the registered instrument cannot run on the registered target, so the profile turns compile off and measures what that costs

### What section 3.3 asked for, and why it cannot happen

Section 3.3 registered the instrument as "MLX's own evaluation boundaries around named regions of the step".
mlx-lm 0.31.3 wraps the whole training step in `mx.compile`, at `mlx_lm/tuner/trainer.py:248`, and exposes no flag to turn it off: the decorator is unconditional and `mlx_lm.lora` never reaches it.
MLX refuses an evaluation inside any such transformation, in as many words:

    [eval] Attempting to eval an array during function transformations
    like compile or vmap is not allowed.

Reproduced 2026-08-20 against mlx 0.32.0 and mlx-lm 0.31.3.
So the instrument and the target as registered cannot both be had, and no code was written against section 3.3 as it stood.

### The ruling

The profile runs with compile disabled, through `mx.disable_compile()`, and the named regions are timed by evaluation boundaries inside the uncompiled step.

Two facts about the switch were measured before it was adopted, because the whole ruling rests on them.
It is read at CALL time, not at decoration time: a function already decorated with `mx.compile` runs uncompiled while the switch is off, and refuses an interior `eval` again the moment it is re-enabled.
That means mlx-lm's own `step`, which is decorated inside `train()` and is not ours to redefine, becomes uncompiled without touching mlx-lm at all, and the same process can run the step both ways.

### What this costs, and the guard that prices it

An uncompiled step is not the shipped step.
`mx.compile` fuses elementwise chains, so the composition of the step's time can move even when its total does not, and a share f taken from the uncompiled step is a claim about the compiled one only to the extent the two agree.

The profile therefore measures the step total BOTH ways, in the same run, on the same batch, and reports the ratio `uncompiled_total / compiled_total` beside every share it publishes.
The ratio is not a correction and nothing is scaled by it.
It is the stated bound on the transfer: a ratio near 1 says the two steps cost the same in total and the shares are worth reading; a ratio far from 1 says the decomposition describes a workload the product does not run, and the profile says so rather than reporting shares that look like the shipped step's.

A ratio outside [0.90, 1.10] REJECTS the profile.
The number is declared here, before the run, and it is declared uncalibrated: it is a judgement about how much redistribution makes a share meaningless, not a measured limit, and the first run's observed ratio is reported whatever it is so a later amendment can replace the guess with evidence.

### What section 3.3's reconciliation check now means

The 2% reconciliation stands and now has something to bite on.
The named regions plus one explicit unattributed remainder are measured inside the uncompiled step, and their sum is compared against that same uncompiled step's own end-to-end time, taken without the interior boundaries.
A gap past 2% means the boundaries themselves moved the workload they were inserted to describe, and the profile is rejected.
This is a different check from the compile ratio above and both must pass: one prices the instrument, the other prices turning compile off.

### What this amendment does not change

The cells, the shapes, the selection rule, the gain formula, the credited ratios and their named assumptions, the kill rule and the run discipline are all untouched.
The candidate ratios r are floors measured on their own, not shares, so they do not pass through the uncompiled step and this amendment does not reach them.
