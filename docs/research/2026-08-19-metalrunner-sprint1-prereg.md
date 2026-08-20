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

---

## Amendment 5, 2026-08-20: the registered instrument measures its own fences, so a share becomes a fitted slope

### What section 3.3 asked for, and what it measured instead

Section 3.3 defines an operation's share as its wall time divided by the step's, both taken from a run "whose only instrumentation is those boundaries".
Amendment 4 established that those boundaries can only be evaluation points inside an uncompiled step, and the profile harness was built that way.

The boundaries are not free, and their cost lands asymmetrically.
A boundary forces every pending computation to finish and THEN reads the clock, so its cost sits inside the span that forms a share's numerator and outside the plain step that forms its denominator.
Every share is therefore an overestimate, and the size of the overestimate grows with how many boundaries the region carries.

Measured on the 0.6B proxy, five rounds, one fixed batch, 2026-08-20:

| Candidate | Boundaries per step | Share by section 3.3 | Share with no instrument in the step |
|---|---|---|---|
| L, output head and cross-entropy | 2 | 0.263 | 0.245 |
| A, attention core | 32 | 0.193 | 0.045 |
| Q, quantized projections | 197 | 1.401 | 0.376 |

A share of 1.401 is not a fraction of anything, and it is the only reason the fault was noticed at all.
The three unmarked figures are disjoint and sum to 0.666, leaving a third of the step for the adapters, the norms, the elementwise work and the optimizer update.
The same three by section 3.3 sum to 1.857.

Two corrections were attempted and both failed.
Subtracting the whole measured excess gave negative shares for two of the three candidates, which proves the excess is not all inside the spans.
Doubling each boundary's fences measured nothing, because a second evaluation of an already-materialised tensor is free.

The registered method would therefore have selected whichever operation has the most call sites, independently of where the time goes, and candidate Q has 197 against candidate L's 2.

Every figure in that table was measured with compilation disabled, which clause 4 reverses.
Clause 4 says what happens to them.

### The dial

**Clause 1.** A share is no longer measured by placing boundaries around an operation.
It is measured by shrinking the operation's work on a dial while holding fixed the number of calls through its seam, the output shapes, and the structure of the graph, and timing the whole step at each setting.
No timing device of any kind is placed inside the step.

`phi` is dimensionless: the ratio of the dialled size to the full size, so `phi = 1` is stock's own size and `phi = 0.5` is half of it.
The registered ladder is `phi` in `{1.00, 0.75, 0.50, 0.25}` for every knob, and where a dial cannot take a value on that ladder the nearest realisable value is used and the ACTUAL `phi` is what enters the fit.
Five rounds, exactly three warm-up passes per arm before any timed round, every arm interleaved within every round.
Exactly three, not at least three, because a count left free is a knob an implementer can turn: at ONE warm-up a probe of this construction returned mutually contradictory answers on consecutive runs, because the first timed round of whichever arm ran first still carried allocation and graph-cache cost, and three was where that stopped.

The batch is the FIRST batch mlx-lm's own iterator yields at the registered seed for that cell and band, and its measured supervised fraction is recorded in the profile.
Which batch is not a free choice: the supervised fraction sets candidate L's floor, so a different batch is a different measurement.

The fit is `T(phi) = a + b*phi`, least squares, and it is taken twice for two different purposes.
Within each round, across that round's four arm times, giving one slope per round: that distribution is what clause 9 credits from.
Pooled over the arm medians, giving one slope, one intercept and one coefficient of determination: that fit is what the gates of clause 6 read.
The registered slope `b` reported beside any share is the median of the per-round slopes.

### The attributed cost, and why the gain formula still holds

Section 4.1's formula is `gain = 1/(1 - f*(1 - 1/r))`, which rearranges to `T_new = T - f*T + f*T/r`.
So the formula is exactly right whenever

    f := A / T        A is what the candidate's operation costs the step
    r := A / F        F is what the replacement would cost the step, measured on the same basis

because then `T_new = T - A + F`, which is what replacing an operation does.
Everything below is the definition of `A` and `F`, and nothing below changes section 4.1.

**Clause 18.** `A` is not the same quantity for every candidate, because the candidates are not the same kind of change.
A retune keeps the dispatch and the host-side construction of its own graph; a rewrite deletes them along with whatever intermediate tensors it stops building.
Which candidate is which is read off the names section 4.2 already gives them, and is not chosen afterwards:

| Candidate | Section 4.2 calls it | Kind | `A` | `F` |
|---|---|---|---|---|
| Q | "quantized matmul retune at training widths" | retune | `b` | `d` |
| A | "tiled attention forward and backward" | rewrite | `b + c` | `d + c_floor` |
| L | "streamed mask-aware output head and cross-entropy" | rewrite | `b + c` | `d + c_floor` |

`b` is stock's fitted slope and `d` is the floor's fitted slope, both from the same dial.
`c` is stock's residue: the pooled fitted intercept minus the same step with the operation ablated, measured in the same run and the same regime.
`c_floor` is the floor's residue, measured by ablating the floor implementation exactly as stock is ablated.

Pairing `A = b + c` with `r = b/d`, as an earlier draft of this amendment did, is wrong: it assumes without saying so that `c_floor = c*d/b`.
On `T = 100`, `b = 20`, `c = 10`, `d = 5` that assumption returns 77.5 where eliminating the residue returns 75 and preserving it returns 85.
The table above removes the assumption by measuring `c_floor`.

Two registered conventions on the residue, both fixed here:
a residue whose magnitude is below the measured resolution floor `R` is recorded as zero, for stock and for a floor alike;
and a residue that measures negative by more than `R` REJECTS that candidate's reading rather than being clamped, because a negative residue means the ablation ran slower than the intercept predicts and the construction is not measuring what it claims.

**Clause 19.** `f` is measured inside the real training step, so `F` must be too, or the ratio is a claim about a bench and the share is a claim about a step.
Candidate Q's floor, the dense fp16 matmul, and candidate A's floor, MLX's fused attention, are both real callables and are installed at the same seams the knobs use and dialled inside the same step.
Candidate L's floor is a streamed kernel that does not exist, so it stays on the bench under this written exception: its `d` is measured in isolation, its `c_floor` is registered as zero because eliminating the full logits tensor is precisely what that candidate is, and both facts are reported wherever candidate L's gain appears.
Candidates A and Q no longer carry either assumption.

**Clause 10.** Section 4.2 credits candidate A's floor as MLX's fused forward multiplied by 3, "to stand for forward plus backward", and MLX has no fused backward at all.
Installing MLX's fused attention in the step buys a fused FORWARD and leaves stock's own composed backward in place, so an in-step measurement already contains a backward.
Adding a modelled backward on top of that would count the backward twice, and subtracting the one that is there would need directional attribution, which is exactly the thing this amendment abandoned.

So candidate A's floor drops the modelling entirely.
`F` for candidate A is the whole attention region's cost with MLX's fused implementation installed, both directions, measured in the step by the same dial as `A`, with no multiplier of any kind.
The registered "times 3" is void, the "twice the forward" relation is void, and neither is replaced.

What this costs is stated rather than hidden: a real tiled implementation would fuse the backward too, and this floor does not, so candidate A's credited ratio is an UNDERSTATEMENT of what that candidate could reach, and it is labelled as one wherever it appears.
Understating a candidate is the safe direction for a rule that picks what to build, and it is the only version of this floor that is entirely measured.

The call counts remain registered because candidate A's own directional reduction needs them, by the same ratio-of-sums construction clause 8 registers for the kill rule; clause 8 governs the kill verdict and this clause borrows only its method.
They are not symmetric: measured 2026-08-20, attention fires once per layer forward and once per ADAPTED layer backward, which on the pinned arrangement is 36 forwards and 16 backwards.

**Clause 9.** `ratio_lo` keeps section 4.2's construction, the smallest numerator over the largest denominator, taken over the per-round distributions: the smallest per-round `A` over the largest per-round `F`.
Both distributions come from the same dial at the same settings in the same rounds, except where clause 15 registers a different dial for a floor, and in that case the ratio is refused rather than taken across two dials.

### Every statistic, baseline and reduction, fixed here

A rule that names a quantity without naming how it is reduced leaves the answer to whoever writes the code.
This section closes each one so that two implementers reading this document compute the same number.

| Quantity | Fixed as |
|---|---|
| `T`, the step total | stock with NO seam installed at all, median over the eligible rounds; never the fitted `T(1)` |
| Warm-ups | exactly three per arm, not a minimum |
| `b`, the reported slope | the median of the per-round slopes |
| `a`, the intercept | from the pooled fit over arm medians |
| `c`, the residue | the pooled intercept minus the median of the ablated arm's rounds, on the same `T` baseline |
| `A` entering `f` | the median reduction, so `f` is a median-over-medians |
| `A` and `F` entering `r` | the worst pairing the samples permit, per section 4.2: smallest per-round `A` over largest per-round `F` |
| The mix of those two | DELIBERATE and declared: a median share against a worst-case ratio gives the smaller gain, which is the conservative direction for a rule that decides what to build. The credited `T_new` is therefore a bound and not a predicted measurement, and it is reported as a bound |
| Per-round residues | not taken; the residue is pooled, so `A`'s per-round distribution varies only through `b` |
| The scaffold-only arm | applies the dial's index arithmetic and any padding, then runs the operation at FULL size, so its cost is the scaffold and nothing else |
| Scaffold price sign | absolute value, in clause 15's comparison and clause 21's limit alike |
| Slope in a denominator | absolute value |
| A non-positive slope | rejects that candidate's reading explicitly, rather than being left to fail a numeric gate by accident |
| `R` | measured per width and per cell, never once and reused across either |
| A missing ratio | excludes that candidate from selection and does not refuse the whole selection; the ruling records which candidate was excluded and why |
| Candidate L's floor halves | its forward and backward costs are combined within each round, and the round distribution is reduced afterwards |
| A partition region's share, clause 22 | `b/T`, with no residue term, because a partition region is a region and not a candidate |
| The validation stage's own memory ceiling | the ceiling the committed end-to-end harness declares for this model, read from its run plan at arm time and recorded in the run record, since the validation stage cannot budget from a measurement it has not taken |

### The knobs

One dial per candidate, each installed at the seam that owns the operation.

| Candidate | Seam or seams | Dial |
|---|---|---|
| L | `QuantizedEmbedding.as_linear` and `nn.losses.cross_entropy` | vocabulary rows kept |
| A | `mlx_lm.models.qwen3.scaled_dot_product_attention` | registered by clause 15 |
| Q | `nn.QuantizedLinear.__call__` and `QuantizedEmbedding.as_linear` | input width of the activation and the weight |
| P3, the partition region of clause 22 | `nn.QuantizedLinear.__call__` alone | input width, head excluded |

Four properties every knob holds, each proven by a test rather than asserted:
the output shape does not move; the number of Python calls through the seam does not move; the dial is applied at every setting including `phi = 1`, so its scaffold enters every arm; and the arm at `phi = 1` is bit-identical to stock in both the loss and the full gradient tree, compared leaf by leaf with `mx.array_equal`.

The second of those counts Python calls, not Metal dispatches.
It proves the operation was invoked the same number of times, which is what a deletion would break; it does not prove no kernel was added, and it is not written as though it does.

**Clause 15.** Attention admits no single obvious dial, so three are built at the 0.6B proxy and one registered criterion selects among them.
The three, in registered completeness order, most complete first:

| Rank | Dial | What moves with it |
|---|---|---|
| 1 | key and value sequence length, with a hand-built mask at every arm | both matmuls and the softmax between them |
| 2 | head dimension on queries, keys and values, output padded back | both matmuls |
| 3 | head dimension on queries and keys only | the first matmul |

The price statistic is the absolute value of the fitted scaffold slope expressed as a fraction of the absolute value of that dial's own knob slope, and the smallest wins.
That statistic is dimensionless, so its tie threshold has to be dimensionless too: two prices within 0.01 of each other, one percentage point of the knob slope, are a tie, and a tie goes to the higher rank in the table above.
Comparing it against the time-valued `R` would be a units error and would let a different normalisation pick a different dial.
A dial that cannot place at least three distinct realisable settings on the registered ladder is not eligible, whatever its price.

Where MLX's fused attention is available at too few settings for the FLOOR to place three, the floor takes the highest-ranked dial that can, the share keeps its own dial, and clause 9 refuses the ratio rather than taking it across two dials; the profile then reports candidate A's share with no credited ratio and section 4.3 treats it as it treats any candidate whose ratio is missing.

### The instrument's own price

**Clause 5.** The scaffold is priced in two numbers, not one, because a scaffold whose own cost changes with the dial enters the slope where a single check cannot see it.
The scaffold OFFSET is the `phi = 1` arm minus stock with no seam installed at all.
The scaffold SLOPE is the fitted slope of a scaffold-only arm, which applies the dial's machinery and discards its effect, fitted over the same ladder.

**Clause 6.** A linearity gate refuses a candidate whose dial is not a dial.

**Clause 21.** The limits are these, and three of the four are pure ratios that need no measured scale and are therefore fixed here outright:

| Limit | Registered value | Needs a measured scale |
|---|---|---|
| Scaffold offset, absolute value | at most `3R` | yes, `R` |
| Scaffold slope, absolute value | at most one tenth of that knob's own slope | no |
| Coefficient of determination on the pooled fit | at least 0.99 | no |
| Largest absolute residual on the pooled fit | at most one twentieth of the absolute slope | no |

`R` is the resolution floor: the median absolute difference between two IDENTICAL arms over the registered rounds at that cell.
It is measured by an instrument-only stage that runs identical arms and scaffold-only arms, dials no candidate, and computes no share, and its value lands in an addendum committed before any knob is dialled.
The four multipliers are judgements, declared as judgements, registered here before the measurement they multiply, in the same spirit as Amendment 4's band.

A slope of zero or of negative sign fails the linearity gate by construction, because the residual limit is a fraction of the absolute slope and no residual can be at most zero.
That is the intended behaviour: a dial that does not move the step is not a dial.

**Clause 16.** Section 4.3's two-percentage-point tie band STANDS and is not replaced.
What this clause adds beside it is a second reason two candidates may be inseparable: the machine cannot resolve the difference.
A pair is TIED if it falls inside section 4.3's two-point band, OR if it is unresolvable by the test below; both routes lead to the same place, which is section 4.3's registered tie-breaks.

A gain is dimensionless and the harness's resolution is a time, so the test is made in time.
Two candidates with scores `g1` and `g2` predict step times `T/g1` and `T/g2`, so the difference to resolve is

    delta(width) = T(width) * | 1/g1(width) - 1/g2(width) |

A single constant cannot express this: at `T = 100 ms`, two points of gain is 1.96 ms near a gain of 1.00 and 1.17 ms near 1.30, and 1.30 is the region the sprint aims at.

Two widths and three candidates make this ambiguous unless the quantifiers are fixed, so they are fixed here.
The test is evaluated at EACH width with that width's own `T` and its own `R`, and a pair is RESOLVABLE only if `delta(width) >= 2 * R(width)` at BOTH widths.
Anything less is unresolvable, which is the conservative direction: a pair the machine can separate at only one width is not a pair the machine can separate.

Three candidates can be pairwise resolvable in a way that admits no consistent ordering, so the band is anchored rather than transitive, and two things about the anchor have to be nailed down or the band is not a function of the measurements.

The ANCHOR is the highest-scoring candidate.
Where two or more candidates share the highest score exactly, the anchor is the earliest of them in section 4.2's table, which is the same last-resort ordering section 4.3 already registers.
Without that, two candidates on an identical score give two different anchors and two different bands: at `T = 100 ms` and `R = 1.2 ms` both widths, with candidate scores `A = (1.20, 2.00)`, `B = (2.00, 1.20)` and `C = (1.17, 1.50)`, A and B both score 1.20 and anchoring on A admits C while anchoring on B excludes it.

The BAND is the anchor plus every candidate that satisfies EITHER tie route against the anchor: within section 4.3's two percentage points of the anchor's score, OR not resolvably worse than the anchor by the test above.
Both routes are unions, not intersections.
An earlier draft defined the band by the second route alone while the sentence above it offered both, which left a candidate inside the two-point band but resolvably worse simultaneously in and out of it.

That is one pass over the anchor and it yields exactly one band, whatever the pairwise relations among the others are.

Measured uncompiled, two identical arms sat 0.155 ms apart; compiled the step is faster and nothing has measured whether the margin survives, which is what the instrument-only stage settles.

### What replaces section 3.3's reconciliation

**Clause 2.** The requirement that named regions plus a remainder equal the step within 2% is void.
There are no spans to sum and the check it was written for cannot be posed.

**Clause 22.** What replaces it is weaker than a reconciliation and is named as what it is: a DOUBLE-COUNTING TEST over a disjoint partition.

| Partition region | Measured by |
|---|---|
| P1 | the vocabulary dial, head matmul and cross-entropy |
| P2 | the attention dial |
| P3 | the projections dial with the head excluded, which is its own registered knob |
| P4 | one minus the other three, by subtraction |

P1 plus P2 plus P3 must be at most one, and P4 is reported as the remainder.
A sum above one REJECTS the profile, which is the check the boundary instrument could never have passed.

Two limitations are stated rather than implied.
It detects double-counting and cannot detect omission, because P4 absorbs anything unmeasured by construction.
And three separately fitted slopes are not guaranteed additive under compilation, so the test is a necessary condition on the decomposition and not a proof of it.

The candidates' own region sets are unchanged from section 3.3 and DO overlap, because the tied output head belongs to both L and Q.
That is why the partition is measured separately, with its own fourth knob, and why clause 13 forbids composing two candidates' shares.

### The marks are retained, and demoted

**Clause 3.** The boundary instrument stays in the tree and stops being a timing instrument.
Its call counts are exact and load-bearing: the asymmetry it verifies, that attention fires once per layer forward and once per ADAPTED layer backward while the projections fire `7 * adapted - 3` times backward, has already caught a seam that never fired and a test that left gradient checkpointing installed for an entire session.
The profile runs one structural pass per cell AND per width, because a fault that only appears at one width would otherwise pass, and it takes no time from that pass.
Gradient identity is compared leaf by leaf against the plain pass with `mx.array_equal`, and the loss with the same.

### Compilation

**Clause 4.** Amendment 4 turned compilation off because MLX refuses an evaluation inside a compiled step and the registered instrument needed one.
A dial needs none.
The profile therefore runs COMPILED, which is the configuration `mlx_lm.lora` actually runs, and Amendment 4's `uncompiled_total / compiled_total` ratio and its [0.90, 1.10] band no longer gate the profile.
Amendment 4 is not withdrawn: it remains the record of why the boundary instrument could not be had on the registered target.

Every figure quoted in this amendment was measured uncompiled and is re-established compiled before it is relied on.
The threshold for replacing one has to match its units, because the figures quoted here are times, shares and counts and `R` is a time:

| Kind of figure | Replaced when the compiled value differs by more than |
|---|---|
| a time | `R` |
| a share or any other fraction of the step | `R / T` at the cell it was measured on |
| a ratio | `R / F`, the floor cost it divides by |
| a count | any difference at all, because a count that moves is a structural fault and not noise |

A figure inside its threshold stands as written.
No figure is revised for any other reason, and none is revised after a candidate share exists.

### Trace discipline

MLX keys its compilation cache on the underlying callable and the input signature, not on the wrapper `mx.compile` returns.
A design that swaps a seam between arms can therefore serve one arm's traced graph to another, and a dial that silently does nothing produces a clean fit through a horizontal line.

**Clause 25.** Three requirements, and a guard that covers what they miss:

| Requirement | The fault it prevents |
|---|---|
| Every arm at every width owns a freshly built closure, retained for the whole run | two arms sharing one traced graph |
| The optimizer's state is initialised and evaluated before any arm is built | the first update growing the captured state and forcing a second trace, which can land after the seam is gone |
| A trace counter refuses the run if any trace occurs during the timed rounds | every remaining version of the same fault, including a width change retracing an arm whose seam has been removed |

The third is the general guard: a trace during timing is always a fault, whatever caused it.

### The corpus and the two widths

Section 3.2 fixes the sequence length at 2048.
mlx-lm pads each batch only to one plus the next multiple of 32 above its own longest row, so a step's width is a property of the data and `max_seq_length` is a cap the registered corpus never approaches.
Measured 2026-08-20, databricks-dolly-15k's median row is 116 tokens under the pinned tokenizer and its widest band that can fill 1024 training rows is 160.

Width is not a detail here.
The unmarked ablation puts attention at 0.042 of the step at width 97 and 0.161 at width 769, a four-fold move across a range narrower than the one in dispute, and the mechanism is geometric: attention's score matrix is quadratic in the sequence length while the head, the projections and the loss are all linear in it.
Choosing one width would therefore choose which candidate the rule selects.

**Clause 14.** Two widths are registered and both are measured, and the selection takes each candidate's gain at its WORSE of the two.
The two widths apply to the deciding cell; cells A, C and D decide nothing under section 4.3 and are measured at the short width only, which the profile records rather than leaving to be inferred.

Section 3.2's "Fixed: sequence length 2048" is superseded for the profile, and the fixed quantity becomes the two band edges of clause 20; everything else in that sentence stands.
Section 3.2's token counts M of 2048 and 8192 are superseded with it, and M becomes `batch * (width - 1)` at each registered band, which is what `default_loss` actually trains on because it takes `batch[:, :-1]`.
Section 10 says this document claims "nothing about sequence lengths other than 2048 ... unless a cell is added here in writing first", and this clause is that writing: the profile's claims are scoped to the two registered widths and to nothing between or beyond them.
Section 8's end-to-end harness still registers 2048, this amendment does not reach it, and the profile and the shipping measurement are therefore scoped to different widths until a later amendment closes that gap.

**Clause 20.** Both widths come from ONE corpus, `HuggingFaceH4/ultrachat_200k`, at two length bands.
Two corpora would have differed in their supervised fraction, and the supervised fraction is not a detail either: candidate L's floor is defined as the same work on only the supervised rows, so a second corpus would have changed that candidate's ceiling for reasons that are not width.

The bands are not chosen, they are DERIVED, so that registering the rule registers the answer:
the short band is the shortest 32-token band that can fill 1024 training rows from `train_sft` AND 128 validation rows from `test_sft`;
the long band is the shortest such band whose lower edge is at or above 1024 tokens;
both measured under the pinned tokenizer, selection seeded by the constant already in the pinner.
The two splits are shuffled independently, each with that same seed, because they are separate upstream populations and a single pooled shuffle across them would not be reproducible from either alone.
The rest of the corpus specification, fixed here rather than at pin time:

| Field | Registered value |
|---|---|
| Dataset | `HuggingFaceH4/ultrachat_200k`, default configuration |
| Split | `train_sft` for the training rows, `test_sft` for the validation rows |
| Revision | pinned by the upstream commit hash, resolved once at pin time and committed beside the slices; a different hash invalidates the slice and the profile refuses rather than re-pinning silently |
| Chat adapter | the first user turn becomes the prompt and the first assistant turn becomes the completion, single turn, discarding the rest of the thread, so both bands carry the same mask shape the pinned Dolly slice carried |
| Row selection | the seeded deterministic selection already in the pinner, seed 20260820: shuffle the band's eligible rows by that seed and take the first 1024 and 128 |
| Recorded beside the slices | both derived band edges, the row counts, the measured supervised fraction of each band, and the tokenizer hash |

If either band cannot be filled, no slice is written and the profile does not run, rather than the rule being relaxed.

The pinned Dolly slice is retired from the profile and left in the tree, because section 8's harness still references it.

### The selection rule

**Clause 14, continued.** For each candidate the gain is computed at both widths and the candidate's SCORE is the smaller of the two.
The largest score selects.

**Clause 24.** Section 4.3's shipping floor applies to the SELECTED candidate and the order of operations is registered here because two readings of that sentence give different answers.
The registered order is: discard every candidate whose score is below 1.10 FIRST, then apply section 4.3's tie band and tie-breaks to whatever remains.
If nothing remains, section 4.3's existing branch applies unchanged: nothing is selected, the reading is recorded, and the top two by score are kept in score order.

As implemented today the floor is tested against the leader and the winner is then chosen from inside the tie band, so a candidate below 1.10 can be selected by the footprint tie-break.
Reproduced 2026-08-20 by execution, not by reading: a leader at 1.1012 and a tie-band member at 1.0812 returns the member as SELECTED.

### The kill rule

**Clause 7.** Section 3.2 lists five projection shapes and the model contains six.
The key and value projections at `(1024, 2560)` are counted in candidate Q's share and omitted from the registered list; S6 is added at that shape.

**Clause 23.** Section 5 removes candidate Q if stock is within 1.10 of the dense ceiling at 4 of the 5 shapes, which is "all but one shape".
Over six shapes that reading is 5 of 6, registered as "all but one" rather than as a fraction, because inheriting the number 4 would have loosened the rule from four-fifths to two-thirds without saying so.

**Clause 8.** Two reductions section 5 never names, both fixed here before any ratio exists.

A shape's forward and backward reduce to one scalar as a ratio of SUMMED COSTS, not as an average of ratios:

    shape ratio = sum over directions of (calls * stock cost per call)
                / sum over directions of (calls * floor cost per call)

with the call counts supplied by the structural pass of clause 3.
An average of per-direction ratios is a different quantity and it is not the one the kill rule asks for.
On equal counts with stock costs 100 and 20 against floor costs 100 and 10, the average of ratios is 1.50 and the ratio of sums is 1.09, so the two land on opposite sides of the 1.10 threshold and only the second answers "is stock close to the ceiling in the time it actually spends".

The sums are taken WITHIN each round, giving one shape ratio per round, and the ratio the kill rule reads is the LARGEST of them.
The largest is the one furthest from the ceiling and therefore the least likely to kill, which is the conservative direction for a rule whose effect is to remove a candidate from contention.
That is deliberately the opposite reduction from `ratio_lo`, which takes the worst pairing because its effect is to CREDIT a candidate; both pick the reduction that makes their own action harder to take, and neither is left to an implementer.

The two widths reduce by requiring a shape to be within 1.10 at BOTH widths before it counts toward the kill, so headroom at either width keeps the candidate alive.

"Orientations" wherever it appears in this amendment means those same two directions, forward and backward, and nothing else.

### The footprint tie-break

**Clause 12.** Section 4.3's first tie-break asks for the measured peak-footprint delta of implementations that do not exist yet.
Its source is the floor arms' own measured peaks against a stock baseline, which is a statement about the floor and not about a kernel, and it is labelled as such wherever it is reported.
The stock baseline is the peak of the arm the floor replaced, measured in the same place: in the step for candidates A and Q, on the bench for candidate L, which makes candidate L's delta not comparable with the other two and is recorded as a further reason the tie-break may fall through.
It reduces by taking the largest delta over shapes, directions and widths, so the tie-break reads the worst case rather than an average.
A positive delta means the floor arm's peak exceeded the baseline, and the tie-break prefers the smaller delta.
Where any candidate still in contention has no measured delta, or where candidate L is in contention against another candidate at all, the whole band falls through to table order and the ruling records that it did.

### The overlap between two candidates

**Clause 13.** The tied output head is a quantized matmul reached through `QuantizedEmbedding.as_linear`, and it belongs to candidate L's region set and candidate Q's alike.
Candidate Q's dial therefore moves the head along with the projections, because section 3.3 puts it there.
Two candidates' shares are never added, never composed, and never presented as a combined gain.

### The memory budget

**Clause 17.** The profile's budget was to be declared from the peak of the instrumented pass, which existed because fencing at the head forced every logit resident at once.
There is no such pass now.
The declared budget for the binding run is the largest peak measured across all arms and both widths in the validation run, multiplied by 1.25 and rounded up to the next whole gigabyte.
The binding run refuses to start if that budget exceeds the window granted, rather than being killed part-way through it.
The validation run itself cannot budget from a measurement it has not taken, so it declares the budget section 8's end-to-end harness already uses at this model, which is a larger number and is recorded as a placeholder rather than as evidence.

### Ordering

**Clause 11.** The VALIDATION RUN is the run of steps 10 and 12 of the increment plan: an instrument-only stage and a knob stage, both at the pinned target, both detached under section 3.4's discipline.
It PASSES when every knob's pooled fit clears clause 21's four limits at both widths, no trace occurs during any timed round, and no arm's rounds are refused by the spread gate.
The binding profile run is blocked on that pass, not on approval alone.

A reading is derived only from a COMPLETE matrix: three candidates by two widths, every entry present, every entry carrying the same context record of model, batch, width, band, supervised fraction and adapter count.
An incomplete or context-mismatched matrix refuses rather than reporting what it has, so a short-width share can never be paired with a long-width floor.
Candidate L's bench floor cannot share the in-step context by construction; its context record carries the bench's own parameters and is compared for the fields both have, which is the exception clause 19 already names.

### What this amendment does not change

The four cells and their supervision, the model, the LoRA rank and adapter count, the gain formula of section 4.1, the two-point tie band, the 1.10 shipping floor, the 1.10 kill ratio, the funnel, the held-out draw, the end-to-end measurement and its fairness conditions, and the run discipline of section 3.4 are all untouched.

Two of those need a word so that "untouched" is not read as more than it is.
The two-point tie band keeps its threshold and its meaning; clause 16 adds a SECOND route into the same tie-break, it does not widen or replace the first.
Section 3.4's requirement of a declared memory budget stands unchanged; clause 17 changes only where the declared number comes from.

Stated separately because omission is not the same as a ruling: the sequence length is NOT among them.
Section 3.2's 2048, its M values, and section 10's scope sentence all move under clause 14, and section 8's end-to-end harness does not.

### What this amendment does not yet contain

One measured number: the resolution floor `R`, from which the scaffold offset limit and clause 16's tie-band demand both take their scale.
It lands in an addendum committed alone, after the instrument-only stage and before any knob is dialled, and nothing else enters that addendum.
