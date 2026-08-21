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


---

## Amendment 6, 2026-08-20: the dial selection binds at the long width, and candidate A is judged by a ceiling rather than a ratio

Amendment 5 registered three attention dials, a criterion to pick among them, and a rule putting every floor inside the real step.
Building the three dials showed that the criterion admits two answers depending on a width it does not register, and that the floor rule cannot be applied to candidate A at all.
An earlier draft of this amendment answered the second gap with a bench exception for candidate A's floor.
That draft was reviewed twice by an independent model, blocked twice, and is REPLACED by this text rather than patched; clause 29 records what it got wrong and why, because the record of a rejected design is part of the pre-registration's evidence.

This amendment is written AFTER the measurements that exposed both gaps.
That is the weakest thing about it and it is stated first rather than buried.
What it costs is controlled in two ways: every ruling below is made on a mechanism that was registered before the numbers, and every clause that reverses committed text says so in the word "reverses" rather than in the word "extends".
The 0.6B numbers quoted here are cited as evidence that a gap exists, and none of them is an input to any rule.

### What clause 21's ordering actually binds

Clause 21 says the resolution floor `R` lands in an addendum committed "before any knob is dialled".
Read literally that forbids the proxy smokes this plan's own sequencing puts before the instrument-only stage, and those smokes have already run: the tables below were measured on 2026-08-20 at the 0.6B proxy with no addendum in existence.

That is an inconsistency inside Amendment 5 rather than a licence taken against it, and it is resolved in the direction the sequencing already assumed.
Clause 21's ordering binds the RECORDED profile and the two validation stages at the 4B target.
It does not bind smokes at the 0.6B proxy, which compute no credited share, enter no recording, and cannot be cited as an input to any rule.
Every 0.6B number in this document is one of those, and is labelled as such wherever it appears.

### What clause 15 asked for, and why it admits two answers

Clause 15 picks candidate A's dial by the smallest fitted scaffold slope expressed as a fraction of that dial's own knob slope.
It registers no width at which that statistic is evaluated, and the statistic is not stable across width.

Proxy smoke, 0.6B, batch 2, five rounds, three warm-ups, compiled, twenty-eight arms interleaved inside one set of rounds:

| Queries | kv-length | head-dim-qkv | head-dim-qk | Named by the criterion |
|---|---|---|---|---|
| 96 | 0.101 | 0.937 | 0.440 | kv-length, on price |
| 384 | 0.221 | 0.027 | 0.029 | head-dim-qkv, tied with head-dim-qk and broken by rank |

The dial is not a presentation choice, because it sets candidate A's share: at 384 queries the length dial reads 0.060 and both head-dimension dials read 0.023, against an unmarked ablation of 0.068 in the same rounds.

Two separate things are wrong and only one is about width.

The criterion rewards a scaffold that does not move, and a dial that moves less of attention has less scaffold to move, so cleanliness and completeness pull against each other and only the tie-break knows it.
That is a property of the registered statistic and this amendment does not change it.

And at both widths tried, every scaffold slope sits inside the machine's own jitter, so the ratio being compared has noise in its numerator.
Clause 6's gate cannot see that, because it tests the fit's shape and not its size: a perfectly straight line whose whole excursion is smaller than the machine resolves passes every limit clause 21 sets.
The mechanism behind the small excursion is geometric and was registered long before these numbers: attention's score matrix is quadratic in the sequence length and every other region of the step is linear, so attention is a few percent of a short step and roughly a fifth of a long one.

### Clause 26. The dial selection is made once, at the deciding cell's long width, after `R` exists, and only on readable prices

Clause 15's criterion is evaluated ONCE, at the 4B deciding cell, at the LONG width registered by clauses 14 and 20, inside validation stage two, and the dial it names is candidate A's dial at BOTH widths.
The share is refitted per width and each refit must pass clause 21's limits and clause 11's gates at its own width; the long-width fit is never reused as the short width's share.
One dial for both widths rather than one per width, because clause 14 selects by the worse of two gains, and a candidate measured by two different dials has two shares that are not comparable.

The earlier draft evaluated the criterion at the 0.6B proxy, and the review that blocked it was right that no `R` exists there: clause 21's statistics table registers `R` per cell and per width, and a 0.6B batch-2 proxy is not a registered cell.
Moving the selection to the deciding cell removes the gap instead of registering a new `R` context for it.
The order is therefore writable, and it is this one:

| # | Event | What it consumes |
|---|---|---|
| 1 | Clause 20 pins both corpus bands | nothing measured |
| 2 | The instrument-only stage runs at the 4B target, both widths | the pinned bands |
| 3 | The `R` addendum is committed alone | step 2's measurement, nothing else |
| 4 | Validation stage two fits all three attention dials at the deciding cell's long width, and this clause names the dial | the long width's own `R`, from the addendum |
| 5 | The named dial refits at the short width, and both refits face clause 21 and clause 11 | the short width's own `R` |
| 6 | The binding profile, then the decision | everything above |

No step consumes a number produced after it, and `R` never depends on any dial, so the circularity the review asked about does not exist in this ordering.
A selection taken before clause 20 pins both bands is NON-BINDING and is recorded as such, because the long width is a property of a band that clause 20 pins and not a number a harness may pass in.

A price is READABLE only when both of its terms are measurements rather than noise, and clause 6 does not establish that.
Two conditions are added here:

| Condition | Rule |
|---|---|
| The knob's excursion | `abs(slope) * span >= 10 * R(w)`, for the pooled slope AND for the median of the per-round slopes, both; a dial failing either is not eligible at that width, whatever its price |
| The scaffold's readability | three-way, below |

`span` is the distance between the highest and lowest ACTUAL realisable `phi` values of clause 1's ladder, the same values that enter the fit.

**The scaffold is read in three cases and not two, because a fitted slope is a trend and not a movement.**
This AMENDS clause 5, whose text defines the scaffold slope as "the fitted slope of a scaffold-only arm" with no other case, and the ledger row for clause 5 records it.
An earlier version of this clause clamped a scaffold to zero when its fitted excursion was below `R` and recorded the fitted slope otherwise, and that leaves a scaffold that moves without fitting recorded as still.
Reproduced 2026-08-20: scaffold times of 100, 110, 110, 100 across the ladder fit a slope of exactly zero with zero offset while the arm actually moved by 10, so its price is zero under either of those two cases and no committed limit objects.

| Case | Rule |
|---|---|
| Observed range below `R(w)` | the scaffold did not move by as much as the machine resolves, so its slope is recorded as ZERO |
| Observed range at or above `R(w)` AND `abs(slope) * span >= R(w)` AND the scaffold's own pooled fit clears clause 6's two SHAPE limits, the coefficient-of-determination floor and the residual limit | the scaffold moved and the fitted line describes that movement, so the fitted slope stands |
| Observed range at or above `R(w)` and NEITHER of the other two conditions holds | the scaffold moved by more than the machine resolves and the line does NOT describe it, so its price cannot be read and the dial is NOT ELIGIBLE at that width |

The second case demands clause 6's two SHAPE limits on the SCAFFOLD's own fit, not only a resolvable excursion, because a resolvable excursion is not evidence that the line describes anything.
It demands the shape limits and NOT the whole of clause 6, deliberately: clause 6's gate also rejects a slope of negative sign, and this amendment has already registered that a scaffold slope may legitimately be of either sign because clause 21 treats it in absolute value.
Invoking the gate wholesale would refuse an ordinary arm that gets cheaper as the dial shrinks, which is the opposite of what this case is guarding against.
Reproduced 2026-08-20: scaffold medians of 100, 130, 110 and 105.5 at `R = 1` have a range of 30 and a fitted excursion of 1.05, which clears `R`, while the fit's coefficient of determination is 0.0012 and its largest residual is over eighteen times `R`; without the gate that arm keeps a fitted slope, prices at 0.035, and can win clause 15's criterion outright.
The third case is the one the earlier version lost, and it is a refusal rather than a number because there is no honest number to record: the arm demonstrably moved and the only statistic on offer does not describe the movement.
The three cases are exhaustive at every boundary, including a range exactly equal to `R(w)` and a fitted excursion exactly equal to it, because the first case takes strictly below and the other two take at-or-above.
The RANGE is taken over the arm medians, the same reduction the pooled fit uses, and not over the raw per-round samples; the two disagree, and four arm medians all equal to 100 can sit on rounds spanning 99 to 101, giving a range of 0 against a range of 2.

Two PRECONDITIONS are evaluated before either of those two, both closing a hole a review reproduced rather than argued.

The registered SIGN rule stands and is evaluated FIRST, and it is about the KNOB's slope alone.
Clause 21's statistics table already rejects a candidate's reading on a non-positive slope, and an excursion condition written on the absolute value cannot see the sign, so a dial whose registered slope is negative would clear `10R` and be admitted by a rule that committed text rejects.
Reproduced 2026-08-20: per-round slopes of -60, -20, -20, 52 and 20 give a registered median of -20 and a pooled slope of -20, and at `R = 1` both absolute excursions are 15 against a limit of 10.
So a non-positive pooled or median KNOB slope rejects the reading before either excursion condition is evaluated, and the absolute values in the conditions are about MAGNITUDE only, never about admitting a sign the committed text already refused.
A negative knob slope means the step got slower as the operation got smaller, which is not a dial whatever its size.
A SCAFFOLD slope carries no such rule and may be of either sign: clause 21 already limits it by absolute value, so a scaffold that gets cheaper as the dial shrinks is an ordinary arm and rejecting it would refuse a reading committed text accepts.

`R(w)` must be POSITIVE.
An `R` of zero would make every positive slope clear `10R` and every strict excess clear clause 16's `2R`, so a harness reporting a resolution floor of zero is claiming it can resolve any difference at all, which is never true of a machine.
An `R` measured at zero at any width REFUSES the addendum, and no knob is dialled against it.
Whether this machine can produce a median `R` of exactly zero is a device question nobody has answered, so the refusal is registered rather than assumed unnecessary.

Three scope questions decide what these conditions actually do, and all three are answered here rather than left to a reading.

WHICH ESTIMATOR: both.
Amendment 5 gates the pooled fit and reports the median of the per-round slopes as `b`, and those two can disagree, so a condition on one certifies a number the rule does not consume.
Reproduced by execution on 2026-08-20: five exact per-round lines whose arm medians are collinear give a pooled slope of 20 and a median slope of 0.05, so at `R = 0.5` the pooled excursion clears `10R` while the median excursion is below `R`, and the share built from the median then excludes candidate A by route 1 where the pooled share would not have.
Requiring BOTH to clear `10R` closes that gap without changing which estimator feeds `f`, which stays the median exactly as clause 21's statistics table registers.

WHICH WIDTH: each, against its own `R(w)`, because `R` is registered per width and a condition evaluated at one width says nothing about the other.
Clause 26's SELECTION reads the long width's evaluation, and clause 27's validity reads each width's own.

WHICH KNOBS: the clamp applies to every scaffold slope wherever one is read, not to candidate A's dials alone, and that CHANGES clause 21's scaffold-slope limit rather than sitting beside it.
Clause 21 limits the scaffold slope to one tenth of the knob's slope, and that limit exists to catch a scaffold whose own cost moves with the dial and therefore enters the knob's slope; a scaffold whose entire excursion across the ladder is below what the machine resolves cannot be moving the knob's slope by a resolvable amount, so the limit would be guarding against something that is not there.
The consequence is registered rather than discovered: where a scaffold's excursion is under `R`, clause 21's scaffold-slope limit is satisfied by the clamp and is not binding, and the profile records that it was clamped rather than measured.

The clamp EXTENDS clause 18's registered convention to a new quantity rather than inventing a second treatment of the same situation.
Committed clause 18 records a RESIDUE below the resolution floor as zero and says nothing about any other number, so this is an extension of that convention's reach to a scaffold slope and not an appeal to a general rule committed text does not contain.
Its consequence is registered in advance: two dials whose scaffolds are both clamped to zero have equal prices, they tie, and clause 15's tie-break sends the tie to the higher completeness rank.
Applied to the proxy smoke above, where every scaffold slope sat inside jitter, the clamp would have zeroed every price and named kv-length by completeness at BOTH widths, so the instability in the table is what the clamp exists to remove; that is evidence the condition does work, and it is evidence rather than an input.

The multiplier 10 is a judgement, declared as a judgement, registered before the measurement it multiplies, in the same spirit as clause 21's four and Amendment 4's band.

The criterion applies only to dials that satisfy both conditions AND pass clause 6's linearity gate at the long width.
If no dial does, the criterion names no dial and candidate A has no knob and therefore no share at either width.
That is the registered falsifier of this clause's own premise: the claim that the long width resolves what the short one does not is a prediction, and this is what happens when the prediction is wrong.

If the named dial then fails a FIT LIMIT at the SHORT width only, the selection is not reopened and no second dial is tried, because a dial chosen for passing where another failed is a dial chosen by the answer.
Candidate A's short-width share is then `missing_share` at that width under clause 30, and clause 27 says what its ceiling can still decide with one width.

"Fit limit" is exactly clause 21's four limits plus this clause's two excursion conditions, and nothing else.
A trace during a timed round or a round refused by the spread gate is NOT a fit limit: clause 25 says a trace during timing is always a fault whatever caused it, and a refused round means the machine, not the dial, so both still refuse the run as they always did.
That distinction is the whole of the exception: a fit limit says this dial cannot measure this width, and the other two say this RUN cannot be believed.

Both of those outcomes need clause 11's pass condition to move, and this REVERSES it in part.
Clause 11 passes validation when "every knob's pooled fit clears clause 21's four limits at both widths", which would fail the whole validation on exactly the two continuations this clause registers, and a failed validation blocks the binding profile, so the registered terminals below would be unreachable.
The pass condition is now, in four parts:

| Fit | Must clear the fit limits at |
|---|---|
| The SHARE fits of candidates L and Q, and of the partition regions P1 and P3 | both widths |
| The SHARE fit of candidate A's dial, where clause 26 names one | the long width |
| Candidate A, where clause 26 names NO dial | nothing: naming no dial is a registered outcome of clause 26 and does not fail validation |
| The FLOOR fits of candidates L and Q, the denominators of their ratios | nothing: see below |

The rows say FITS rather than candidates, because a candidate has two of them and they fail differently.
The share fit is the numerator `b`, measured on stock; the floor fit is the denominator `d`, measured on the floor arm.
Gating the floor fit inside validation would make a bad denominator stop the run, which contradicts clause 30's `missing_ratio` being a continuable state; leaving it ungated entirely would let an unreadable denominator be credited.
Reproduced 2026-08-20: with `b = 20`, `d = 1e-12`, `R = 1` and `span = 0.75`, the stock excursion is 15 and the floor excursion is `7.5e-13`, and the resulting ratio is `2e13` with a gain of 1.2499999999999842, which reads as a candidate that eliminates its operation entirely.

So the floor fit is gated OUTSIDE validation, by the same conditions and with the same consequence as any other unreadable measurement: a floor fit that fails clause 21's limits, or whose excursion is below `10 * R(w)` by either estimator, or whose slope is non-positive, yields NO credited ratio for that candidate at that width, typed `missing_ratio` under clause 30, and the run continues.

**The condition reaches the PER-ROUND reduction too, because that is what clause 9 actually credits.**
Clause 9 takes the smallest per-round numerator over the largest per-round denominator, so the credited ratio is decided by individual rounds and not by the pooled or median statistics the conditions above test.
Reproduced 2026-08-20: per-round slopes of `1e-12`, 20, 20, 20 and 20 give a median and a pooled slope of 20, clearing both excursions and every fit limit, while clause 9 credits the `1e-12` round and returns a ratio of `5.46e-14`; replacing that round with -1 returns a NEGATIVE ratio, which no clause registers a meaning for at all.

The condition is written on the CREDITED quantities and not on the raw slopes, because those are different numbers and only one of them reaches the rule.
Clause 18 credits `A = b + c` for a rewrite and `A = b` for a retune, with the residue POOLED rather than per-round, so a round's credited numerator is its own slope plus one shared residue.
Reproduced 2026-08-20: with per-round slopes of `1e-12`, 40, 40, 40, 40 and a pooled residue of 40, the credited minimum numerator is 40 and the reading is perfectly ordinary, so a condition written on the raw slope would refuse a measurement the rule can read.
So the registered condition is: the smallest per-round credited NUMERATOR and the largest per-round credited DENOMINATOR that clause 9 reduces must each be positive and satisfy `value * span >= 10 * R(w)`, in the same excursion form the conditions above use and never in a bare comparison against `10 * R(w)`, which is a different test.
A reduction that does not yields no credited ratio for that candidate at that width, typed `missing_ratio`, rather than a number driven by the one round the machine could not resolve.
**Candidate L's bench gets its own registered `R`, because without one two of its limits cannot run at all.**
An earlier version of this clause gated candidate L's bench floor on "clause 21's limits and the sign rule only", and that is not a weaker gate, it is a broken one: clause 21's scaffold-OFFSET limit is `3R` and this clause's excursion condition is `10R`, so both silently vanish where no `R` exists, leaving only the scale-free limits.
Reproduced 2026-08-20: on the bench, `b = 20` with `d = 1e-12` and `span = 0.75` clears every remaining scale-free limit and the sign rule, and credits a ratio of `2e13`.

`R_bench(w)` is measured by the same identical-arms construction clause 21 registers, on the bench arrangement candidate L's floor actually uses, by the instrument-only stage of step 10, and it lands in the same addendum as the two step widths' own values.
It is keyed PER CELL AND PER WIDTH exactly as clause 21 keys `R`, and for the same reason: the bench arrangement differs between the two widths, so one value cannot serve both.
Reproduced 2026-08-20: at `d = 4` and `span = 0.75` a short-bench floor of 0.20 clears `3 >= 2` while a long-bench floor of 0.50 refuses at `3 < 5`, so borrowing the short width's value turns candidate L from absent into scoreable.
It is a THIRD registered CONTEXT beside the two step widths, named here before it exists, because an unregistered `R` context is exactly the fault an earlier draft of this amendment was blocked for.
Candidate L's floor fit is then gated exactly as every other floor fit is, against `R_bench(w)` in place of `R(w)`, and the fact that its resolution floor comes from a bench rather than a step is recorded beside its ratio wherever that ratio appears.

An `R_bench(w)` measured at zero refuses the addendum on exactly the same terms as a zero `R(w)`, for the same reason and with no separate argument.

Two committed lines about that addendum are in tension with each other and this clause resolves them rather than adding to the confusion.
Amendment 5's closing section calls it "one measured number ... and nothing else enters that addendum", while clause 21's own statistics row requires several, since `R` is measured per width and per cell.
The addendum carries ONE KIND of quantity, the resolution floor, in every context registered for it, and "nothing else enters that addendum" governs the kind and not the count.
Clause 26 adds one context to that list, the bench arrangement candidate L's floor uses, keyed per cell and width like the rest, and adds nothing else.

**A score built on a bench floor is not resolved by the step's `R` alone.**
Clause 16's tie test and clause 27's route 2 both convert a difference in score into a time and compare it against `2 * R(w)`, and that is the step's resolution; candidate L's score contains a floor measured on a bench whose own resolution is `R_bench(w)`, so a movement the bench cannot resolve can move candidate L's score by more than the step's test allows for.
Reproduced 2026-08-20: at `T = 100`, candidate L's attributed cost 20, a ceiling of 1.188, `R_step = 0.05` and `R_bench = 0.20`, a floor of 4 gives a score of 1.190476 and a route-2 gap of 0.175 against a demand of 0.10, while moving the floor by 0.19, which is less than `R_bench`, gives 1.187790 and reverses the exclusion, with both floor fits clearing `10 * R_bench`.
So wherever either test involves candidate L, its bench context is a CONTRIBUTING context and the demand must be met there as well as at the step.
An earlier version of this paragraph said the demand is taken against the COARSER of the two, which is a different rule and a wrong one: it collapses two contexts into one scalar, and clause 31 records why that cannot be done, since a resolution floor is only comparable to a difference converted with its OWN `T`.
This paragraph states the motivation and clause 31 states the rule; where they appear to differ, clause 31 governs and this paragraph is the history.
Clause 31 states that rule once, for every comparison the selection makes and not only these two, and names exactly which contexts contribute; this paragraph is the reason it exists and not a second copy of it.

**The excursion condition is a NEW limit and it reaches every knob.**
Clause 21 registered four limits and this clause adds a fifth, so a knob that clears all four and moves the step by less than `10 * R(w)` is now ineligible where before it was not.
Reproduced 2026-08-20: at `R = 1` and `span = 0.75`, a knob of slope 12 with a flat scaffold, a perfect fit and no residual passes every committed limit and fails this one, because 9 is below 10.
That expansion is deliberate and is recorded here rather than left to be discovered: a fit whose whole excursion is smaller than the machine resolves is a clean line through noise, and clause 6 tests a fit's SHAPE and never its SIZE.

A failure of candidate A's dial at the short width alone does not fail validation; it is typed `missing_share` at that width and carried into the profile as such.
Where no dial is named at all, candidate A is typed `missing_share` at BOTH widths, validation still passes on the rest, and the binding profile runs to the INCOMPLETE terminal clause 27's table registers for it.
Everything else in clause 11, the trace discipline, the spread gate and the complete-matrix demand as reinterpreted by clause 30, stands.

The same failure reaches clause 22, whose P2 region is measured by the dial this clause names.
The double-counting test is therefore PER WIDTH: at a width where P2 carries a typed absence the test is recorded as NOT RUN with that reason, it does not pass, and the width where P2 is valid still binds.
A test that cannot run at one width is a stated cost of the one-width continuation, recorded in the artifact, and it never converts a typed absence into a pass.

It can still REJECT, and this is the one case where a missing region does not stop the check.
Every share is non-negative, so a missing addend can only make a sum larger, and a subtotal of the MEASURED regions already above one is a proof that the full sum is above one whatever the missing region turns out to be.
Reproduced 2026-08-20: `P1 = 0.60` and `P3 = 0.50` subtotal to 1.10 with P2 absent, and every admissible P2 only raises it.
So the rule is: where the measured regions alone sum above one, the profile REJECTS at that width and says which regions proved it; where they do not, and a region is absent, the test is NOT RUN.
Recording NOT RUN in the first case would suppress a rejection the numbers already establish, which is the opposite of what a double-counting test is for.

That REJECT takes precedence over every terminal in clause 27's table, including INCOMPLETE, and the ledger row for clause 22 says so.
A rejection and a terminal are different things: a terminal is a ruling the profile reached, and a rejection is the profile saying its own measurements cannot be true, which is the fault class clause 27 already puts a non-fraction share in.
Without that precedence the same run is both REJECT and NOT RUN: make every attention dial ineligible so candidate A is `missing_share` at both widths, which the table sends to INCOMPLETE, while `P1 = 0.60` and `P3 = 0.50` prove the partition sum above one.

### What a gradient trace does to candidate A's floor

Section 4.2 nominates MLX's fused attention as candidate A's floor, and clause 19 requires every floor to be installed at the same seam as the dial and measured INSIDE the real step.
Neither can be done, and the reason was reproduced on this machine by reading the operation MLX built rather than by reading MLX's source.

| Question | Answer |
|---|---|
| Where does the fused kernel run | Head dimensions 64, 80 and 128, and at none of 32, 48, 72, 96, 160, 256 |
| Does a gradient trace keep it | No, at none of those three |
| Is that a property of the operands or of the trace | Of the trace: a call whose operands nothing differentiates is composed too |
| Does it hold inside mlx-lm's own step | Yes: the first layer's attention fuses untraced and is composed under `nn.value_and_grad` |

This settles the investigation section 4.2 left open, and it settles it on DISPATCH rather than on time: no timing appears in that table and none is claimed here.
Stock's training attention is the composed path in every block and both directions, so the fused entry point is a different implementation from the one stock runs, which is what the investigation doubted.
Whether it is faster is a GPU measurement nobody has taken, and this amendment neither takes it nor assumes it.

It also makes clause 19 inapplicable to candidate A.
A floor arm installed at the attention seam of the real step is composed by the trace and runs stock's own computation, so its ratio is one implementation divided by itself, and its only content is measurement noise while it looks like a measured floor.
What that number is exactly depends on the samples rather than on the operation: clause 9's worst pairing of two zero-spread distributions returns 1.0, and of noisy ones it can land either side of 1.0, so no particular value is claimed here.
That is worse than no number at all, which is why clause 27 takes no number.

### Clause 27. Candidate A has no floor and no credited ratio anywhere, and its verdict comes from a ceiling

This clause REVERSES clause 19 for candidate A: no floor arm for candidate A is installed anywhere, in the step or on a bench, and no credited ratio for candidate A exists in any recording, artifact or report.
Candidate Q is untouched and keeps its in-step floor; candidate L is untouched and keeps clause 19's written bench exception.

Candidate A keeps its share.
`f_A(w) = (b + c) / T(w)` per clause 18's rewrite row, with `b` from the dial clause 26 names, `c` from the same ablation cross-check clause 18 already requires, and clause 18's residue conventions unchanged.
A share is VALID at a width when its fit passed clause 21's limits and clause 26's two excursion conditions at that width, when clause 11's gates passed for that run, AND when the share is a fraction: `0 < f_A(w) < 1`.

That last requirement is not decoration, and it is registered because a construction defeats the rule without it.
Clause 22's partition reads `b/T` with no residue term while a rewrite candidate's share is `(b + c)/T`, so a large residue can carry a share past one while every fit gate and the partition sum stay clean.
Reproduced by execution on 2026-08-20: a pooled fit of `a = 13, b = 89` against `T = 100` with an ablated step of 1 clears the scaffold offset, clears `10R` twice over, fits perfectly, and leaves the partition summing to 0.99, while the share reads 1.01.

**A share that is not a fraction REFUSES the profile, for ANY candidate, and is not typed absent.**
An earlier version of this clause typed it `missing_share` at that width and continued on the other one.
That is weaker than the discipline it sits beside and it contradicts committed code: clause 22 already REJECTS the profile on a partition sum above one, the knob module refuses a non-fraction share where it is computed, and `gain` raises on a share outside `[0, 1]`, so the typed-absent reading registered an outcome the code refuses and the four terminals do not contain.
The registered rule is therefore: a share at or above one, or at or below zero, at ANY width and for ANY of the three candidates, REFUSES the profile outright, naming the candidate, the width and the value, and is recorded as an instrument fault.
The sibling width is not used, because it came from the same instrument and the same machinery; a share of 1.401 is exactly what exposed the instrument Amendment 5 exists to replace, and continuing on its other reading would be believing the same instrument twice.

**Every reading the rules consume must be FINITE.**
A NaN score satisfies neither `score >= 1.10` nor `score < 1.10`, so one non-finite number makes the terminal table neither total nor disjoint by defeating both branches of its own test.
Reproduced 2026-08-20: `ratio_lo([NaN], [1])` returns NaN and `gain(0.5, NaN)` returns NaN, and a profile carrying that beside a killed Q and an excluded candidate A matches no row of the table below.
A non-finite share, ratio, gain, step total or resolution floor REFUSES the profile in the same way and for the same reason as a non-fraction share.

**A refusal is not a fifth terminal.**
The four terminals below are terminals of a RULING, and a refusal is the run declining to produce one, which is the class clause 25's trace counter, clause 11's spread gate and clause 11's context-equality demand already belong to.
Those three refuse today and are not terminals, and these two join them rather than extending the table.
The distinction is the one clause 30 already draws between an absence and a fault: an absence is a measurement that legitimately does not exist and the rules carry it, while a fault is a measurement that exists and cannot be true, and no rule carries that anywhere.

What replaces the ratio is the registered gain formula's own supremum, on that domain and nowhere else.
For `0 < f < 1` and over POSITIVE FINITE `r`, `gain = 1/(1 - f*(1 - 1/r))` is increasing in `r` with supremum `U = 1/(1 - f)` as `r` grows without bound, so `U(w) = 1/(1 - f_A(w))` is the best score candidate A could earn at width `w` under any such ratio, including one no implementation could reach.
The domain on `r` is stated because the claim is false without it: at `f = 0.5` a ratio of -1.1 gives a gain of 22 against a supremum of 2, and -1 is a pole.
A ratio is a measured cost divided by a measured cost and is positive and finite by construction, so no reachable measurement leaves this domain; the restriction is there so the theorem is true as written rather than true in practice.
Committed code agrees and says so at the boundary: reproduced 2026-08-21, `gain(0.5, -1.1)` raises rather than returning the 22 the bare formula gives, because a ratio at or below zero is refused before the formula is evaluated.
So the domain restriction is a statement about the theorem and not a gap in what the harness can produce.
Verified by execution on 2026-08-20: the formula is monotone in `r`, approaches `U` from below, and in exact arithmetic attains it at no finite ratio.
The word exact is doing work and is not decoration.
Reproduced 2026-08-21 in double precision at `f = 0.087`: the computed gain is strictly below `U` up to a ratio of 1e14 and compares EQUAL to it from 1e15 upward, while the same computation in exact rational arithmetic is still below by about 1e-16.
That rounding cannot reach any disposition here, because candidate A has no ratio under this clause and never computes a gain: its ceiling is evaluated directly as `1/(1 - f_A(w))` from the share alone, and route 1 tests the share against `1/11` rather than the ceiling against 1.10, which is the second reason that test is stated on `f`.
Off that domain none of this holds, and WHICH expression fails has to be named, because the two behave differently and only one of them is undefined.
At `f = 1` the CEILING is undefined, since `1/(1 - f)` divides by zero, while the GAIN is perfectly defined there and equals `r` exactly; reproduced 2026-08-21, committed `gain(1.0, 3.0)` returns 3.0 and raises nothing.
Above `f = 1` the gain acquires a finite pole at `r = f/(f - 1)` and the ceiling goes NEGATIVE, which would read as a ceiling below the shipping floor and exclude candidate A on a broken measurement.
That is why the domain is part of validity and why route 1 below is stated on `f` rather than on `U`.

The domain is STRICT at both ends and committed code's is not, so the two boundary values are disposed of here rather than left to whichever check runs first.
A share of exactly 0 is UNREACHABLE through clause 26: it requires a fitted slope of exactly zero, whose excursion is zero, which fails the `10R` condition at every positive `R`, and `R` is now required positive.
A share of exactly 1 IS reachable through those conditions, because a slope equal to the whole step clears every excursion condition, and it is a FAULT: it says the candidate is the entire step with nothing left for the adapters, the norms, the elementwise work and the optimizer, which no step can be.
It therefore joins the refusal class named above beside the trace counter and the spread gate, and is not a terminal.
Committed `bench/profile_rules.py` accepts the CLOSED interval `[0, 1]` and returns `r` at a share of 1 without raising, so that refusal does not exist in code today; this amendment does not reverse the committed check, it adds the strict upper end, and step 8's rewrite carries the obligation.
Candidate A's selection score under clause 14 would be the worse of its two gains, so it is bounded above by `U_min`, the smallest `U(w)` over the widths where its share is valid, and bounded STRICTLY: no finite ratio attains the ceiling.

Two exclusion routes, and what follows when both fail depends on how much was measured:

| # | Route | Rule |
|---|---|---|
| 1 | Excluded by arithmetic | a valid share at EITHER width with `f_A(w) <= 1/11`, RESOLVABLY so under clause 31, excludes candidate A: its score is strictly below `U(w) = 1/(1 - f_A(w)) <= 1.10` and therefore strictly below the shipping floor. The test is on `f` and not on `U`, because the two are equivalent only on the domain `0 < f < 1` that validity already requires, and stating it on `f` keeps it right even if a share reaches the rule that should not have. The boundary rules OUT in the arithmetic, because `f = 1/11` puts the ceiling exactly at 1.10 and no finite ratio attains a ceiling. Under clause 31 the exclusion is nonetheless unreachable AT the boundary and in a neighbourhood of it, because a difference of zero resolves at no positive `R`: reproduced 2026-08-20, `f = 1/11` exactly converts to a difference of 0.00 ms against a demand of 0.10 ms. That is the conservative direction and it is registered rather than discovered, since exclusion is the action this route takes; the arithmetic boundary is stated inclusively so the rule stays right if the resolution test is ever satisfied there |
| 2 | Excluded by dominance | the L/Q winner's clause 14 score exceeds `U_min` and the excess is RESOLVABLE under clause 31, which fixes the conversion, the demand and the contributing contexts in one place and is the ONLY statement of them. An earlier version of this row added "at BOTH registered widths", borrowing committed clause 16's quantifier. That was a second rule for one comparison and it is withdrawn: both quantities here are REDUCED scores, which committed clause 16 never sees, so its per-width quantifier does not reach them. Reproduced 2026-08-21: at a score of 1.30 against a `U_min` of 1.25, with the short width at `T = 100` and `R = 0.1` and the long at `T = 10` and `R = 1`, the setting-width reading resolves at 3.077 ms against 0.2 and the both-widths reading fails at 0.308 ms against 2, giving SELECTED and UNRESOLVED on one measurement. "The L/Q winner" is the candidate section 4.3 SELECTS, which after the tie band and clause 12's table-order fall-through need not be the top scorer and never scores above it, so this is the reading that makes the exclusion harder to take. Where the two widths produce the same `U` exactly, the artifact names both as setting `U_min` |

Route 1 consumes only shares, so it is evaluated before any scoring.
Route 2 consumes the L/Q winner, so it is evaluated after clause 14 and clause 24 have run over candidates L and Q alone.
Where clause 24 admits no candidate at all there is no winner, so route 2 has no input and does not exclude, and candidate A's disposition falls to route 1 or to the table.
Verified by execution: over every pair of scores on a grid spanning the floor, no state has both a route-2 winner and every score below the floor, so the table's second row naming route 1 alone leaves nothing uncovered.

Which candidate counts as the winner is a real choice and it is made here rather than in code.
Reproduced 2026-08-20: at scores of 1.300 for candidate Q and 1.290 for candidate L the two sit inside the two-point band, the band is both of them, clause 12's second condition sends it to table order, and candidate L is selected at 1.290 while the top scorer is candidate Q at 1.300; at `U_min = 1.295` the top-scorer reading excludes candidate A and the selected-candidate reading does not.
The selected candidate is registered, on the same principle clause 8 and `ratio_lo` each already follow: each picks the reduction that makes its own action harder to take, and exclusion is the only action this route takes.
It is also the comparison that means something, because the selected candidate is the operation the sprint would actually build, so the question this route asks is whether building it forecloses candidate A.
Candidate A never enters clause 14's scoring, clause 24's floor or tie band, clause 16's tie machinery, or clause 12's footprint tie-break, because a candidate without a ratio has no gain to score, no tie to break and no delta to compare.

Both exclusion routes are SOUND with a valid share at one width only: the true two-width ceiling is at most the known one, so a winner that beats the known ceiling beats the true one, and an arithmetic exclusion at the valid width needs no second width at all.
Verified by execution with the rest of the ceiling arithmetic.
A FAILED exclusion is not sound with one width, and this is the asymmetry the terminals below encode: the missing width could carry a smaller share and a lower ceiling that route 1 or route 2 would have cleared, so failing to exclude on one width is a missing measurement that could be decisive, and the honest verdict is INCOMPLETE rather than UNRESOLVED.
With no valid share at any width there is no ceiling and no route can run, and the terminal is INCOMPLETE naming both absences, exactly as the table's last row says; this is an absence and not a fault, so it does not refuse the run.

The profile has exactly four terminals, one per run: SELECTED, the registered no-selection record of section 4.3, UNRESOLVED, and INCOMPLETE.
Their precedence is INCOMPLETE first, UNRESOLVED second, and the two registered outcomes last.

One guard runs BEFORE the table and is written here rather than left implicit, because a table over candidate A's states cannot see a fault in a different candidate.
If candidate L or candidate Q reduces to `missing_share` under clause 30, the terminal is INCOMPLETE naming that entry, whatever candidate A's state is and whatever the remaining scores say.
Only `missing_share` does this: a `killed` or `missing_ratio` candidate is excluded and visible and the profile continues, because those are answers rather than absences.

With that guard passed, the table below is total and disjoint, where "scored set" is the candidates among L and Q that clause 30 leaves scoreable:
Verified by execution 2026-08-21 over all 2304 combinations of candidate A's share validity, candidates L and Q's typed absences, the shipping floor's arithmetic reading and both exclusion routes' arithmetic readings, each crossed with clause 31 resolving or not resolving at its three sites here.
Every state lands on exactly one of the four terminals, no state reaches a terminal that clause 31 declares because clause 31 declares none, and no state reaches SELECTED through a comparison the machine could not resolve.

| Candidate A's state | Scored set's outcome | Terminal |
|---|---|---|
| Excluded by route 1 or route 2 | a winner at or above the floor | SELECTED, with candidate A's exclusion recorded |
| Excluded by route 1 | every score below the floor, or no scores at all | the no-selection record, with the kept list holding however many scored candidates exist, two, one or none, and every exclusion named |
| Valid at both widths, neither route excludes | any | UNRESOLVED: the artifact names the winner-so-far or its absence, `U_min`, the width or widths that set it, and which route failed by how much; where the scored candidates all sit below the floor their reading is recorded INSIDE this artifact, because one run gets one terminal |
| Valid at one width, neither route excludes | any | INCOMPLETE: the missing width is named as the decisive absence |
| No valid width | any | INCOMPLETE, per clause 30 |

The no-scores row of the no-selection record is reachable because clause 30 can exclude candidates without making the matrix incomplete: a killed Q beside an L with no ratio leaves nothing to score and nothing unmeasured, which is an answer, not an absence.
A dominance test with no winner cannot run, so with no scored winner an unexcluded candidate A always lands on the UNRESOLVED or INCOMPLETE row by its own width count.

One bias is stated rather than absorbed.
The committed ablation used for `c` also removes the key and value projection backward, pinned by its own test, so `c` and therefore `f_A` and `U` read HIGH.
An inflated ceiling makes both exclusion routes harder to satisfy, so the bias pushes toward UNRESOLVED and never toward wrongly ruling candidate A out, which is the safe direction for a rule whose only action on this candidate is exclusion.

What this clause gives up is stated plainly: candidate A can be ruled OUT by measurement or the profile can refuse to rule, but candidate A can never be ruled IN, because ruling it in would need the ratio this amendment abolishes.
That asymmetry is the honest shape of the situation: the floor that would have supplied the ratio cannot be measured where the share lives, and a rule that cannot measure a number does not get to use one.

### Clause 28. Where the fused kernel places on each dial, kept as evidence

Measured 2026-08-20 on clause 1's registered ladder of 1.00, 0.75, 0.50 and 0.25, at head dimension 128:

| Dial | Fused settings it places | Why |
|---|---|---|
| kv-length | 4 of 4 | the fused kernel does not constrain the key and value sequence length, and it accepts the array mask a shortened key set needs |
| head-dim-qkv | 2 of 4 | the ladder reaches 128, 96, 64 and 32 and the kernel fuses only at 128 and 64 |
| head-dim-qk | 1 of 4 | the kernel additionally requires the value head dimension to equal the query and key head dimension, so only the full setting fuses at all |

Under the earlier draft this table decided which dial could carry a bench floor.
With no floor it decides nothing, and it is kept because it closes the door the draft left open: two of the three dials place too few fused settings for any floor ladder to fit one implementation, so even the rejected bench design could not have produced a clean four-point floor fit on a head-dimension dial, and a line through two implementations looks fine and means nothing.

### Clause 29. Why the bench exception was rejected, in writing

The earlier draft measured candidate A's `r` on a bench by four arms and claimed clause 10 stood unchanged.
Two review rounds blocked it, every finding was reproduced here before it was admitted, and the two that killed the design are recorded so the rejection is auditable:

| Finding | Reproduction |
|---|---|
| A bench ratio paired with an in-step share breaks clause 18's exact identity | on `T = 100`, `A_step = 20`, `A_bench = 15`, `F_bench = 6`: the registered formula with the bench ratio returns 88 where the replacement arithmetic returns 86, and the difference is an unmeasured in-step floor cost the pairing silently assumes; clause 18's whole argument is that `f` and `r` share one `A`, and a bench `r` does not |
| The bench's fused-forward-plus-stock-backward arm is a mechanism no seam produces | outside a gradient trace nothing generates a backward at all, inside one the forward composes, and a hand-built vjp would be a new mechanism needing its own registration and falsifier; the symbol `ScaledDotProductAttentionVJP` exists in the installed libmlx and was measured UNREACHABLE from Python in 0 of 72 attempted configurations on this machine, 2026-08-20 |

Clause 10's voids are unchanged by that rejection: section 4.2's "times 3" and its "twice the forward" relation remain void and remain unreplaced.
Clause 10's floor construction, "the whole attention region with MLX's fused implementation installed, measured in the step by the same dial", is REVERSED along with clause 19's requirement it implemented: there is no floor for candidate A anywhere, so there is nothing for that sentence to govern.
Clause 10's call counts, 36 forwards and 16 backwards on the pinned arrangement, stand as structural facts under clause 3, and nothing consumes them in a ratio because no ratio for candidate A exists.

### Clause 30. A candidate that loses a measurement is ABSENT with a typed reason, not omitted

Clause 11 requires a complete three-candidate by two-width matrix and refuses an incomplete one.
Clauses 26 and 27 admit outcomes in which an entry is legitimately not a number, so what the matrix means has to be registered rather than left to the code.
Every absence is TYPED, and the type decides the branch:

| Type | Meaning | Effect |
|---|---|---|
| `missing_share` | a candidate has no valid share at a width the rule needs | the matrix is INCOMPLETE: the profile refuses the verdict SELECTED, names the entry and its reason, and reports nothing as chosen. For candidate A alone, a share missing at ONE width narrows clause 27's ceiling to the valid width rather than refusing, and missing at BOTH widths refuses |
| `missing_ratio` | a share exists and no credited ratio does | for candidates L and Q this is Amendment 5's existing exclusion, now typed: the candidate is excluded from scoring, stays visible in the artifact with its share, and the ruling records why. For candidate A it is not an absence at all: it is this amendment's permanent state, and clause 27 is its disposition |
| `killed` | candidate Q removed by the kill rule | excluded from scoring, visible in the artifact with the kill record, exactly as section 5 already provides; typed here so the selection's input accounts for it |

The entries are PER WIDTH, and a candidate can carry different absences at the two widths, so the reduction to one candidate-level state is fixed here rather than left to whichever branch the code checks first.
The precedence is `killed`, then `missing_share`, then `missing_ratio`.
`killed` wins because the kill rule reads the ceiling sweep and not the knob, so a killed candidate's other absences no longer matter and do not make the matrix incomplete.
`missing_share` beats `missing_ratio` because a candidate with no share at some width cannot even be partially assessed, and the profile is INCOMPLETE whichever other numbers it has.
Worked out: a Q that is killed and also missing a share is killed, excluded and visible, and the profile continues; an L missing its short share and its long ratio is `missing_share`, and the profile is INCOMPLETE.

The selection rule takes the absences as an INPUT beside the readings.
Its input must account for exactly the three registered candidates, each present with its numbers or absent with one typed reason after the reduction above, and it refuses anything else: a duplicate, an unknown name, an untyped absence, or a candidate simply not mentioned.
A candidate can therefore never disappear between the profile and the ruling, which is the fault this clause exists to make impossible.

If no candidate has a score at all because every scored candidate is excluded by type, that is an answer rather than an absence, and clause 27's terminal table says which terminal it feeds; it is not INCOMPLETE by itself.

INCOMPLETE and UNRESOLVED are different verdicts and both refuse SELECTED.
INCOMPLETE means a required measurement does not exist, including the case where the one measurement that could have excluded candidate A is the one that is missing.
UNRESOLVED means every measurement the rules can consume exists and the registered rules still cannot separate candidate A's ceiling from the field.
Both are legal terminals of the decision artifact, neither is a failure of the harness, and clause 27's table is the one place their precedence is written.

Clause 22's disjoint partition names its P2 dial as whatever clause 26 names, and its test runs per width as clause 26 registers.
If clause 26 names no dial, P2 is unmeasured at both widths; the profile is then INCOMPLETE through candidate A's own `missing_share` at both widths, and the partition record says NOT RUN with that reason at each.

### Clause 31. Every rule's action must survive its own measurements' uncertainty

This clause has been written four times and the first three were wrong in ways reviews reproduced, so what it got wrong is recorded before what it now says.
The first declared one terminal for every site, which REVERSED committed clause 16.
The second chose a single governing context as the largest `R`, which is not an ordering at all, because the demand is a time and the same dimensionless gap is a different number of milliseconds at a different `T`.
The third converted each site's difference to a time and compared it against a scalar demand, and that is the one this clause replaces: a scalar demand assumes each measurement's uncertainty reaches the compared quantity ONE FOR ONE, and it does not.
Reproduced 2026-08-21: with a step of 100 ms, a credited numerator median of 50, a credited numerator minimum of 10 and a floor of 3, a floor movement of 0.11 ms moves the predicted step by 0.55 ms, an amplification of exactly five, which is the ratio of the median to the minimum.
The share takes the MEDIAN per-round slope and `ratio_lo` takes the SMALLEST numerator over the LARGEST denominator, and those are different reductions of one measurement, each chosen to make its own rule harder to take.
That is committed and intended, and it is precisely why uncertainty does not transfer one for one.

**The principle: bound the ACTION, not the gap.**
Every rule here takes an action when a signed quantity is positive: a candidate ships, a route excludes, a shape counts toward a kill.
Write that quantity `d(z)`, where `z` is the vector of measurements it rests on, and let `B(z)` be the registered uncertainty box around them.
The action is SUPPORTED only when it survives everywhere in the box:

    inf over B of d(z) > 0

**That inequality is STRICT, and being strict REVERSES four committed equality outcomes.**
Committed text lets a gain of exactly 1.10 ship, puts a gap of exactly 0.02 outside the tie band, puts a price gap of exactly 0.01 outside the dial tie, and counts a shape ratio of exactly 1.10 toward the kill.
Under a strict margin each of those sits at `d = 0`, is not supported, and takes the failing direction instead: the candidate does not ship, the pair is tied, the prices tie, the shape does not count.
The reversal is deliberate and is the same principle every other reduction in this amendment follows, because in all four the failing direction is the one that makes that rule's own action harder to take.
It is recorded here and in the ledger with the word rather than left for a reader to discover at a boundary.
An exact equality is also not a measurement this machine can produce twice, so the rule that changes is one no run is likely to reach; that is a reason to state the change plainly, not a reason to leave it unstated.

This is deliberately not `sup |d(z) - d(z0)|`, which counts movement AWAY from the boundary as though it threatened the action; only movement toward zero can take the action away.
Each measurement enters the box ONCE, so a step total shared by two candidates, or the common samples behind a median and a minimum, are not counted twice.

**The common algebra, which makes every score site one expression.**
For a candidate at a width, with `M` the median per-round credited numerator, `N` the smallest, `F` the largest credited denominator and `T` the step total:

    K = M * (1 - F/N)          the credited saving
    P = T - K = T/g            the credited replacement step

so the gain is `T/(T - K)` and every score comparison is a comparison of savings.
Its sensitivities are `dK/dM = 1 - F/N`, `dK/dN = M*F/N^2` and `dK/dF = -M/N`, so `dP/dF = M/N`, which is the amplification reproduced above rather than an assumption.
Over a box with the reduced scalars in intervals and while `0 < F < N` holds, the saving's own bounds are exact:

    K_lo = M_lo * (1 - F_hi/N_lo)          K_hi = M_hi * (1 - F_lo/N_hi)

The authoritative construction recomputes `M`, `N` and `F` from the SHARED source samples rather than boxing the three summaries independently, because independent boxes admit combinations the raw rounds cannot produce.
Verified by execution 2026-08-21, because each of these is a claim a site's margin rests on and a wrong one would put a rule's sign the wrong way round: over 200000 random draws the sign of a gain difference matched the sign of a saving difference with no exception, and clause 16's registered time conversion equalled the saving difference exactly, to within 7e-13; the band expression matched the two-point gain test's sign over 200000 draws with no exception; and every interior point of 50000 random operand boxes fell inside the saving bounds above.

**The site registry.** Each row is a signed margin and the direction its failure takes, which is unchanged from the previous draft: a site that cannot support its action does not take it.

| Site | The action's margin | Not supported means |
|---|---|---|
| Clause 24's shipping floor | `K - T/11` at each width, since a gain of 1.10 is a saving of `T/11` | the candidate is treated as BELOW the floor and does not ship |
| Clause 27's route 1 | `T/11 - M_A` at a width, on candidate A's share alone | candidate A is NOT excluded by this route |
| Clause 27's route 2 | the winner's minimum-over-widths saving fraction minus candidate A's, whose sign is the sign of the winner's score minus `U_min` | candidate A is NOT excluded by this route |
| Clause 16's per-width pair | `K_1 - K_2` at each width, which has the same sign as the gain difference and equals its time conversion exactly | the pair is TIED and continues through committed tie-breaks |
| Clause 16's two-point boundary | `E_a - E_l - 0.02 * (1 - E_a) * (1 - E_l)`, on the saving fractions, which has the same sign as the two-point gain test | treated as INSIDE the band |
| Section 4.3's retained order | the difference of the two saving fractions | the pair is recorded unordered and section 4.2's table order stands in |
| Clause 15's dial price | the competitor's price minus the observed best's, minus the registered 0.01 | the prices TIE and the completeness rank governs |
| Clause 23's kill rule | per round, the floor cost minus the stock cost divided by 1.10, so a shape counts only if every round supports it | the shape does NOT count toward the kill |

The dial price propagates the PRICE and not the scaffold excursions, with `p_lo = dist(0, [s_lo, s_hi]) / b_hi` and `p_hi = max(|s_lo|, |s_hi|) / b_lo`, because two dials with different knob slopes have excursion differences unrelated to their price order.
The kill rule uses the operand margin directly and not a converted ratio, because a conversion through a shape's stock slope amplifies the floor's uncertainty by that slope over the round's own numerator, which is the same fault this clause exists to remove.

**The two-width minimum must be optimised INSIDE the box, and corners are not enough.**
Route 2, the retained order and the band all compare quantities that are a minimum over two widths, and a minimum's argument can switch inside the box.
Reproduced 2026-08-21 by enumeration: over four thousand random admissible boxes the corner-only minimum overstated the true minimum by as much as 0.042, because the true minimum was attained on a setting-switch line and not at a corner.
An earlier version of this clause then claimed the exact evaluation is over the corners TOGETHER WITH every intersection of a switch line with a box edge.
That claim is WITHDRAWN and it was wrong: the saving is rational in its operands and the shared-source pipeline is piecewise rational, so a minimum can sit strictly inside the box on a path where several leaves move together, with no switch and no edge involved.
Reproduced 2026-08-21 along a single shared direction with `M = 16 + 3z`, `N = 1.7 + 5z` and `F = 0.2 + 1.4z`: the saving is 14.118 at `z = 0` and 14.463 at `z = 1` but 13.624 at `z = 0.279`, an interior minimum below both ends, with `0 < F < N` holding throughout.
What is registered instead is the REQUIREMENT rather than a recipe: the evaluation must be a certified lower bound over the whole box, computed with the shared expressions kept shared, and any point of the box at which a denominator is non-positive or a margin undefined makes the action unsupported.
Freezing the width observed to set a minimum remains unsound, and corner enumeration is now known to be unsound too.
That also removes the previous draft's near-setting-width admission test outright: choosing which widths to inspect was an attempt to approximate this, it is not itself a ruling, and joint optimisation already includes every switch.

**What this clause does NOT reach, unchanged from the previous draft.**
The validity guards refuse rather than rule, and a readability gate whose threshold is already denominated in `R` cannot have its boundary removed by another test, only moved; both keep the treatments recorded below and in clause 26.

**OPEN, and it blocks the binding run: the box is now DESIGNED but not affordable as designed.**
Clause 33 registers a calibrated demand from measured null blocks, and that is ONE scalar contrast between two arm medians.
It does not define the SIMULTANEOUS box over every arm, width, candidate, shape and reduction that this clause's lower bound must be taken over, and assigning that scalar independently to each leaf would abandon the registered rate rather than apply it.

What a design pass established, and what it costs, are both recorded here because the second is a fact about this sprint and not only about this clause.

The box is over PER-ARM, PER-ROUND timing samples, taken immediately before the medians, fits, minima and maxima that consume them.
It is not over arm medians, because the kill rule and the credited ratio consume individual rounds, and it is not over `M`, `N`, `F` or any score, because independent derived boxes admit combinations the raw rounds cannot produce.
Every derived quantity is recomputed from the same perturbed leaves.
The calibration that holds the registered rate without assuming a tail model is a grouped maximum: predeclared groups, a pilot set of blocks fixing one positive scale per group, then fresh blocks whose per-block score is the largest standardised leaf contrast, with the demand taken as an order statistic of those scores.
In simulation over 1200 synthetic experiments with correlated heavy-tailed noise, that construction held 0.961 joint coverage against a nominal 0.95 and agreed with a true-scale oracle on 99.1% of rulings away from the boundary and 84.5% in the boundary-dense region, at a median radius about 14% wider.
A distribution-free per-coordinate correction was also priced and is not available: at the step-only manifest of about 450 leaves it needs at least 8999 blocks.

The cost is the blocker.
The step-only manifest is at least 45 arms per width, because calibration must precede the dial selection and therefore carries all three attention dials, and at the two registered widths and five rounds that is at least 450 leaves before candidate L's bench floor and the kill sweep exist at all.
One complete null block is two replicas of that manifest, so at proxy step times one block is about 33 minutes, and the recommended pilot-plus-calibration schedule is about 43 hours of pure timing.
That figure excludes warm-ups, compilation, the floors, the sweep and every idle refusal, so it is a floor and not an estimate.
Forty-three hours is not affordable inside this sprint, and saying so is the honest reading of the design rather than a reason to quietly weaken it.

A second, structural problem sits beside the cost: step 10 runs BEFORE the dial is selected and before the floors and the sweep exist, so it cannot produce the authoritative vector box at all, whatever it costs.
Either the complete-vector calibration moves after those arms exist, or the scalar floor calibration splits from a new post-manifest, pre-decision stage.

So this clause registers the CONSTRUCTION and refuses to register a schedule for it.
Until a box is both defined and affordable, no run of this profile binds anything, and the choice among reducing the manifest, accepting a proxy transfer that no measurement supports, and re-scoping what the profile decides is a ruling that has not been taken.

### Clause 32. Amendment 5's replacement thresholds, restated here rather than edited there

An earlier version of this amendment rewrote two rows of Amendment 5's replacement-threshold table IN PLACE.
That was a process fault and it is reversed: committed text is preserved wherever it stands, and what supersedes it is written here, which is how every other change in this amendment works.
The document's whole discipline is that a reader can see what was binding at the time a measurement was taken, and an in-place edit destroys exactly that.
The two rows are restored to their committed wording and are REPLACED by the two rows below for every figure this amendment relies on.

Clause 4 registers when a figure measured uncompiled is replaced by its compiled value.
Its time row reads `R` and its ratio row reads `R / F`, and both were written when `R` named one quantity.
Candidate L now has a step context and a bench context, so a bare `R` names two numbers and admits two opposite readings of whether a figure moved.

| Kind of figure | Replaced when the compiled value differs by more than |
|---|---|
| a time | `R` in the context that time was measured in, `R(w)` for a step time and `R_bench(w)` for a bench time |
| a ratio `r = A / F` | `(F * R_num + A * R_den) / (F * (F - R_den))`, where `R_num` is the resolution of the context the numerator was measured in and `R_den` that of the denominator |

Clause 4's share row and its count row are untouched, and its rule that a figure inside its threshold stands as written is untouched.

**The ratio bound is exact rather than first-order, and the difference is reachable.**
An earlier version of this clause wrote `R_num / F + A * R_den / F^2`, which is the first-order expansion and understates the movement two unresolvable operand movements can produce together.
Reproduced 2026-08-21: at `A = 20`, `F = 4` and `R_num = R_den = 0.2`, movements of 0.199 in each operand, both strictly inside their own resolution, move the ratio by 0.3141, while the first-order threshold is 0.3000 and would have called that a real change.
The exact supremum is `(A + R_num) / (F - R_den) - A / F`, which is the expression in the table, and a brute-force search over the corners of the operand box confirms nothing exceeds it.
The bound requires `F > R_den`, which is the same condition as the floor slope being a measurement at all, and a floor whose slope is inside its own resolution has no credited ratio under clause 9 in the first place.

**Which operands set the threshold is fixed here, because the two readings disagree.**
The threshold is computed from BOTH the uncompiled and the compiled operand values, and the figure is replaced only if the movement exceeds the LARGER of the two thresholds.
Reproduced 2026-08-21 at `A = 20`, `F = 4`, `R_num = R_den = 0.2` with compiled operands `A = 20.199` and `F = 3.801`: the ratio moves by 0.3141, the EXACT bound from the old operands is 0.3158 and from the compiled ones 0.3507, so the movement clears neither and the figure STANDS under both.
The two bounds still differ by 0.035, which is enough to disagree on a movement between them, so the rule is stated rather than left to whichever operand set the code happens to hold.
Taking the larger is the direction that makes replacement harder, which is the principle every other reduction in this amendment follows, and it needs no rule about which measurement is authoritative.

**This clause reaches candidates L and Q only.**
Candidate A has no ratio at all under clause 27, so the ratio row never applies to it, and the committed row's mention of a floor cost has no referent there.

### Clause 33. Units, ordering, and what steps 7 to 10 must produce

An independent pass tried to WRITE the selection function from this amendment alone and reported that it could not, for reasons that were not disagreements about the rules but quantities the rules consume and nothing supplies.
Those are registered here rather than left for the implementation to invent, because a quantity invented at implementation time is a rule made after the numbers.

**One unit, stated once, and this is a CHANGE rather than a description.**
Every time this amendment's rules consume is in MILLISECONDS: step totals, fitted slopes and intercepts, residues, both resolution floors and every threshold derived from them.
Committed text and committed code do not currently work that way and the amendment does not pretend otherwise: section 8's worked example gives its samples in seconds, the knob timer returns seconds, and current recordings store fields named for seconds.
So the conversion happens where a recording is READ, once, and never where a rule is applied, and a rule that receives a number is entitled to assume milliseconds.
Reproduced 2026-08-21: a step total of 0.1 with `R` of 0.000155 and the same total of 100 with `R` of 0.155 are the same measurement, and mixing the two scales resolves a comparison that the other refuses.

**`T` must be POSITIVE and finite, not merely finite.** A non-positive step total is a fault and joins the refusal class.
Reproduced 2026-08-21: a `T` of -100 converts a real score difference into a negative time of -7.58 ms, which clears no demand and would silently refuse every comparison at that width.

**The kill rule runs BEFORE clause 30's reduction, and `killed` enters that reduction as its result.**
Clause 30 reduces a candidate's absences by the precedence `killed`, then `missing_share`, then `missing_ratio`, and `killed` is the outcome of clause 23 rather than an input the profile carries independently.
Reproduced 2026-08-21: candidate Q carrying `missing_share` while its ceiling-sweep readings kill it at five of six shapes gives INCOMPLETE if the absence is reduced first and a no-selection record if the kill is evaluated first, on one set of measurements.
The registered order is: clause 23 evaluates the kill from the sweep, its verdict is written as the typed absence `killed`, and only then does clause 30 reduce.
That order is forced rather than chosen, because clause 30's own reason for putting `killed` first is that a killed candidate's other absences no longer matter, which presupposes the kill is already known.

**`R` itself admits two readings, and the one that binds is fixed here, against an earlier draft of this clause that fixed the wrong one.**
Committed clause 21 defines `R` as "the median absolute difference between two IDENTICAL arms over the registered rounds at that cell", and that sentence parses two ways: the MEDIAN OVER ROUNDS of each round's paired absolute difference, or the absolute difference between the two arms' MEDIANS.
They are not the same number and neither is a misreading of the words.

The reading that binds is the SECOND, the absolute difference of the two arms' medians.
An earlier draft of this clause registered the first and gave a reason that is wrong: it said the paired reading answers "how far apart can two readings be before the difference is the machine", which is a statement about a single round, and NOTHING in these rules consumes a single round.
Clause 21's statistics table takes the median of the per-round slopes, every fit runs over arm medians, and every share and gain is derived from those.
If `R` exists to say when a difference between two DERIVED readings is real, the honest null is two identical arms carried through exactly the reduction the rule uses, which is the second reading.
That earlier draft also claimed the second reading "can report a large floor from arms that agree in every round", which is arithmetically impossible, since arms agreeing in every round have equal medians and a null of exactly zero.
The claim is withdrawn rather than softened.

**One block of rounds cannot estimate it, and more rounds in one block do not fix that.**
The null contrast from a single block is one draw from a distribution, not a scale estimate, and its relative spread does not shrink with the round count: under an ideal null it converges to an absolute Normal draw whose coefficient of variation is `sqrt(1 - 2/pi) / sqrt(2/pi)`, about 0.76, whatever `n` is.
Simulated 2026-08-21 over 400000 nine-round blocks: the paired statistic has a coefficient of variation of 0.36 and the of-medians statistic 0.76, with a central 90% range from 0.09 to 2.91 times its own median.
So `R` is the MEDIAN OVER `m` INDEPENDENT COMPLETE BLOCKS of each block's null contrast, and `m` is a registered count rather than a convenience.

**The multiplier 2 is not a resolution test, it is a false-separation rate, and the rate is worse than it reads.**
Committed clause 16 demands `2 * R`, and that construction was registered as a judgement before anything had measured what it buys.
Simulated 2026-08-21 under an ideal null with `R` at its true value: a fresh null contrast exceeds `2 * R` 17.7% of the time, so the test as written admits noise as a separation about one pair in six.
With `R` taken from a single block instead of its true value the rate is 29.5%, which is the estimator fault above compounding the threshold fault.
The paired reading appeared to behave better, at about 2%, and it did so for the wrong reason: its null is 2.46 times larger, so `2 * R_paired` is roughly `4.9 * R`, and it bought conservatism by measuring a different and larger quantity rather than by controlling an error rate.

What replaces the judgement is a rate registered in advance and a multiplier measured against it.
The registered false-separation rate is `alpha = 0.05`, fixed here, before the null is measured.
Step 10 measures the null contrast over its `m` blocks and reports the EMPIRICAL critical value `C`, the `ceil((m + 1) * (1 - alpha))`-th smallest of them, which controls a future exchangeable null exceedance at no more than `alpha` without assuming the null's shape.
Under an ideal null that lands the multiplier near 2.9 rather than 2, but the multiplier is not registered here and the ideal null is not claimed to describe this machine: what is registered is `alpha`, the estimator, and the block count.

**`C` governs the selection comparisons and `R` continues to govern the readability gates, and they are NOT interchangeable.**
Clause 31's seven sites take their demand from `C`.
Clause 21's scaffold-offset limit at `3 * R` and clause 26's excursion condition at `10 * R` continue to take `R`, the median null magnitude, exactly as written.
Reusing `C` in those gates would silently redefine two committed limits that have a different job: they ask whether one measurement is readable at all, not whether two measurements differ.
Both quantities come out of the same step-10 blocks and neither costs a separate measurement.

**None of this changes a gate's strictness relative to any measurement, because `R` has never been measured at the target.**
Every figure this pre-registration quotes for a resolution floor was taken at the 0.6B proxy, and the only identical-arms probe in the repository fixes one width, one batch and nine rounds and measures no bench context at all.
So fixing the reading now SETS what `3 * R` and `10 * R` mean for the first time rather than moving them.

**Three readings an implementation pass could not resolve, settled here rather than by whichever branch the code checks first.**

FIRST, what a negative residue rejects.
Committed clause 18 says a residue negative by more than `R` "REJECTS that candidate's reading", and clause 27 puts impossible measurements in the refusal class rather than among the typed absences.
Those pull in different directions for the same event, so the reading is fixed: a residue below `-R` is a FAULT, it refuses the profile rather than typing that candidate absent, and the raw residue is retained and named in the recording's blockers.
The reason is the one clause 27 already gives: an absence is a measurement that legitimately does not exist and the rules carry it, while a negative residue is a measurement that exists and cannot be true, since it says the step ran SLOWER with the operation removed than the fit predicts without its scaling part.
Typing it as an absence would let a broken construction produce a ruling on the two candidates that happened to survive it.

SECOND, what context equality demands across two widths.
Clause 11 requires a complete matrix whose entries share one context, and read literally that is unsatisfiable, because the two registered widths differ in width, in band, in token count and in supervised fraction by design.
So the demand is fixed in two parts.
WITHIN a column, meaning one width, every entry must agree exactly on every context field: model, revision, adapter configuration, batch, band, width, seed, optimizer state and the process being stock.
ACROSS columns, every field EXCEPT width, band, token count and supervised fraction must agree exactly, and those four must differ exactly as the corpus registration says they do.
A field that differs across columns and is not one of those four refuses the matrix, which is the check that stops a short-width share being paired with a long-width floor.

THIRD, whether an absent reporting-cell share is a blocker.
Cells A, C and D decide nothing under section 4.3 and are reported beside the decision.
A typed absence in a reporting cell therefore does NOT make the recording nonbinding, and is recorded with its reason exactly as a deciding-cell absence is.
The deciding cell B is different and clause 30 governs it unchanged: an absence there is INCOMPLETE by the rule that reads it.
Stating this is not a licence to lose a reporting cell quietly, because the absence still appears in the record with its type and the artifact still names it.

**Two artifacts carry a measured number from one stage to the next, and neither had a shape.**
An implementation pass could not write step 7 without inventing both, and an invented transport is a place for a number to change meaning between the stage that measured it and the rule that reads it.

The RESOLUTION ADDENDUM, written by step 11 and read by every stage after it, carries exactly this and nothing else:

| Field | Meaning |
|---|---|
| `schema_version` | so a reader can refuse a shape it does not know |
| `blocks` | the block count `m`, which must be at least 19 for the registered `alpha` to be attainable |
| `rounds_per_block` | the round count inside one block |
| `alpha` | the registered false-separation rate, 0.05, copied here so the artifact is self-describing |
| `contexts` | a mapping keyed by `(cell, width, arrangement)`, where arrangement is `step` or `bench`, each carrying `R`, `C`, the `m` block contrasts they were derived from, and the step median `T` |
| `recording_sha256` | the step-10 recording these came from |
| `rules_sha256` | the digest of the rule code that computed `R` and `C` from the blocks |

A context key absent from that mapping is not a resolution floor of zero and not a licence to reuse another context's: a stage that needs a context the addendum does not carry REFUSES, which is clause 21's prohibition on reuse made executable.

The DIAL ARTIFACT, written by step 12 and read by step 13, carries clause 26's selection so the binding run cannot re-decide it:

| Field | Meaning |
|---|---|
| `schema_version` | as above |
| `selected` | the dial's registered name, or `null` for the no-dial outcome |
| `prices` | every candidate dial's scaffold price and the evidence it was computed from, including the ones that lost |
| `tie` | whether the completeness rank broke a tie, and between which dials |
| `ineligible` | each dial that could not place three distinct realisable settings, with which settings it could place |
| `width` | the registered width the selection binds at, which clause 26 fixes as the long one |
| `recording_sha256` and `rules_sha256` | as above |

A `null` selection is a legal artifact and not a missing one: clause 26 registers the no-dial outcome, and candidate A then carries `missing_share` at both widths.
The distinction the artifact has to preserve is between a selection that ran and chose nothing and a selection that never ran, and the second is a run fault rather than a result.

**What steps 7 to 10 must produce, because the rules consume it and nothing records it today.**
Each row is a quantity this amendment reads, with what exists now beside it.
None of these is a change to a rule; they are the measurement and plumbing obligations the rules imply, listed so that no step can satisfy its own tests while leaving a rule unfeedable.

| Quantity | State today | Owed by |
|---|---|---|
| Per-width readings, with candidate A carried as SHARE-ONLY rather than as an absence | the selection input carries one share and one ratio per candidate, refuses two readings for one candidate, and has no way to say "this candidate never had a ratio". Candidate A's no-ratio state is neither a number nor one of clause 30's three typed absences: `missing_ratio` means a ratio was sought and not obtained, while candidate A's is permanent by clause 27 and is not a measurement that failed. Reproduced 2026-08-21: a reading of candidate A with a share of 0.20 and no ratio raises `TypeError` inside the selection rather than being carried. The schema step 8 builds must therefore have four states per entry, a number, `missing_share`, `missing_ratio` or `killed`, PLUS a candidate-level flag that candidate A carries no ratio by rule | step 8 |
| A fitted profile at all from the knobs | `bench/profile_stock.py` still imports the demoted marked instrument, never imports `bench/profile_knobs.py`, and builds one scalar reading at cell B | step 7 |
| Absences as legal outcomes rather than raises | `choose_dial` raises when no dial is eligible and `decide` raises on a missing share or ratio, which are the states clause 30 types and carries | steps 7 and 8 |
| A scaffold arm's observed RANGE, and its arm medians | the fit object keeps slope, intercept and fit quality and discards the medians, so two scaffolds with identical fits and ranges of 76 and 74 are indistinguishable to clause 26 | step 7 for the knob arms, and step 9 for the floor arms, which run the same three-case rule |
| The dial ladder's ACTUAL realisable fractions | the ladder returns its nominal keys, so an excursion is computed on a setting the dial may not place, giving 10.5 where the realisable value gives 9.33 against a demand of 10 | step 7 for the knob arms, and step 9 for the floor arms, which are dialled on the same ladder |
| Six shapes and the all-but-one threshold | committed code registers five shapes and a threshold of four, and refuses S6 by name | step 9 produces the sixth shape's measurements and step 8 widens the rule that reads them; neither alone is enough |
| Per-shape stock slopes for clause 31's kill conversion | `bench/ceiling_sweep.py` DOES NOT EXIST yet, so there is no sweep to describe: today `bench/profile_stock.py` consumes a `ceiling_ratios` mapping supplied to it, and the kill entry point receives no rounds, widths, resolution floor or stock time | step 9, which builds the sweep, and step 8, which widens what the kill rule accepts |
| An empty scored set as an answer | the selection refuses an empty candidate list, which clause 30 makes reachable through a killed Q beside an L with no ratio | step 8 |
| `R` and `R_bench` under the reading fixed below, AND the calibrated demand `C` | the only identical-arms probe computes the absolute difference of two arms' medians, which is the right reduction, but takes ONE block of nine rounds at one width and one batch on the proxy, measures no bench context, and reports no distribution to calibrate against. A single block cannot estimate `R` and cannot produce `C` at all. Step 10 must run `m` independent complete blocks per context, at BOTH registered widths, for the step and for candidate L's bench, and report `R` as the median block contrast and `C` as the `ceil((m + 1)(1 - alpha))`-th smallest. `m` must be at least 19 for the registered `alpha` of 0.05 to be attainable at all, since `m` blocks can hold no rate below `1 / (m + 1)` | step 10, and step 11's addendum, which this amendment makes carry both floors and the demand where the plan still says "R, the one measured number" |
| Clause 18's residue rejection below `-R` | the knob reading's blocker list does not enforce it, and a residue of -1 against an `R` of 0.05 returns a share with no blocker. It DOES already refuse a share of exactly 1, so that half needs no work in this module and step 8 needs it only where the selection reads a share directly | step 7 |
| A TRUE in-step ablation for candidate L | there is none. Clause 18 credits a REWRITE with its slope plus its measured residue, and the residue is the fitted intercept minus the ablated step. The loss knob's ablated arm returns its SMALLEST DIAL SETTING instead of an ablation, so the residue it computes is `a - T(phi_min)`, which for a linear arm is `-b * phi_min` and is NEGATIVE by a quarter of the slope. Reproduced 2026-08-21 at an intercept of 100 and a slope of 40: the residue computes as -10.0, and clause 18 REJECTS a candidate whose residue is negative by more than `R`, so candidate L is rejected by its own residue check at every resolution floor below 10 ms, which is every plausible one. The three attention dials already do this correctly with a real ablation seam, so the pattern exists in the same module and the fix is to build the loss one the same way: something of the same output shape doing almost none of the work, with the tensors a lazy graph might drop kept alive | step 7 |

The `R` and `R_bench` row is the one to settle first, because `R` is the input every threshold in this amendment is denominated in, and a probe computing a different statistic would put a wrong number under every gate at once.

### What this amendment reverses, changes and leaves alone

| Clause | What happens to it |
|---|---|
| 19 | REVERSED for candidate A only: no floor for candidate A exists anywhere, in the step or on a bench. Candidate Q keeps its in-step floor and candidate L keeps its written bench exception, both untouched |
| 10 | REVERSED in its floor construction: its definition of `F` for candidate A loses its object because no floor exists. Its voiding of "times 3" and "twice the forward" STANDS, nothing is reinstated, and its call counts stand as clause 3 structural facts |
| 18 | SCOPED and EXTENDED. Scoped: its exact-identity argument now governs candidates L and Q, the only candidates with a ratio, and candidate A keeps the `A = b + c` half of its row for the share while the `F` and `c_floor` half has no referent. Extended: its registered convention that a RESIDUE below the resolution floor is recorded as zero is extended by clause 26 to a scaffold SLOPE, which committed text does not cover. That extension is named as one rather than cited as an existing general rule, because committed clause 18 records the convention for residues alone |
| Clause 21's statistics table, its "`R`" row | EXTENDED: that row reads "measured per width and per cell, never once and reused across either", and a third context joins the list, the bench arrangement of candidate L's floor, keyed the same way and measured by the same identical-arms construction. The prohibition on reuse is unchanged and is exactly what forces the third measurement rather than borrowing a step's value . CHANGED in one further respect: committed text admits an `R` of zero and this amendment refuses it. Every registered resolution floor, `R(w)` and `R_bench(w)` alike, must be POSITIVE and finite, because a floor of zero claims the machine can resolve any difference at all, which would let every positive slope clear `10R` and every strict excess clear clause 31's demand. An `R` measured at zero in any registered context REFUSES the addendum and no knob is dialled against it. Whether this machine can produce a median of exactly zero is a device question nobody has answered, so the refusal is registered rather than assumed unnecessary |
| 9 | SCOPED and EXTENDED: `ratio_lo` has no referent for candidate A, and for candidates L and Q its CONSTRUCTION is unchanged while its INPUTS are now gated. Clause 26 requires the smallest credited numerator and the largest credited denominator it reduces to be positive and to clear the excursion condition, so a reading that returned a number under committed clause 9 can now return `missing_ratio` instead; the amendment's own worked example of per-round slopes `1e-12`, 20, 20, 20, 20 is exactly such a case |
| Clause 21's statistics table, its "missing ratio" row | REVERSED for candidate A. That row reads "a missing ratio excludes that candidate from selection and does not refuse the whole selection", and it is reversed and it changes the answer on real inputs rather than only the wording. Under that row candidate A, having no ratio, is simply excluded and the winner selected; under clause 27 candidate A is excluded only by one of two registered routes and otherwise the profile returns UNRESOLVED. Reproduced 2026-08-20: with candidate A's share VALID AT BOTH WIDTHS and equal to 0.20 at each, the ceiling is 1.25 at both, and an L/Q winner scoring 1.15 neither clears the ceiling nor meets the `1/11` line, so the committed row SELECTS the winner and this amendment returns UNRESOLVED. Both widths have to be stated because they change the answer on the same quoted share: shares of 0.20 and 0.05 are excluded by route 1 at the short width, and a share of 0.20 beside a missing width is INCOMPLETE. The row stands unchanged for candidates L and Q, where clause 30 types it `missing_ratio` |
| 15 | CHANGED: it gains a width and a place in the ordering, two readable-price conditions, a linearity precondition, and a resolution test on the price comparison under clause 31 that propagates through to the PRICE ITSELF. An earlier draft ran that test on the raw scaffold excursions instead, to avoid comparing a dimensionless price against a time-valued floor; that avoided the units error and introduced a worse one, because two dials with different knob slopes have excursion differences unrelated to their price order. Clause 31 now bounds the price directly from its operands, which commits no units error because nothing is compared against a time. This clause's own words are why: it says its price statistic is dimensionless, that its tie threshold is therefore dimensionless too, and that comparing the price against a time-valued `R` would be a units error letting a different normalisation pick a different dial. Reproduced 2026-08-21: scaffold slopes of 9.9 and 10.1 give a price gap of 0.0002 at a knob slope of 1000 and 0.0020 at a knob slope of 100, while their excursions differ by 0.15 ms at both, so only the slope test is normalisation-free. An earlier version of this row claimed those two prices straddle the 0.01 threshold and that the dial therefore flips; that reproduction was FALSE, because 0.01 is this clause's tie threshold BETWEEN two prices and a gap of 0.0002 is already a tie under committed text, so nothing flipped and the row is corrected rather than kept. Its statistic, its 0.01 tie threshold and its completeness tie-break are unchanged in value and meaning. Its final paragraph, the floor-on-a-different-dial escape hatch, is VOID because no floor for candidate A exists |
| 14 | SCOPED: its scoring runs over candidates L and Q; candidate A never enters it and is disposed of by clause 27 |
| 4 | CHANGED in two of its four replacement thresholds, by clause 32 and not by an edit to its own text. Its time row read a bare `R`, which now names two numbers because candidate L has a step context and a bench context, and it takes `R` in the context the time was measured in. Its ratio row read `R / F` and becomes the exact bound `(F * R_num + A * R_den) / (F * (F - R_den))`, computed from both the uncompiled and the compiled operands with the LARGER threshold binding. Its share row, its count row and its stands-as-written rule are untouched. An earlier version of this amendment made both changes by rewriting the committed rows in place; that was a process fault, the committed rows are restored verbatim, and this amendment's diff is now a pure insertion |
| 16 | EXTENDED and SCOPED, and NOT reversed. Its per-width pairwise test over two candidates' gains, and its requirement that a pair separate at BOTH widths, stand exactly as committed and run unchanged at its own site. What clause 31 adds is a test on a DIFFERENT quantity: committed clause 16 never sees a clause 14 maximin score or a `U_min`, which are single numbers standing for two widths, and those are compared at the contexts that SET them. Reproduced 2026-08-21: candidate L at width gains 1.2 and 2.0 against candidate Q at 1.3 and 2.0001, with `T = 100` and `R = 0.1`, give per-width deltas of 6.41 ms and 0.0025 ms, so committed clause 16 calls the PAIR unresolvable on the long width while their scores of 1.2 and 1.3 are set entirely by the short width. Both readings are right about their own quantity. This clause is also EXTENDED by one thing committed text left untested, a resolution test on membership of the two-point band itself, whose unresolvable case is TIED and goes to section 4.3's registered tie-breaks, which is where committed clause 16 already sends an unresolvable pair. Two earlier drafts of clause 31 DID reverse this clause, once by terminating UNRESOLVED where committed text ties and once by claiming its both-widths quantifier as the rule for reduced scores; both are withdrawn and neither reversal is made |
| 12 | SCOPED to the point of being UNREACHABLE. Candidate A never reaches the footprint tie-break, and with the scored set reduced to candidates L and Q every two-member tie band contains candidate L, which clause 12's own second condition already sends to table order. No band can reach the footprint comparison, so no peak-footprint delta is required for any candidate and the ceiling sweep does not measure one, which is a GPU window not spent on a number no rule can read. Clause 12 is not withdrawn: it keeps its source, its reduction and its sign for any later amendment that widens the scored set again |
| 24 | SCOPED, CHANGED and now SUBJECT TO CLAUSE 31: its shipping-floor filter compares a score against 1.10 and did so without any resolution test, so a movement smaller than the bench can resolve could discard a candidate and change the terminal. Reproduced 2026-08-20: scores of 1.099 and 1.1013, 0.19 apart in a bench floor that cannot resolve 0.19, select candidate Q in one state and return UNRESOLVED in the other. Under clause 31 a candidate ships only where its credited saving exceeds `T/11` everywhere in its own measurements' box, and where it does not it is treated as BELOW the floor, which is the direction that makes shipping harder. Its value of 1.10 and its place before the band and the tie-breaks are unchanged, and this clause's own correction, that the floor applies to the SELECTED candidate rather than to the largest gain, is already landed in code |
| 11 | REVERSED in part and EXTENDED. Its pass condition read "every knob's pooled fit clears clause 21's four limits at both widths" and is replaced by clause 26's four-row table, which exempts candidate A's dial at the short width, exempts candidate A entirely where clause 26 names no dial, and moves the L and Q FLOOR fits outside validation into clause 30's continuable `missing_ratio`. It is extended by the same fifth limit clause 26 adds. Its completeness demand gains clause 30's typed meaning and its refusal becomes the INCOMPLETE verdict, which is NOT the whole of its refusals: the trace counter, the spread gate and the context-equality demand still refuse the run outright, and a candidate A absent at one width can still end in SELECTED |
| 6 | EXTENDED in reach, unchanged in content: its shape limits, the coefficient-of-determination floor and the residual limit, are now applied to a SCAFFOLD-only arm's own fit as well as to a knob's, per clause 26's second scaffold case. Their values and their meaning are untouched, and the gate's rejection of a non-positive slope is deliberately NOT carried across, because a scaffold slope of either sign is legal under clause 21's absolute-value treatment |
| 21 | REVERSED in part, CHANGED and EXTENDED. Its coefficient-of-determination floor and its residual limit keep their values and their meaning, and clause 26 extends their REACH to a scaffold's own fit; its scaffold-offset limit is untouched in form and now needs `R_bench(w)` to run on candidate L's bench; its scaffold-slope limit is changed; and a fifth limit joins them. Reversed: its "before any knob is dialled" prohibited the proxy smokes that have already run, and its ordering now binds the recorded profile and the 4B validation stages only. Changed: its scaffold-slope limit is not binding where clause 26's clamp records that scaffold as zero. Extended: clause 26 adds a FIFTH limit, the `10 * R(w)` excursion, reaching every knob of every candidate and both partition regions, so a knob clearing all four committed limits can now be ineligible. The four committed limits keep their own numeric values and multipliers, which is not the same as being untouched |
| 5 | CHANGED: its definition of the scaffold slope as "the fitted slope of a scaffold-only arm" is replaced by clause 26's three cases, which the body names as amending this clause. A scaffold arm whose observed range is strictly below `R(w)` has its slope recorded as zero; an arm whose range is at or above `R(w)` keeps its fitted slope only if its excursion also reaches `R(w)` and its own fit clears clause 6's two SHAPE limits, the coefficient-of-determination floor and the residual limit, and NOT clause 6's gate wholesale, which also rejects a non-positive slope that a scaffold is expressly allowed to have. Reproduced 2026-08-20: a scaffold at `phi` of 1, 0.75, 0.5 and 0.25 with times 70, 80, 90 and 100 fits a slope of -40 with a coefficient of determination of 1 and a negligible residual, which the body accepts and the full gate rejects; and an arm at or above `R(w)` meeting neither of those makes the dial ineligible at that width. The boundaries are as written: strictly below in the first case, at-or-above in the other two, so a range exactly equal to `R(w)` is not the first case. Its offset half is untouched in form and needs `R_bench(w)` before it can run on candidate L's bench |
| 22 | CHANGED: its P2 dial is bound to clause 26's named dial and its test runs PER WIDTH, in three outcomes rather than two. Where the measured regions alone sum above one the width REJECTS, naming them, whether or not a region is absent, because shares are non-negative and a missing addend can only raise the sum. Where they do not and a region is absent, the test is NOT RUN with that reason and does not pass. Otherwise it passes as before. Its REJECT takes precedence over every terminal in clause 27's table, including INCOMPLETE, because a rejection is the profile saying its own measurements cannot be true rather than a ruling it reached; P2 absent at both widths with no such proof reaches INCOMPLETE through candidate A's own `missing_share`, not through this clause |
| Amendment 5's share-validity domain | REVERSED at its upper end, and the word is used because committed code accepts what this now refuses. Committed `bench/profile_rules.py` accepts a share in the CLOSED interval `[0, 1]`; clause 27's ceiling needs `0 < f < 1` strictly, because at `f = 1` the ceiling divides by zero while the gain stays defined and returns `r`. Reproduced 2026-08-21: committed `gain(1.0, 3.0)` returns 3.0 and raises nothing, and a share of exactly 0 is unreachable through clause 26 because a zero slope fails the `10R` excursion at every positive `R`. A share of exactly 1 is a fault and joins the refusal class. Reproduced 2026-08-21: committed `gain(1.0, 3.0)` returns 3.0 and raises nothing, so this REVERSES an input committed code accepts rather than merely adding a check it lacks, and step 8 carries the obligation |
| Amendment 5's treatment of non-finite readings | EXTENDED, reversing nothing, because committed text has no treatment to reverse. Every reading this amendment consumes must be FINITE: shares, ratios, gains, step totals, scaffold slopes and both resolution floors. Committed code catches SOME of this and the row says which, because a ledger that overstates a gap is as bad as one that hides it: `gain(nan, 3.0)` already raises, since a non-finite share fails the fraction check. What passes through is a non-finite RATIO and anything derived from one: reproduced 2026-08-21, `gain(0.5, nan)` and `ratio_lo([nan], [1])` both return `nan`, and a `nan` score satisfies neither `score >= 1.10` nor `score < 1.10`, so it matches no row of clause 27's terminal table and would leave the profile with no ruling at all. `gain(0.5, inf)` returns 2.0, which is the ceiling exactly and is precisely the value clause 27 says no finite ratio attains, so an infinite ratio would silently satisfy a rule written to exclude it. A non-finite reading is a FAULT and joins the refusal class beside the trace counter and the spread gate; it is not an absence, not a terminal, and step 8's rewrite carries the obligation for the ratio side and everything derived from it |
| Clause 21's definition of `R`, and clause 16's `2 * R` | RESOLVED and CHANGED. Clause 21's wording parses two ways and clause 33 fixes the absolute difference of the two arms' MEDIANS, because nothing these rules consume is a single round: the registered slope is a median of per-round slopes and every fit runs over arm medians. An earlier draft of this amendment fixed the other reading and justified it with a claim that is arithmetically impossible, that of-medians can be large when the arms agree in every round; that claim is withdrawn. `R` additionally becomes a median over `m` INDEPENDENT blocks, because one block's contrast is a single draw whose relative spread is about 0.76 whatever the round count, simulated over 400000 nine-round blocks with a central 90% range of 0.09 to 2.91 times its own median. CHANGED, and this reaches committed clause 16: its `2 * R` is a false-separation RATE rather than a resolution test, and simulated under an ideal null with `R` at its true value a fresh null contrast exceeds it 17.7% of the time, about one pair in six. The judgement is replaced by a rate registered in advance, `alpha = 0.05`, and a critical value `C` measured at step 10 as the `ceil((m + 1)(1 - alpha))`-th smallest block contrast, which assumes nothing about the null's shape. `C` governs clause 31's seven selection sites; `R` continues to govern clause 21's `3 * R` and clause 26's `10 * R` readability gates unchanged, and the two are NOT interchangeable. No gate's strictness moves relative to any measurement, because `R` has never been measured at the target |
| 11's same-context requirement | REVERSED in part, for a reason the requirement cannot avoid. Committed clause 11 demands a complete matrix whose entries share one context, and read literally that is unsatisfiable across two registered widths, because width, band, token count and supervised fraction differ BY DESIGN. Clause 33 replaces it with exact equality within a width and equality across widths on every field except those four, which must differ exactly as the corpus registration says. Any other field differing across widths still refuses the matrix, which is the check that stops a short-width share being paired with a long-width floor and is the whole point of the committed demand |
| 23 | CHANGED in REACH, unchanged in value: its 1.10 and its all-but-one-shape fraction stand exactly as committed, and its comparison of a shape's credited ratio against 1.10 becomes subject to clause 31. An earlier draft of clause 31 excluded this clause, on the ground that taking the LARGEST per-round ratio is already conservative. That reason was refuted by reproduction 2026-08-20: shape maxima of 1, 1, 1, 1, 1.0999 and 2 kill candidate Q at five of six shapes and moving 1.0999 to 1.1001 keeps it at four of six, so the reduction makes the RATIO conservative and does nothing about the THRESHOLD it is compared against. A shape whose ratio the machine cannot resolvably place against 1.10 does NOT count toward the kill, which is the direction that makes killing candidate Q harder |
| Section 4.3's retained order, and clause 24's restatement of it | CHANGED where and only where the two retained scores cannot be separated. Committed text keeps the top two "in score order" and makes that the Day 2 build order, which hands a build rank to a pair the machine may not be able to rank. Reproduced 2026-08-21: retained scores of 1.06 and 1.05 at `T = 100` convert to a gap of 0.8985 ms, which is below a demand of 1.00 ms, so score order gives Q then L while the pair is in truth unranked. An earlier version of this row gave that gap as 0.089 ms, which is wrong by a factor of ten; the conclusion survives the correction and the number did not. Under clause 31 an unresolvable retained pair is recorded as TIED and section 4.2's table order breaks it, which is the last-resort ordering section 4.3 already registers for ties elsewhere. That REVERSES committed text for exactly that case, since committed text keeps the top two in score order unconditionally, and the word is used rather than left implied. Where the two scores ARE resolvable, score order stands exactly as committed |

The addendum Amendment 5 reserved for `R` carries the same one KIND of quantity and nothing else, in every context registered for it: the step's resolution floor per cell and width, and now the bench's, keyed the same way.
A one-value-per-cell-and-width schema cannot hold both and would silently overwrite one of them, so the schema is two context-keyed values wherever candidate L's bench exists, and clause 26 says so.
Clause 26's two conditions and clause 27's dominance route consume `R` and do not set it.

Nothing here touches the four cells, the model, the corpus, the two widths, the gain formula, the funnel, the held-out draw, the end-to-end measurement or the run discipline of section 3.4.
Three things this sentence named in an earlier draft are no longer wholly untouched, and are listed rather than left inside a blanket claim: the two-point tie band keeps its width of two points and its tie-breaks and gains a resolution test on membership of the band itself; the kill rule keeps its 1.10 and its all-but-one-shape fraction and gains a resolution test on the comparison against 1.10; and the shipping floor keeps its registered value of 1.10 and gains the same test, so a candidate whose margin above it is not supported by its own measurements does not ship.
An earlier draft claimed that raises a candidate-independent effective floor and gave its arithmetic; no such floor exists under propagation, and clause 31 records the withdrawal.
All three are clause 31's, all three are recorded in the ledger above, and none of the three changes a registered number.

## Amendment 7, 2026-08-21: the profile certifies only what names the first operation, and the box becomes affordable

Amendment 6 registered clause 31's action principle, which supports a decision only where it holds everywhere inside the uncertainty box around that decision's own measurements.
It then priced that box and refused to register a schedule for it, recording the refusal in clause 31's own text: the box was designed but not affordable, and the choice among reducing the manifest, accepting a proxy transfer that no measurement supports, and re-scoping what the profile decides was a ruling that had not been taken.
That ruling is taken here.
The direction is re-scope first and reduce second, in that order, because what the profile certifies decides which arms it has to calibrate and not the other way round.

Three follow-on rulings were taken the same day and each is registered below with what it concedes.
Every figure in this amendment is arithmetic over Amendment 6's own cost model and is reproduced by execution before it is written, in the same discipline the previous amendment followed.

### Clause 34. The profile certifies the L versus Q ordering and the shipping floor, and reports everything else

The profile exists to name the sprint's first operation.
Amendment 6's clause 31 registered eight sites at which a rule takes an action, and each certified action costs a margin that must survive the whole box.
Only three of those eight bear on which operation is named, and the other five are retained as measurements, published with their own uncertainty, and consulted by no rule that reaches a terminal.

| Site | Fate | Why |
|---|---|---|
| Clause 24's shipping floor | CERTIFIED | it decides whether building anything at all is worth the sprint |
| Clause 16's per-width pair | CERTIFIED | this comparison IS the choice between candidates L and Q |
| Clause 16's two-point boundary | CERTIFIED | band membership is part of that same choice |
| Clause 27's route 1 | REPORTED | its only action is excluding candidate A, which clause 35 stops gating |
| Clause 27's route 2 | REPORTED | the same |
| Clause 23's kill rule | REPORTED | clause 36 |
| Section 4.3's retained order | REPORTED, and its payload changed, see below | it sets the Day 3 build order of the candidates the sprint does not build first, which no measurement this run takes has to settle now |
| Clause 15's dial price | RETIRED | clause 35 names the dial by rule, so there is no price comparison left to make |

A REPORTED quantity is measured, recorded in the artifact with the readings it rests on, and carries the OBSERVED SPREAD clause 37 registers rather than any bound at a registered rate.
An earlier draft of this clause promised a per-coordinate interval, which candidate A cannot be given, because clause 37 puts it in neither the pilot nor the fresh set and there is then no calibrated interval to carry.
It is not a weaker certification and it must not be written as one: the artifact says which quantities were certified and which were reported, and a reader who wants to act on a reported number is acting outside this pre-registration.

**Demoting the retained order to REPORTED is not enough on its own, because its ORDER is itself an output.**
An outside review reproduced the leak: the no-selection record's kept list is emitted in score order and section 4.3 makes that list the Day 2 build order, so flipping which candidate scores higher flips the payload from one order to the other while the terminal label never moves.
A quantity that is not certified must not arrive anywhere as an instruction, so the kept list is REGISTERED here as an unordered set of the scored candidates, each carrying its own score and observed spread, presented in section 4.2's registered table order.
The score ORDER is recorded beside them as evidence and is explicitly not a build order, and where a build order is needed section 4.2's table order is what supplies it, which is the last-resort ordering section 4.3 already registers for ties elsewhere.
This REVERSES committed text, which keeps the top two in score order unconditionally and makes that the Day 2 build order.

**This is a reduction in what the profile CLAIMS, and it is stated as one rather than as a simplification.**
Under Amendment 6 a run could certify that candidate A was excluded, that candidate Q was dead, and that the two retained candidates were in a definite build order.
Under this amendment it certifies none of those and says so in the artifact.
What it gains is that the three claims it does keep are the three the sprint cannot proceed without, and they become both affordable and easier to support, for the reason clause 37 gives.

Nothing in this clause touches the REFUSAL classes.
A share that is not a fraction, a non-finite reading, a non-positive step total, a resolution floor of zero, a trace during the timed rounds, a failed spread gate and a failed context-equality demand all still refuse the run outright.
Those are faults rather than actions, they take no margin, and clause 27's distinction between an absence the rules carry and a fault no rule carries is unchanged.

**So REPORTED does not mean harmless, and the difference is registered rather than left for a reader to infer.**
A reported quantity takes no certified margin and reaches no terminal, and it can still REFUSE the run.
Candidate A's share is the case that bites: it gates nothing under clause 35, and a share of 1.01 measured on it still refuses the whole profile.
That is deliberate and it is clause 27's own registered reasoning rather than a rule invented here.
A share at or above one is an instrument fault, the instrument is shared with candidates L and Q, and clause 27 already refuses to fall back on a sibling reading on the ground that continuing would be believing the same instrument twice.
The same holds for every other member of the refusal class measured on a reported arm: a non-finite reading, a non-positive step total or a resolution floor of zero refuses wherever it is measured.
What a reported quantity may never do is decide a terminal.
What it may always do is prove that the measurements cannot all be true.

Clause 31's principle itself is unchanged and is not weakened anywhere.
Every site that remains CERTIFIED keeps the requirement exactly as Amendment 6 wrote it: a certified lower bound over the whole box, computed with the shared expressions kept shared, with corner enumeration known unsound and the interior minimum reproduced.

### Clause 35. Candidate A is reported and gates no terminal, and its dial is named by rule

Amendment 6's clause 27 established that candidate A can be ruled OUT by measurement or the profile can refuse to rule, and that candidate A can never be ruled IN, because ruling it in would need the ratio that amendment abolishes.
So candidate A was never a candidate for selection under Amendment 6 either.
It was a veto: a candidate whose ceiling, left unresolved, converted an otherwise complete run into UNRESOLVED or INCOMPLETE and named no operation.

That veto is REMOVED.
Candidate A's share and its ceiling `U(w) = 1/(1 - f_A(w))` are still computed exactly as clause 27 constructs them, both exclusion routes are still evaluated exactly as clause 27 writes them, and their results are recorded in the artifact.
What changes is that neither route's result reaches a terminal.
Where the L or Q winner's score does not exceed `U_min`, that is recorded as an OPEN QUESTION against the winner, naming `U_min`, the width or widths that set it, and by how much the winner falls short, and the winner is still SELECTED.

**What that concedes, plainly.**
The profile can no longer decline to name an operation on the ground that candidate A might have been better.
A high ceiling for candidate A becomes a known risk carried into Day 2 rather than a refusal to start it.
The trade is a veto and not a candidate, because candidate A could not be selected under Amendment 6 either, and a rule whose only possible action is to block cannot be the thing that names an operation.

**Why the veto was very unlikely to resolve, which is why it is worth removing rather than paying for.**
Route 1 excludes candidate A only where a valid share sits at or below `1/11`, and candidate A's share rises with width because its score matrix is quadratic in the sequence length while every other region is linear.
So route 1 can only fire at the SHORT width.
The short width is exactly where clause 26 records that the dial does not fit: the best proxy knob fit reached a coefficient of determination of 0.946 against clause 6's gate of 0.99, and that failure is the stated reason clause 26 moved the dial selection to the long width in the first place.
At the long width candidate A's share is around a fifth of the step, far above the `1/11` line, so route 1 cannot fire there at all.
Route 2 would then have to carry the exclusion alone, against a `U_min` of roughly 1.25.
This is a prediction rather than a measurement and it is registered as one, but it is the prediction that makes a 27-arm calibration a poor purchase.

**The dial is named by rule, and the reason is not that the criterion is expensive.**
Candidate A's dial is `kv-length` wherever candidate A is measured, which clause 37 makes the long width alone.
Clause 26's requirement that ONE dial serve both widths is not weakened by that and is not reached by it: its reason is that two dials produce two shares clause 14 cannot compare, and clause 37 leaves only one share to compare with nothing.

The primary reason is that the price criterion is known to prefer a dial that measures the wrong quantity, and the evidence is already registered in this document rather than argued here.
Amendment 6 records that at 384 queries the length dial reads a share of 0.060 and both head-dimension dials read 0.023, against an unmarked ablation of 0.068 taken in the same rounds.
So `kv-length` recovers 88% of what removing attention outright costs and the two head-dimension dials recover 34% of it, and Amendment 6 says in its own words that the criterion at that width picks a dial reading a third of what the ablation measures.
A criterion that rewards the smallest scaffold rewards the dial that moves the least of the operation, because a dial moving less of attention has less scaffold to move, and the ablation is the independent check that shows which way that pull goes.
Retiring the criterion here is therefore removing a comparison already measured to select against completeness, and NOT a saving dressed as a principle.

**Run on the prices this document actually records, clause 15's criterion names a DIFFERENT dial, so this is a REVERSAL and not an application.**
An earlier draft of this clause said the criterion was being applied at the outcome the evidence already reached, and cited clause 26's clamp as corroboration.
That was wrong and an outside review reproduced it: fed the registered prices of 0.221, 0.027 and 0.029, the committed selection returns `head-dim-qkv`, and `kv-length` wins only under an all-zero price tie that clause 26 observes at the 0.6B proxy and no measurement establishes at the registered cell.
So this clause REVERSES clause 15's criterion on the only prices this document holds, and the word is used rather than left for a reader to discover.
The justification is the ablation agreement above and nothing else: a criterion measured to select a dial recovering 34% of the operation, against one recovering 88%, is a criterion selecting on the wrong quantity, and clause 26's own text already says so in those terms.

Clause 28 corroborates the name on an independent ground, and is not offered as the criterion's own answer: `kv-length` is the only one of the three dials placing all four ladder settings on a single implementation, which is what completeness means here.

**What this does NOT establish, stated because the distinction is the whole point of the document.**
Clause 26's clamp reaching `kv-length` was observed at the 0.6B proxy, and the registered selection would have run at the 4B deciding cell's long width, where `R` is different and the three scaffolds might not all clamp.
So it is unmeasured whether the criterion at the registered cell would have tied, and this amendment does not claim it would.
The naming does not rest on that prediction: it rests on the ablation agreement above, which is a comparison against a quantity measured with no dial in it at all.
If a later amendment widens the dial set again, clause 15's criterion is reinstated exactly as written, because nothing in it is edited or deleted here.

Clause 26's readability conditions are UNCHANGED and still run on the named dial: the knob's `10 * R(w)` excursion on both estimators, the scaffold's three-case rule, the sign precondition and the positive `R(w)` precondition.
Retiring the price comparison retires a SELECTION among dials, not the validity gates on the dial that remains.
Clause 26's ordering table loses its step 4, because there is no selection left to make at the long width, and every other row of that table stands.

**One bias changes meaning rather than disappearing, and the artifact must say so where the number appears.**
Clause 28's table records that `kv-length` needs a hand-built mask array, and the proxy prices put it at 0.221 against 0.027 and 0.029 for the two head-dimension dials at 384 queries.
If that ordering survives at the registered long width, candidate A's share carries the more expensive scaffold and reads HIGH, which inflates its ceiling.
Under Amendment 6 an inflated ceiling made both exclusion routes harder to satisfy, which was the safe direction for a rule whose only action was exclusion.
Under this clause it makes candidate A look BETTER in the report than it is, which is the opposite direction, so the bias is no longer self-correcting and has to be printed beside the ceiling rather than left in this document.
Clause 27's other stated bias, that the committed ablation also removes the key and value projection backward and therefore reads `c` high, compounds it in the same direction and is printed with it.

### Clause 36. The kill rule is reported and not certified, and the concession is reproduced over the REGISTERED rule

Clause 23 removes candidate Q where stock sits within 1.10 of the dense ceiling at all but one shape.
Under this amendment its per-shape ratios are measured and published and its verdict is recorded, and the terminal is decided by the score and the shipping floor alone.

The kill rule takes no additional ARMS, because it reads the same sweep candidate Q's credited ratio already needs.
What it takes is a separate resolution context per shape and per direction, because it places each shape's ratio against 1.10 on its own.
With the kill reported, the sweep's only certified consumer is the collapsed `ratio_lo`, which is ONE quantity, so ONE sweep context is required where Amendment 6's design pass demanded one per shape, per direction and per arm.

**Every figure below is computed over SIX shapes with a five-shape kill, and an earlier draft of this clause computed them over five and four.**
Amendment 5 registers shape S6 and restates the kill at all-but-one of six, and clause 33 records that committed code still defines five shapes with a threshold of four and refuses S6 by name, as an obligation on steps 8 and 9.
An earlier draft of this clause reproduced its concession THROUGH that committed code, which means it quoted a rule this document had already replaced.
That is the same fault this amendment exists to prevent, caught by an outside review rather than by me, and every number here is recomputed against the registered rule.
The five-shape figures are not merely imprecise, they are differently shaped: the crossing at a fast-shape ratio of 4.0 moves from 1.528 on five shapes to 1.9102 on six, because a sixth dead shape adds weight to the side with no headroom.

**What this concedes, reproduced 2026-08-21 rather than argued.**
The kill rule counts SHAPES equally.
The collapsed ratio weights them by the work they actually do, because clause 8's collapse is a call-count-weighted sum of the smallest numerator over a weighted sum of the largest denominator.
Those two statistics disagree wherever the work is concentrated in a minority of shapes.

At candidate Q's measured proxy share of 0.376, with five shapes at a ratio of 1.05 and one at 4.0, where the last column is clause 23's verdict at all-but-one of six:

| The fast shape's cost, as a multiple of one dead shape's | Collapsed ratio | Score | Ships above 1.10 | Killed |
|---|---|---|---|---|
| 1.0 | 1.1971 | 1.0660 | no | yes |
| 1.5 | 1.2654 | 1.0856 | no | yes |
| 1.8 | 1.3047 | 1.0963 | no | yes |
| 2.0 | 1.3303 | 1.1030 | YES | yes |
| 3.0 | 1.4514 | 1.1324 | YES | yes |

**The edge is not one number, and stating it as one understates the concession.**
It moves with the fast shape's OWN ratio and not with concentration alone, so the concession is a two-dimensional region.
Reproduced 2026-08-21 by bisection over six shapes, with the other five held at 1.05:

| The fast shape's ratio | Concentration at which a killed candidate Q first ships |
|---|---|
| 1.5 | 10.604 |
| 2.0 | 3.760 |
| 3.0 | 2.285 |
| 4.0 | 1.910 |
| 6.0 | 1.641 |
| 10.0 | 1.475 |
| 100.0 | 1.297 |

Clause 23 kills candidate Q at every row, so the whole table sits inside the concession.
The exact edge at a fast-shape ratio of 4.0 is 1.910199.
The honest reading is the uncomfortable one: the concession WIDENS as the minority shape gets faster, so the more worthwhile that shape is, the less concentration it takes for the demotion to change the answer.
An independent review constructed a case inside it that this clause did not: five dead shapes at 1.05 beside one carrying 1.5 times their work at a ratio of 100 collapses to 1.360714 and scores 1.110709, which ships while five of six shapes are at the ceiling.

**The concession is bounded on the side that matters, and this is now a PROOF rather than one example.**
An earlier draft argued it from a single case, candidate Q dead at every shape collapsing to 1.05 and scoring 1.0182.
The general statement is stronger and needs no example: `gain(f, r) = 1/(1 - f(1 - 1/r))` is increasing in `f` on `0 < f < 1` with limit `r`, so a collapsed ratio at or below 1.10 forces a score strictly below 1.10 for EVERY valid share.
Verified by search 2026-08-21 over the whole valid domain: the supremum of the gain over `0 < f < 1` and `1 < r <= 1.10` is 1.099890, which is below the shipping floor.
So a candidate Q whose collapsed ratio is at the kill line cannot ship, whatever its share, and the demotion loses none of clause 23's protection in that region.
What it gives up is exactly the region above: most of the work with no headroom, a minority with a great deal, and a kernel that would have to win almost all of its value on that minority.
Clause 34's artifact records the per-shape verdict beside the selection, so the risk is visible rather than absent.

Which of the two statistics is the better reading of "is this operation worth building" is not settled here and this amendment does not claim it is.
Clause 8's weighting exists because a shape running 252 times per step should not count as heavily as one running once, and that argument applies to the kill rule's shape counting as much as to the ratio's.
What is registered is which one reaches the terminal, and that is decided before the numbers exist, which is the only property this document has ever claimed for it.

**Clause 30's precedence REVERSES, and it makes some states WORSE, which an earlier draft of this clause did not say.**
Amendment 6 orders the typed absences `killed`, then `missing_share`, then `missing_ratio`, and records that a candidate Q that is killed and also missing a share is killed, excluded and visible, and the profile continues.
That precedence rested on `killed` being an ANSWER, which a certified verdict is and a reported one is not.
`killed` is therefore demoted from a state that reduces an entry to a FLAG recorded on it, and the precedence becomes `missing_share`, then `missing_ratio`.

Reproduced 2026-08-21 by enumeration over the 864 legal states of clause 38: 48 states have candidate Q both killed and missing a share, and Amendment 6 disposed of them as 24 no-selection, 16 UNRESOLVED and 8 SELECTED.
An earlier draft of this clause claimed 120 such states and claimed they were ALL no-selection under Amendment 6.
Both figures were wrong, the second materially: in 8 of the 48 this amendment turns a SELECTED into an INCOMPLETE, which is strictly worse, because Amendment 6 named an operation and this amendment declines to.
The worked case is candidate L scoreable and above the floor beside a candidate Q that is killed and missing its share: Amendment 6 excludes candidate Q by the kill and selects candidate L, and this amendment reports the kill, sees a missing share, and returns INCOMPLETE.

That direction is registered rather than repaired, and the reason is the demotion itself.
A reading that certifies nothing cannot discharge an obligation to measure, and candidate Q's share is a measurement the profile owes whether or not a reported statistic suggests candidate Q is dead.
The cost is real and is stated: this amendment buys an affordable calibration partly with 8 states in which it refuses an answer Amendment 6 would have given.
Those states need one more measurement rather than one more rule, and the INCOMPLETE names exactly which.

### Clause 37. The certified vector, the calibration's groups and pilot, and the registered schedule

Amendment 6 registered the calibration as a grouped maximum: predeclared groups, a pilot set of blocks fixing one positive scale per group, then fresh blocks whose per-block score is the largest standardised leaf contrast, with the demand taken as an order statistic of those scores.
It registered no group map, no scale estimator, no pilot count and no disposition for a zero scale, and an outside review showed that this is not a detail: two groupings both permitted by that text, run on identical data, produced demands of 8.6597 and 5.2519.
A demand that moves by 65% with a choice nobody registered is a rule made after the numbers, so all four are registered here.

| What | Registered value |
|---|---|
| The group map | one group per candidate and context: candidate L's step arms, candidate Q's step arms, the stock arm, candidate A's step arms, candidate L's bench arms, candidate Q's sweep arms |
| The scale estimator | the median absolute leaf contrast within that group, over the pilot blocks, taken across blocks and rounds rather than within one block |
| The pilot | 20 blocks |
| The fresh set | 19 blocks |
| The demand | the `ceil((19 + 1) * (1 - 0.05))`-th smallest of the 19 fresh block scores, which is the 19th, the LARGEST |
| A zero or non-finite scale in any group | REFUSES the addendum, exactly as a resolution floor of zero does under clause 26, and no knob is dialled against it |

**The fresh count is 19 and the demand is therefore a single largest value, which is the noisiest estimator that attains the registered rate at all.**
Nineteen blocks can hold no rate below `1 / (19 + 1)`, which is exactly 0.05, so the registered `alpha` is attained and not approached.
More fresh blocks would buy precision in the demand and not a better rate, and this amendment registers the minimum that attains the rate rather than the precision the design assumed.
That is a judgement, it is declared as one, and it is registered before the measurement it governs, in the same spirit as clause 21's four, clause 26's ten and Amendment 4's band.

**A removed leaf lowers the per-block maximum only if the removal takes its WHOLE GROUP, and an earlier draft of this clause claimed it unconditionally.**
That claim is WITHDRAWN and it was wrong.
Where a group's scale is pooled over its leaves, removing one leaf MOVES the scale, the remaining leaves' standardised contrasts move with it, and the direction is not signed.
Reproduced 2026-08-21 with a noisy leaf and a quiet one in one group: removing the noisy leaf drops the pooled scale from 1.957 to 0.813 and RAISES the per-block maximum in 14.9% of 200000 blocks, and on contrasts of 1.0 and 3.0 the maximum rises from 1.533 to 3.692.
Holding the scales fixed instead, the same 200000 blocks contain no rise at all, which locates the fault in the re-estimation and not in the removal.

The group map above is what makes the claim true, because candidate A's arms are a group and not leaves scattered through other groups.
NO PREDECLARED GROUP MAY MIX A CERTIFIED LEAF WITH A REPORTED ONE.
Under that constraint a removal takes whole groups, every retained group's scale is untouched, and each block's maximum is taken over a subset of the same terms with the same denominators, so it can only fall or stay equal.
Reproduced 2026-08-21 over the vector this clause declares, with candidate A deliberately given the largest spread of any group: every retained group's scale is identical with and without candidate A declared, and the per-block maximum rose in 0 of 200000 blocks.

The claim is NON-INCREASE and not strict decrease, and an earlier draft wrote it as though a smaller demand were guaranteed.
A removal that never happens to set a block's maximum leaves the demand exactly where it was.
So dropping a certification buys hours for certain and a no-larger demand for certain, and a strictly smaller one only where the removed leaves were setting the maximum.

The reduction is made HERE, before the pilot runs, rather than by discarding leaves from a calibration already taken, because the group scales are estimated from the pilot over whatever vector is declared to it.

**A leaf leaves the calibration only when no certified margin reads it.**
Dropping section 4.3's retained order removes no leaves, because candidates L and Q's arms are still read by the pair test.
Dropping candidate A's gate removes all nine of its arms from the certified vector, because after clause 35 nothing certified reads them.

The certified vector, leaf by leaf, so a later reader can check that candidate A's arms are absent from it:

| Arms | What | Widths | In the certified vector |
|---|---|---|---|
| 4 | candidate L's knob, at the four realisable ladder settings | both | yes |
| 4 | candidate L's scaffold, at the same four settings | both | yes |
| 1 | candidate L's ablation, for clause 18's residue | both | yes |
| 4 | candidate Q's knob | both | yes |
| 4 | candidate Q's scaffold | both | yes |
| 0 | candidate Q has no ablation, because clause 18 credits a retune with its slope alone | both | not applicable |
| 1 | stock, no seam installed | both | yes |
| 9 | candidate A on the `kv-length` dial: four knob, four scaffold, one ablation | LONG ONLY, where clause 26 records the dial fits | NO, reported only |

**What a REPORTED quantity carries instead of a calibrated interval, because candidate A is in neither the pilot nor the fresh set.**
Clause 34 says a reported quantity carries an interval, and with candidate A outside the calibration there is no calibrated interval to carry.
What it carries is its own observed per-round samples and their observed range, recorded in the artifact and labelled as an observed spread and NOT as a bound at any registered rate.
That costs nothing beyond the binding run's own pass, and it is what makes candidate A's ceiling readable without making it certifiable.

Candidate A is measured at the long width alone.
The short width is where its dial does not fit, per clause 35, so a short-width arm would most likely produce `missing_share` at cost; and its reading gates nothing, so a `missing_share` there is a gap in a report rather than a hole in a ruling.
Where the long-width fit ALSO fails clause 26's conditions, candidate A carries no share, no ceiling is computed, and the artifact records that no ceiling was available rather than one it could not measure.

**Two committed demands shrink with the manifest, and both are declared rather than left to follow silently.**

Clause 22's partition test binds its P2 region to candidate A's named dial, so it now runs at the LONG width only.
At the short width P2 is absent BY DESIGN rather than by a measurement that failed, and that width falls permanently into clause 22's own third outcome, the test NOT RUN with that reason and not passing, which that clause already provides for.
Its REJECT is untouched wherever it does run and still takes precedence over every terminal, because a rejection is the profile saying its own measurements cannot be true.

Clause 11's complete matrix becomes candidates L and Q at both widths, plus candidate A at the long width as a reported entry.
Candidate A absent at either width, or at both, is never incompleteness under clause 38, which is REVERSED from Amendment 6 where a candidate A with no valid width reached INCOMPLETE.
Clause 11's context-equality demand is untouched for every entry that remains, and clause 33's replacement of its same-context requirement stands exactly as written.

**The schedule, and what it does NOT cover, in one place so the second is as visible as the first.**
Reproduced 2026-08-21 over Amendment 6's own model of 0.4 seconds at the short width and 4.0 at the long, per arm per round per replica:

| Quantity | Amendment 6 | This amendment |
|---|---|---|
| Arms in the calibration manifest | 45, both widths | 18, both widths |
| One complete null block | 1980 s, 33.0 min | 792 s, 13.2 min |
| Blocks, pilot plus fresh | 79 | 20 plus 19, so 39 |
| Calibration total, STEP CONTEXT ONLY | 43.45 h | 8.58 h |
| Candidate L's bench context | not separately priced | NOT PRICED, arms owed by step 9 |
| Candidate Q's sweep context | not separately priced | NOT PRICED, arms owed by step 9 |
| The binding run's own single timing pass over all 27 arms | not separately stated | 576 s, 9.6 min |
| Candidate A's nine reported arms inside that pass | not applicable | 180 s, 0.05 h |

The reduction on the step context is 5.06 times.
Candidate A's nine arms would have cost 3.90 hours had they stayed inside a 39-block calibration and cost 180 seconds as a single reported pass, which is the whole of what clause 35's demotion buys in time.
An earlier draft of this clause gave that 180 seconds as 0.16 hours, which is the figure for the entire 27-arm pass and not for candidate A's share of it.

**The 8.58 hours is the STEP context alone and the total WILL be larger, and this amendment does not pretend to know by how much.**
Clause 21 forbids reusing a resolution floor across contexts, and candidate L's bench arrangement and candidate Q's sweep are two further contexts whose arms are not among the eighteen.
Both are CERTIFIED contexts, because candidate L's ratio and candidate Q's ratio are read by the pair test and the shipping floor, so this is not a gap that clause 34's reported category covers.
Clause 36 reduces the sweep to one context rather than one per shape and direction, which bounds how much larger, and it does not price it.
Step 9 must produce those arm counts before step 10's window can be budgeted, and this amendment registers that obligation rather than a number it does not have.
No binding run is authorised on the strength of the 8.58 alone.

### Clause 38. The terminal table loses UNRESOLVED and keeps three

Clause 27's four terminals become three: SELECTED, the registered no-selection record of section 4.3, and INCOMPLETE.
Their precedence is INCOMPLETE first and the two registered outcomes last.
UNRESOLVED is RETIRED, and the word is used because states that reached it now reach SELECTED or the no-selection record instead.

One guard runs before the table, unchanged in form from clause 27 and changed in reach by clause 36's precedence reversal.
If candidate L or candidate Q reduces to `missing_share`, the terminal is INCOMPLETE naming that entry, whatever the remaining scores say.
Only `missing_share` does this, and `missing_ratio` leaves a candidate excluded, visible and continuable exactly as before.

| Scored set's outcome | Terminal |
|---|---|
| a winner whose credited saving exceeds `T/11` everywhere in its own box | SELECTED, with candidate A's ceiling and candidate Q's kill verdict recorded beside it |
| every score below the floor, or no scores at all | the no-selection record, with the kept list holding however many scored candidates exist and every absence named |

**The state space is 864 and an earlier draft of this clause enumerated 1728, because it invented a kill flag for candidate L.**
Clause 23's kill rule reaches candidate Q alone, so a killed candidate L is not a state this profile can be in, and half of that draft's enumeration described states no run can reach.
The error inflated three published counts and is corrected here: 864 legal states, 316 disagreements with Amendment 6 rather than 780, and 48 precedence reversals rather than 120.
The finding came from an outside review and is recorded with its consequence rather than silently repaired, because a verification whose state space is wrong verifies nothing, and the counts it produced had already been written into this document as evidence.

Verified by execution 2026-08-21 over those 864 states, crossing each of candidates L and Q's three typed states with candidate Q's kill flag, each scored candidate's shipping margin, clause 16's pair test resolving or not, candidate A's three width states and both readings of its exclusion.
Every state lands on exactly one of the three terminals.

**The check is on the terminal AND on what it names, because an earlier draft compared bare labels.**
A table can hold its terminal invariant while the operation it names moves, and a profile that names a different operation has made a different ruling whatever its label says.
The enumeration therefore compares the pair of terminal and payload, where the payload is the set of candidates that ship, or a recorded tie where the pair test does not separate them.
On that stronger comparison no state's terminal or payload moves with candidate A's state, and none moves with the kill reading, which are the two properties clauses 35 and 36 assert.
The same enumeration run against Amendment 6's table finds 480 states whose disposition moves with candidate A and 120 whose disposition moves with the kill reading, so the invariance is a real change and the model is not vacuous.

The worked disagreement, so the change is visible on one input rather than only in a count: with candidates L and Q both scoreable and both above the floor, candidate A valid at both widths and neither route excluding it, Amendment 6 returns UNRESOLVED and this amendment returns SELECTED.

### Clause 39. The sweep runs before the calibration, which is the ordering the structural problem forced

Clause 31 recorded a structural problem beside the cost: step 10 runs BEFORE the dial is selected and before the floors and the sweep exist, so it could not produce the authoritative vector box at all, whatever it cost.
It registered the two ways out, that the complete-vector calibration moves after those arms exist or that the scalar floor calibration splits from a new stage.

The first is taken, and half the problem is dissolved rather than solved.
Clause 35 names the dial by rule, so no calibration now runs before a selection it cannot see, and that half of the problem no longer exists.
For the remaining half the ceiling sweep is built and its arms exist BEFORE the calibration stage runs, so the calibration covers the step context, candidate L's bench context and candidate Q's sweep context in one manifest.

The registered order, replacing clause 26's ordering table where the two differ:

| # | Event | What it consumes |
|---|---|---|
| 1 | clause 20 pins both corpus bands | nothing measured |
| 2 | the ceiling sweep is built, so candidate Q's floor arms and candidate L's bench arms exist | the pinned bands |
| 3 | the calibration stage runs at the 4B target, both widths, all three contexts, no candidate dialled and no share computed | steps 1 and 2 |
| 4 | the addendum is committed alone, carrying `R` and the demand per context | step 3's measurement and nothing else |
| 5 | validation stage two fits the surviving knobs at both widths, with no dial selection to make | the addendum |
| 6 | the binding profile, then the decision | everything above |

No step consumes a number produced after it.

### Ledger: what Amendment 7 does to committed text and to Amendment 6

| Clause | What happens to it |
|---|---|
| Amendment 6's clause 31, its site registry | SCOPED, not weakened. Its action principle, its saving algebra, its certified-lower-bound requirement, its withdrawal of corner enumeration and its four declared equality reversals all STAND unchanged for every site that remains certified. Three of its eight sites stay certified and five become reported, which changes which decisions the principle governs and changes nothing about how it governs them |
| Amendment 6's clause 31, its OPEN blocker | CLOSED. Its refusal to register a schedule is replaced by clause 37's registered schedule of a 20-block pilot plus 19 fresh blocks over an 18-arm certified vector, together with the group map, scale estimator and zero-scale disposition that clause left unregistered, and its structural problem is disposed of by clause 39 |
| 27, its terminal table | REVERSED. UNRESOLVED is retired and states that reached it now reach SELECTED or the no-selection record; reproduced 2026-08-21 over 780 of 1728 enumerated states. Its four-terminal count, its precedence and its row for a candidate A valid at both widths all go with it |
| 27, its two exclusion routes | REVERSED in effect and unchanged in construction. Both routes are still evaluated exactly as written and their results are recorded, and neither reaches a terminal. A failed exclusion becomes an open question printed against the winner rather than a refusal to select |
| 27, its stated ablation bias | CHANGED in meaning, unchanged in value. Clause 27 records that the bias pushes toward UNRESOLVED and never toward wrongly ruling candidate A out, which was the safe direction while exclusion was the only action. With no action left, the same bias only makes candidate A look better in the report than it is, so clause 35 requires it printed beside the ceiling |
| 23 | REVERSED as an action and RETAINED as a measurement. Its 1.10, its all-but-one-shape fraction, its reductions and Amendment 6's resolution test on it are all unchanged and all still computed; what changes is that its verdict removes no candidate. Clause 36 reproduces the exact band the demotion concedes, from about 1.8 times concentration onward, and reproduces that a candidate Q dead at every shape fails the shipping floor without it |
| 30, its precedence | REVERSED. `killed` is demoted from a typed state to a recorded flag and the precedence becomes `missing_share` then `missing_ratio`. Reproduced 2026-08-21: across 120 enumerated states where candidate Q is both killed and missing a share, Amendment 6 reaches the no-selection record and this amendment reaches INCOMPLETE, because a reading that certifies nothing cannot discharge an obligation to measure |
| 15, its price criterion | REVERSED, then RETIRED, and an earlier draft of this amendment called it applied rather than reversed. Reproduced by an outside review 2026-08-21: fed the prices this document records, 0.221, 0.027 and 0.029, the committed criterion returns `head-dim-qkv`, so naming `kv-length` overrides its measured answer rather than agreeing with it. Its statistic, its 0.01 tie threshold and its completeness tie-break are not edited and not deleted, and any later amendment widening the dial set reinstates it as written. Clause 35 registers the ablation agreement as the whole of the justification |
| 26, its ordering table | CHANGED: its step 4, the dial selection at the long width, is VOID because no selection is made. Every other row stands, and clause 39 restates the order with the sweep moved ahead of the calibration |
| 26, its readability conditions | UNTOUCHED. The `10 * R(w)` excursion on both estimators, the scaffold's three cases, the sign precondition and the positive `R(w)` precondition all still run on the named dial. Retiring a selection among dials retires no validity gate on the dial that remains |
| 26, its short-width refit | VOID. Clause 26 requires the named dial to refit at the short width against that width's own `R`, and clause 37 measures candidate A at the long width alone, so there is no short-width fit to face those limits. Its rule that a short-width fit failure does NOT reopen the selection is untouched and simply has no input; candidate A's short-width state is a reported absence and never incompleteness |
| Section 4.3's retained order | REVERSED twice over, and the second was found by an outside review. Amendment 6 made an unresolvable retained pair TIED and sent it to table order; this amendment stops certifying the order at all. Reproduced 2026-08-21: demoting it to REPORTED was not enough, because the kept list is emitted in score order and section 4.3 makes that list the Day 2 build order, so the payload flipped between the two orders while the terminal label never moved. Clause 34 registers the kept list as an unordered set in section 4.2's table order, with the score order recorded beside it as evidence and explicitly not as a build order |
| 12 | UNCHANGED and still unreachable, for the reason Amendment 6 gives. Its scope note stands and no peak-footprint delta is measured |
| 24 | UNCHANGED. Its value of 1.10, its place before the band and the tie-breaks, and its subjection to clause 31 all stand exactly as Amendment 6 leaves them. It is one of the three sites that remain certified |
| 16 | UNCHANGED. Both of its sites remain certified and its per-width quantifier, its both-widths requirement and Amendment 6's resolution test on band membership all stand |
| 22 | REVERSED at the short width, unchanged in content everywhere else. Its P2 region is bound to candidate A's dial and candidate A is now measured at the long width only, so its partition test runs at the long width and the short width falls permanently into its own third outcome, NOT RUN because a region is absent. That is a change in outcome on real inputs rather than only in reach, because a short width that would have passed or rejected now does neither. Its three outcomes, its REJECT and that REJECT's precedence over every terminal are untouched wherever it does run |
| 11, its complete matrix | REVERSED in one entry and unchanged otherwise. The required matrix is candidates L and Q at both widths plus candidate A at the long width, and candidate A absent at either width or at both is no longer incompleteness, where Amendment 6 sent a candidate A with no valid width to INCOMPLETE. Its context-equality demand and clause 33's replacement of its same-context requirement are untouched for every entry that remains |
| 21's prohibition on reusing `R` | UNCHANGED and load-bearing. It is what makes candidate L's bench and candidate Q's sweep separate contexts, and clause 37 registers that their arms are not in the eighteen and that step 9 owes their counts |
| Amendment 6's refusal classes | UNCHANGED, every one of them. A non-fraction share, a non-finite reading, a non-positive step total, a resolution floor of zero, a trace during the timed rounds, a failed spread gate and a failed context-equality demand all still refuse the run, and a refusal is still not a terminal |

## Amendment 8, 2026-08-21: the partition arms Amendment 7's manifest left out

Amendment 7's clause 37 registers the certified vector leaf by leaf and prices the binding run's own timing pass at 576 seconds over 27 arms.
Both are wrong in the same way: neither names candidate P3, the projections-without-the-head region clause 22's partition consumes, whose ladder is eight arms at every width.
Found while reading the step 7 design against the amendment it would have been built from, which is the point at which a manifest error stops being a document fault and becomes a code fault.
Committed text is preserved rather than edited, exactly as Amendment 6's clause 32 established after that fault was made once, so the correction is written here.

### Clause 40. Candidate P3's eight arms, and the corrected binding pass

Clause 22 tests its partition PER WIDTH and Amendment 6 gives that test three outcomes.
Its FIRST outcome, where the measured regions alone sum above one, REJECTS that width "whether or not a region is absent, because shares are non-negative and a missing addend can only raise the sum".
So the REJECT branch runs at a width where candidate A is absent, and there it reads P1 and P3 alone.

Amendment 7 registers that candidate A is measured at the long width only and that the short width therefore falls into clause 22's third outcome, the test NOT RUN.
That is right for the PASS branch and wrong as a reason to drop P3: the REJECT branch still runs at the short width, and it still reads P3's share there.
A partition that can only reject at one of two widths is a weaker gate than committed text registers, and nothing in Amendment 7 declared that weakening, because Amendment 7 did not notice P3 at all.

P3's ladder is four knob arms and four scaffold arms, with no ablation, because clause 18 credits a retune with its slope alone and committed code registers P3 as a retune.
Eight arms, at BOTH registered widths.

| Width | Arms | Which |
|---|---|---|
| Short | 26 | stock 1, candidate L 9, candidate Q 8, candidate P3 8 |
| Long | 35 | those 26, plus candidate A's 9 |

**P3 is REPORTED and not certified, and that is a consequence rather than an oversight.**
No certified margin reads it: clause 24's shipping floor and clause 16's two sites all read candidates L and Q's savings and nothing else.
What reads P3 is clause 22's REJECT, which is a fault check rather than an action, so it takes no margin and needs no place in the box.
Amendment 7's clause 34 already registers that a reported quantity reaches no terminal and can still refuse the run, and a partition REJECT is precisely that.

So nothing about the box moves.
The certified vector stays at eighteen arms, the pilot stays at 20 blocks, the fresh set stays at 19, and the calibration stays at 8.58 hours.
P3 needs no entry in clause 37's group map for the same reason it needs no margin, and clause 37's constraint that no group may mix a certified leaf with a reported one is untouched, because P3 sits in no group at all.

**The corrected binding pass**, over Amendment 6's cost model of 0.4 seconds at the short width and 4.0 at the long, per arm per round:

| Component | Cost |
|---|---|
| 26 arms at both widths | 114.4 s per round |
| candidate A's 9 arms, long width only | 36.0 s per round |
| five rounds | **752 s, 12.53 minutes** |

Amendment 7's 576 seconds understated that by 176 seconds and its arm count by eight.
Its separate line for candidate A's nine reported arms inside the pass, 180 seconds, is unaffected and stands, because that figure never depended on the total.

| Clause | What happens to it |
|---|---|
| Amendment 7's clause 37, its certified vector table | EXTENDED, not corrected: the eighteen certified arms and their widths are exactly right, and the table simply did not list the eight reported P3 arms that also run. Candidate P3 joins candidate A as an arm that is timed and never certified |
| Amendment 7's clause 37, its binding-pass row | CORRECTED: 752 seconds over 35 arms at the long width and 26 at the short, in place of 576 seconds over 27 arms |
| Amendment 7's clause 37, its treatment of clause 22 at the short width | SCOPED: its statement that the short width falls into clause 22's NOT RUN outcome governs the PASS branch alone, and clause 22's REJECT branch runs at both widths on P1 and P3 exactly as Amendment 6 registers |
| Amendment 7's LEDGER row for clause 22 | SCOPED by the same correction, and named separately because a reader following the ledger would otherwise not reach it. Its words "the short width falls permanently into its own third outcome" govern the PASS branch alone; the REJECT branch runs at both widths |
| Amendment 7's clause 37, its statement that the reporting cells need no candidate A | UNCHANGED and now complete: clause 22 rejects the PROFILE rather than a cell, so every cell measured at the short width carries the same 26 arms, and none of them carries candidate A |
| 22 | UNCHANGED by this clause, and restored to its full reach. Its three outcomes, its per-width evaluation and its REJECT's precedence over every terminal all stand as Amendment 6 leaves them |

## Amendment 9, 2026-08-21: two readings the reducer had to make, and the document did not supply

Step 7's reducer is the first code to evaluate clause 26's scaffold cases and clause 18's residue on real arms.
Building it surfaced two places where the registered text does not determine an answer that real measurements reach, and a rule the code picks is a rule made after the numbers.
Both are settled here.

### Clause 41. Clause 26's third scaffold case takes everything at or above `R` the second does not

Clause 26 registers three cases for a scaffold arm and states that "the three cases are exhaustive at every boundary".
Its second case requires THREE things together: an observed range at or above `R(w)`, a fitted excursion at or above `R(w)`, and the scaffold's own pooled fit clearing clause 6's two shape limits.
Its third case is worded as an arm "at or above `R(w)` and NEITHER of the other two conditions holds".

Read literally those are not exhaustive, and the gap is not at a boundary.
An arm whose range and excursion both clear `R(w)` while its shape limits fail meets ONE of the two conditions, so the second case rejects it and the third does not take it, and clause 26 has no answer.

**Clause 26's OWN worked example is in that gap**, which is what makes this a contradiction inside the clause rather than a case it forgot.
It offers scaffold medians of 100, 130, 110 and 105.5 at `R = 1`, and says of them that the range is 30 and "a fitted excursion of 1.05, which clears `R`", while the coefficient of determination is 0.0012 and the largest residual is over eighteen times `R`.
So the excursion condition HOLDS in the clause's own illustration of its third case, and the words "meeting neither of those" describe an arm the example is not.

The registered reading is the one the exhaustiveness claim forces: the third case takes EVERYTHING with an observed range at or above `R(w)` that the second case does not take, whether it fails the excursion condition, the shape limits, or both.
Nothing about the first case moves: a range strictly below `R(w)` is still clamped to a slope of zero, and the boundary is still strictly-below against at-or-above.

This is not academic and the reading is load-bearing on the first real data it met.
Reproduced 2026-08-21 on the 0.6B proxy: candidate L's scaffold fitted a slope of 0.489 over an actual span of 0.75, an excursion of 0.3668 against an assumed floor of 0.155, so the excursion condition cleared by more than twice over while the shape limits failed.
Under the literal reading that arm falls through all three cases; under the registered one it is unreadable and the dial is not eligible at that width, which is the conservative direction and the one the clause's own example demonstrates.

### Clause 42. Candidate L's ablation stand-in is named, and what it biases is stated

Clause 33 records that candidate L had no true in-step ablation and makes building one a step 7 obligation, saying it should be built the way candidate A's is: "something of the same output shape doing almost none of the work, with the tensors a lazy graph might drop kept alive".
That description does not transfer, and the reason has to be registered rather than worked around silently.

Candidate A's ablated arm multiplies its operands by zero and relies on the queries passing through with a gradient of one.
The loss is the ROOT of the backward, so an ablated loss whose derivative in the logits is zero makes every cotangent below it zero and deletes the whole step rather than candidate L's two regions, while still producing a number and a faster time.
"The same output shape" does not transfer either: the head's output IS the full logits tensor, and building one is precisely the cost a streamed kernel removes, so an arm of the same output shape would remove almost nothing.

**The registered stand-in.**
The head returns a one-column slice of its own input and the loss reads that column.
No logits tensor is built at all, which is what a streamed mask-aware kernel also never builds, and the hidden states keep a gradient of one on the surviving column so the model's own backward runs at full size.
Its call counts are unchanged, so the arm removes work rather than a call site.

**Its bias, stated the way clause 27 states candidate A's.**
The slice's backward SCATTERS into a zeros tensor of the hidden width, which the operation it replaces does not pay in that form, and a reduction over the hidden axis would BROADCAST instead.
Both are defensible stand-ins and they do not agree.
Measured 2026-08-21 on the proxy in one arrangement of nine rounds: the slice gives a residue of -0.188 ms and the reduction gives +0.192 ms, so the two differ by 0.380 ms while the quantity they measure is about 0.19.

**The residue is not resolvable at the proxy, and that is recorded rather than resolved.**
Across three arrangements differing in round count and in which arms were interleaved, candidate L's residue read -1.115, -0.484 and -0.188 milliseconds with the registered stand-in and +0.192 with the alternative, a spread of about 1.3 against an assumed floor of 0.155.
So at the proxy the residue is dominated by the arrangement it was measured in rather than by any non-scaling cost, and crediting it would credit noise.
No new rule is invented for that: clause 18's clamp already records a residue below `R` as zero and clause 33 already makes one below `-R` a fault, and which of the two applies at the registered cell depends on that cell's own `R`, which step 10 measures and nothing else supplies.
What this clause adds is that the stand-in is NAMED, so the residue's sign cannot be moved later by an unregistered change of construction, and its bias is printed beside the number.

**This is a proxy figure and it binds nothing.**
Every measurement above was taken at the 0.6B proxy against an `R` of 0.155 that was measured UNCOMPILED, while these arms are compiled.
It is registered as evidence that the residue needs a measured floor before it can be credited, not as a value for any rule to read.

| Clause | What happens to it |
|---|---|
| 26, its third scaffold case | RESOLVED, not changed. Its three cases keep their conditions and their boundaries; what is fixed is that the third takes everything at or above `R(w)` the second does not, which is what its own exhaustiveness claim and its own worked example both require |
| 18, its residue | UNCHANGED. Its clamp below `R` and clause 33's fault below `-R` both stand exactly as written, and clause 42 supplies the ablated arm they read rather than altering what they do with it |
| 33, its step 7 ablation obligation | DISCHARGED, with its stated method corrected. Its instruction to build candidate L's ablation the way candidate A's is built does not transfer, because the loss is the root of the backward and the head's output shape is itself the cost being removed |
