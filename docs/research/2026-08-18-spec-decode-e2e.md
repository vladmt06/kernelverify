# End-to-end speculative decode: pre-registration

Status: pre-registered 2026-08-18.
Sections 1 through 8 fix the method and the outcomes before the harness is written and before any measurement exists.
Sections 9 and 10 remain placeholders until the binding run has completed.

## 1. The question

For one user on this machine, is MLX's own speculative decoding with our kernel routed during verification faster than the plain stock decode that user has today?
If it is, how much of the gain comes from speculation alone, and how much comes from the kernel on top of speculation?

This is an end-to-end question about `mlx_lm` generation rather than another projection-level or fixed-step kernel measurement.
The K = 6 meeting-point spike licensed this measurement by showing that routing the seven-row verification step can help, but it did not measure drafting cost, acceptance, or end-to-end throughput.

## 2. The five arms

Every arm runs through `mlx_lm.generate.stream_generate` with the same prompt, greedy sampling from `make_sampler(temp=0.0)`, and 128 generated tokens.
Both draft models run stock in every speculative arm.

- **Arm 0, plain stock:** the stock 3-bit target runs plain decode with no draft model, and this is the baseline against which every product claim is measured.
  The 2026-08-18 confirmation re-run of the earlier A/B measured its B = 1 arm 2 at 64.40 tokens per second per stream, but this experiment remeasures arm 0 in the same rounds as the speculative arms instead of importing that number.
- **Arm 1, ours:** the patch is installed on the 3-bit target in fused mode, with the selected stock draft and `num_draft_tokens = K`.
- **Arm 2, speculative stock 3-bit:** the stock 3-bit target uses the same draft and the same K, and this arm is both the canary and the attribution reference for the kernel.
- **Arm 3, speculative stock 4-bit:** the stock 4-bit target uses the same draft and the same K, and this is the quality-matched alternative that mirrors the earlier A/B's arm 3.
- **Arm 4, forced-stock control:** the patch is installed on the 3-bit target in forced-stock mode with the same draft and the same K, so eligibility is evaluated but discarded and the arm measures the interception's own cost on this path.

Arms 1, 2, 3, and 4 are interleaved in that order within every round.
Arm 0 is sampled once per round beside those four arms.

## 3. Fixed parameters

| Parameter | Registered value |
|---|---|
| Draft-token grid | `K_GRID = (2, 4, 6, 8, 10)` |
| Flattened verification widths | `M = K + 1`, giving `M = (3, 5, 7, 9, 11)` |
| Rounds | `ROUNDS = 5` |
| Generated tokens | `GEN_TOKENS = 128` |
| Prompt length | `PROMPT_T = 64` |
| Sampling | Greedy, with `temp = 0.0` |
| Draft 1 | `qwen3-0.6b-4bit-g64` |
| Draft 2 | `qwen3-1.7b-4bit-g64` |
| Process footprint budget | 24.0 decimal GB |

The prompt is the one produced by `serve_sub4bit.make_prompts` from its fixed prompt seed, truncated to 64 target-tokenizer tokens, and the same prompt is used by every arm.
K = 2 gives M = 3 and K = 10 gives M = 11, so both lie outside the routed window and are built-in null cells where arm 1 must equal arm 2 within spread.
The harness calls `mx.clear_cache()` before every cell's memory guard reads the process footprint.

## 4. What keeps the interception honest

### Exact dispatch accounting

The expected routed-call count is derived per intercepted site rather than by multiplying a whole-model wrapper count by a pass count.
For each observed flattened verification width M, the harness derives `routed_sites_at(M) = sum(1 for cell in patch.site_cells if should_dispatch(M, *cell))`.
For the observed verification passes with flattened widths `M_p`, the exact expected count is `sum(routed_sites_at(M_p) for M_p in observed_verification_passes)`.
Arm 1's observed routed-call count must equal that sum exactly, and arm 4's observed routed-call count must be zero.
A mismatch raises a typed error, stops the run, and names the draft and K cell.

The count uses every observed verification-pass shape rather than assuming every pass has width K + 1.
The last pass can be shorter when `max_tokens` runs out, and a shorter pass can legitimately fall outside the routed window.

### Counting the target rather than every model

`Model.__call__` is a class dunder shared by the target and the other Qwen3 models, so an instance attribute cannot intercept these calls.
The counter wraps `type(target).__call__` once at class level, forwards every call, and records the input shape and wall time only when `self is target`.
The original class method is restored in `finally`, including when an arm raises.

Every round must have zero hard fallbacks.
Registered routing-table declines and arm 4's forced-stock decisions are not hard fallbacks, but any other fallback invalidates the round.
A hard-fallback-invalid round is ineligible for every outcome.

### Amendment, 2026-08-18: what the counter actually sees, and what it can and cannot measure

Found by the dispatched writer refusing a second time, and reproduced here against `mlx_lm` 0.31.3's source before either point was admitted.
Sections 4 and 5 described a seam that does not exist, and the host-side tests written before the harness encoded the same fiction, so a harness that satisfied those tests would have produced two wrong quantities on the real machine and no test could have failed.

**The target is called with token identifiers, not activations.**
`speculative_generate_step` verifies with `logits = model(y[None], cache=cache)` where `y` holds the K + 1 candidate tokens, so `Model.__call__` receives an integer array of shape `(1, K + 1)` with no feature dimension at all; the embedding happens inside the model, below the seam the counter wraps.
Section 4 said "flattened width" and the rules module computed it as `prod(shape[:-1])`, which is the right rule for the activation a projection sees and gives 1 for every pass at this seam.
From this amendment the width of an observed pass is `prod(shape)` over the token-identifier array, so a `(1, 7)` verification pass is width 7 and the 64-token prefill, which `_prefill` presents as `(1, 63)`, is width 63 and routes nothing.
`expected_routed_calls` now refuses any shape that is not rank 2, because the only seam this experiment counts is that one and a rank-3 shape reaching it means the assumption moved.

**Wall time taken around that call measures graph construction, not execution.**
MLX is lazy: `model(y[None], cache=cache)` returns as soon as the graph is built, and `speculative_generate_step` does not force evaluation until `mx.eval(tokens, draft_tokens)` after the draft graph is built too.
A timer around the class dunder therefore measures Python and graph-building cost, which is neither proportional to the target's work nor a share of anything.
Left as registered it would have understated `verify_share` by roughly an order of magnitude, driven every `ceiling_pct` below its noise floor, and made every O2 cell read `not-a-decider` - a run that answers nothing while exiting 0.

From this amendment the timed rounds record passes and shapes only, and the counter takes no time in them.
Each `(draft, K)` cell instead runs ONE probe generation outside the timed rounds, in which the counter forces `mx.eval` on the target's output and records the elapsed per pass.
The probe runs ARM 2's configuration, which is the stock 3-bit target with the cell's draft and K and no patch installed, because the share this section defines is arm 2's and the ceiling asks how much of the UNROUTED baseline's time routing could reach.
Probing arm 1 instead would measure the already-routed pass, understate the baseline share by about the gain being tested, and make the ceiling depend on the result it is supposed to bound.
The cell's share is `verify_share = median_probe_pass_seconds * verify_passes / generation_time`, where the per-pass cost is the median over the probe's calls whose recorded width is exactly K + 1, and `verify_passes` and `generation_time` come from that cell's O2-eligible timed arm 2 rounds.
The width filter is K + 1 rather than "in the routed window": K = 2 and K = 10 have no in-window pass at all, and a share that could not be computed there would stop the grid at its first cell over a quantity those cells do not use, since their registered gain is already zero.
The probe's own `generation_tps` is discarded and enters no outcome, because forcing a synchronisation inside the generation loop changes the thing being timed.
Forcing evaluation is confined to the probe for exactly that reason: a synchronisation applied to every arm would perturb `generation_tps` in all five, and O3 quotes an absolute throughput a user would see rather than a ratio, so a uniform perturbation would not cancel there.

The bias this leaves is stated rather than hidden.
The probe measures the target's forward in isolation, while in the timed rounds that forward can in principle overlap other work, so the per-pass cost is an upper bound and `verify_share` and `ceiling_pct` are therefore upper bounds too.
The direction matters: an overstated ceiling can call a cell a decider when its true expected gain sits below the floor, which weakens the safeguard rather than manufacturing a win, since the ceiling enters no verdict's numerator or denominator and only decides whether a cell is read at all.

Measured rather than read, 2026-08-18, on the pinned 3-bit target with the 0.6B draft at K = 6 and 12 generated tokens, through a wiring probe that took no lock and made no claim.
The target and every draft are the same class, `mlx_lm.models.qwen3.Model`, which is why the counter must be class-level and must filter on `self is target`.
The target was called seven times at shapes `(1, 63)`, `(1, 7)` five times, and `(1, 6)`, every one of them rank 2, so the prefill is the single width-63 call this amendment predicts and the verification passes are width K + 1.
The response object had no `generation_time` attribute, and `generation_tokens / generation_tps` gave 0.3029 s against a 0.7440 s wall clock that also carried prefill and the first token, which is the gap the derivation exists to exclude.

The last pass was width 6, not width 7, and 6 is INSIDE the routed window.
That is the section's "the last pass can be shorter" case occurring on the first probe, and it lands somewhere that routes rather than somewhere that does not, so counting by observed shape is load-bearing here and not a precaution: a count that assumed every pass was K + 1 wide would have been wrong by one whole pass of routed sites in this very run.

The rest of the accounting was verified the same way, on the pinned 3-bit target with the 0.6B draft at K = 6 and 48 generated tokens, before the harness existed.
The patch wraps 252 sites and the routing table routes all 252 at width 7; the observed widths were one pass of 63, seventeen of 7, and one of 2; `expected_routed_calls` therefore derived 4284, and the interception's own counter had recorded exactly 4284, with no hard fallbacks.
This is the first time the flattened-width rule of 2026-08-18 has been exercised through `mlx_lm`'s own speculative path rather than through a shape test.

Token identity was checked across all four comparable arms in the same probe and every one of them matched to the token: arm 4 equals arm 2, so the control that invalidates the run does not fire; arm 1 equals arm 2, so `kernel-diverged` does not fire and O2 and O3 will have eligible rounds; and arm 2 equals arm 0, so `mlx-m-dependent` does not fire either.
None of that is a result and none of it is binding, because the machine was not idle and nothing was timed.
It says only that the run can produce a reading rather than a refusal, which is what an hour of GPU time is worth checking for in advance.

A full non-binding dry run of the finished harness was then taken on 2026-08-18, outside the detached runner and therefore quotable for nothing, purely to establish that the grid completes.
It completed all ten cells at exit 0 with no stop, no invalid cell and a clean closing idle sample, every `finish_reason` came back `length` so no arm was cut short by an end-of-sequence token, and not one identity label fired across fifty arm-cells, so neither our kernel nor MLX's own M-dependence flipped a token anywhere.
Its K = 10 cell on the 1.7B draft read `NULL-uncontrolled` with 1260 routed calls, which is 252 sites across five rounds of one short in-window pass, and is the case the amendment above was written for.

One prediction is recorded here before the binding run rather than after it.
That dry run's per-arm spreads had a median of 1.16% and a worst case of 9.52%, against the binding A/B's recorded noise floors of 0.05% to 0.10%.
The ceilings this experiment computes are roughly `verify_share` times 15 to 16 percent, so floors of that size would put several cells below their ceiling and make them read `not-a-decider`.
The expectation is that the detached run's floors come out far lower, because the dry run shared the machine with an interactive session while the binding run holds the machine alone, and because the A/B reached 0.05% on the same hardware through the same detached path.
If the floors do NOT fall, `not-a-decider` is the pre-registered honest answer for those cells and is recorded as such; it is not a licence to widen a threshold after seeing the numbers.

### Amendment, 2026-08-18: which calls are verification passes, and when a null cell stops being one

Two more gaps, one raised by the dispatched writer refusing a third time and one found here by measuring the null cells rather than reasoning about them.

**Which target calls count as verification passes.**
Nothing above said how the prefill is separated from the passes, and every quantity that divides by `verify_passes` depends on the answer.
From this amendment a verification pass is a target call whose width is at most K + 1, and any wider call is prefill.
With the registered `PROMPT_T = 64` the prefill is a single call of width 63, and K + 1 never exceeds 11, so the two classes cannot overlap.
The harness asserts that separation rather than trusting it: exactly one call per generation may be wider than K + 1, and its width must be `PROMPT_T - 1`, or the cell is invalid and the run stops.
A shorter-than-K+1 final pass stays a verification pass, which is the case section 4 already required the count to survive.

**A null cell whose short final pass lands in the routed window.**
O2 registered K = 2 and K = 10 as null controls that "must have zero routed calls", and any other reading invalidated the run.
Measured 2026-08-18 on the pinned 3-bit target at `GEN_TOKENS = 128` with the registered prompt: the 0.6B draft gives K = 2 widths `{3: 57, 2: 1}` and K = 10 widths `{11: 33, 2: 1}`, both entirely outside the window, but the 1.7B draft gives K = 10 widths `{11: 31, 6: 1, 2: 1}`.
That width-6 pass is inside the routed window 5..9, so arm 1 routes there, the observed count is not zero, and the rule as written would have declared the whole run invalid over a property of where `max_tokens` happens to fall rather than anything the interception did.

From this amendment the null control is stated as the claim it was always making.
`routed_sites_at(K + 1)` must be zero at K = 2 and K = 10, because that is the registered fact about the routing table, and a non-zero value means the table moved and the run IS invalid.
The observed routed-call count is a separate quantity, already bound exactly by section 4's per-cell assertion, and it may legitimately be non-zero when a short final pass lands in the window.
When it is zero the cell reads `NULL` and arm 1 must equal arm 2 within `F_O2`, exactly as registered.
When it is not zero the cell is not a null control for that run: it reads `NULL-uncontrolled`, it is not a decider, and its routed-pass count is reported in the row.
The equality is not enforced there, because a routed pass is a real difference and forcing equality across one would be asserting in advance that the kernel does nothing.

### Token identity

Every arm's complete token sequence is recorded for every round.
Greedy speculative decoding is lossless by construction, so the run performs two named identity comparisons rather than collapsing different failures into one label.

- `kernel-diverged` means arm 1 differs from arm 2 in the same round.
  The mismatch is reported by draft, K, round, and token position, the round is excluded from O2 and O3, and the mismatch remains a kernel finding rather than being averaged away.
  The round remains eligible for O1 and O4.
- `mlx-m-dependent` means arm 2 differs from arm 0 in the same round.
  The mismatch is reported by draft, K, round, and token position, the round is excluded from O1 only, and the mismatch is recorded as a finding about MLX's stock M = K + 1 path against its M = 1 path.
  The round remains eligible for O2, O3, and O4.

Arm 4 must equal arm 2 in every round because arm 4 is stock after eligibility has been evaluated.
Any arm 4 mismatch invalidates the run.
Arm 3 is a different target model and takes no part in any token-identity comparison.
No outcome is read from a round that its corresponding identity rule excludes.

## 5. Metrics and decision arithmetic

`generation_tps` comes from `stream_generate`'s final response.
MLX resets that response's clock at the first generated token, so arm 0 is measured in this experiment instead of being compared directly with the earlier A/B's `decode_window` result.

`verify_passes` and `verify_time` come from the target-only class-level counter.
The 2026-08-18 amendment at the foot of section 4 supersedes the `verify_time` half of that sentence and the `verify_share` definition three lines below it: the counter takes no time in a timed round, and the per-pass cost comes from a per-cell probe instead.
For every speculative arm, `accepted_per_pass = generation_tokens / verify_passes`.
Zero verification passes is a typed refusal rather than a division by zero.
For arm 2, `verify_share = verify_time / generation_time`.
At cell level, `verify_time` and `generation_time` are each summed over the O2-eligible arm 2 rounds before that division, so `verify_share = sum(verify_time) / sum(generation_time)`.

For each outcome, both compared arms use the same paired set of rounds left after that outcome's exclusions.
For arm a in one draft and K cell, let `T_a` be the median `generation_tps` over that paired set.
Let `S_a = 100 * (max(samples_a) - min(samples_a)) / median(samples_a)` be that arm's round-to-round spread in percent.
For any numerator arm a and denominator arm b, let `D(a, b) = 100 * (T_a / T_b - 1)`.
If an outcome has no eligible paired round, its median and spread are undefined and the run is invalid.

The noise floor is outcome-specific and is always the larger spread of the two compared arms.

| Outcome | Compared arms | Exact noise floor |
|---|---|---|
| O1 | arm 2 against arm 0 | `F_O1 = max(S_2, S_0)` |
| O2 | arm 1 against arm 2 | `F_O2 = max(S_1, S_2)` |
| O3 | arm 1 against arm 0 | `F_O3 = max(S_1, S_0)` |
| O4 | arm 1 against arm 3 | `F_O4 = max(S_1, S_3)` |

A positive comparison clears its floor only when `D(a, b) > F`.
A negative comparison clears its floor only when `D(a, b) < -F`.
A comparison is inside spread when `abs(D(a, b)) <= F`.

Every draft and K cell also carries an expected-gain ceiling and a decider label for O2.
Routing can improve only the target verification share, so `ceiling_pct = verify_share * per_step_gain_pct(M)` with `M = K + 1`.
The binding A/B supplies 5.70% at M = 5, 14.69% at M = 6, and 15.65% at M = 8, while the meeting-point spike supplies 16.44% at M = 7.
The three binding A/B gains cleared their recorded noise floors of 0.05% at M = 5, 0.10% at M = 6, and 0.07% at M = 8.
The 2026-08-18 confirmation values 5.76%, 14.61%, and 15.64% are not ceiling inputs and are not cited as the published grid.
The M = 7 gain is above both neighbouring binding A/B gains, so these values are not treated as a monotone curve.

| K | M | Registered per-step gain used by the ceiling | Reading |
|---|---|---|---|
| 2 | 3 | 0% | Built-in null cell because nothing routes. |
| 4 | 5 | 5.70% | Binding A/B value at M = 5. |
| 6 | 7 | 16.44% | Meeting-point spike value at M = 7. |
| 8 | 9 | 15.65% | Binding A/B's M = 8 value stands in for the unmeasured M = 9 value, and the substitution is written in every row. |
| 10 | 11 | 0% | Built-in null cell because nothing routes. |

The binding A/B's 14.69% at M = 6 has no K cell in the registered grid and is therefore recorded as provenance but used by no ceiling calculation.
A non-null cell is a decider if and only if `ceiling_pct > F_O2`.
K = 2 and K = 10 have ceiling zero by construction and are validated separately as mandatory null controls.

The memory guard runs before every cell and after every arm.
The final return passes through `check_idle_after`, so a dirty closing idle sample makes the run non-binding instead of allowing exit 0 under contention.

### Amendment, 2026-08-18: where `generation_time` comes from, and who selects the eligible rounds

Both corrections were found by the dispatched writer refusing to write the harness against a section that could not be implemented, and both were reproduced here before being admitted.
Neither changes an arm, an outcome, a threshold, or a number.

First, this section said `verify_share = verify_time / generation_time` without naming a source for `generation_time`, and `mlx_lm` 0.31.3 does not have one to name.
Its `GenerationResponse` carries `text`, `token`, `logprobs`, `from_draft`, `prompt_tokens`, `prompt_tps`, `generation_tokens`, `generation_tps`, `peak_memory` and `finish_reason`, and no elapsed-time field at all.
From this amendment the harness derives it as `generation_time = generation_tokens / generation_tps` from the same final response the other metrics come from.
That is not an estimate of a different quantity.
`stream_generate` resets its clock immediately after the first generated token and then emits `generation_tokens = n + 1` beside `generation_tps = (n + 1) / (perf_counter() - tic)`, so dividing the first by the second returns the very elapsed window `mlx_lm` measured, to float precision.
Any wall clock the harness started itself would instead include prefill and the first token, which this section already excludes on purpose, so the derived value is the more faithful of the two and not merely the available one.

Second, this section requires the share's inputs to be taken over the O2-eligible arm 2 rounds - `verify_passes` and `generation_time`, once the amendment at the foot of section 4 moves the per-pass cost to a probe - and the rules module kept its eligibility filter private.
A harness that reproduced the filter would hold a second copy of the exclusion rule, which is the duplication this experiment already removed once from the attribution rule.
From this amendment `spec_decode_rules` exposes that selection publicly as `eligible_rounds(rounds, outcome)`, the harness calls it for the share, and the verdict functions keep calling the same code for their medians, so the two can never disagree.

## 6. Pre-registered outcomes

The four outcomes are read in order and are reported separately for each draft.

### O1: Does speculation pay at all on this stack?

For each draft, the best K is the K with the highest arm 2 median `generation_tps` after O1's paired-round exclusions, and O1 compares that arm 2 median with the arm 0 median from the same paired rounds in the same K cell.
If two K cells have exactly equal highest arm 2 medians, the smaller K is selected.
O1 is `WIN` when `D(2, 0) > F_O1`.
O1 is `DECELERATES` when `D(2, 0) < -F_O1`.
O1 is `INCONCLUSIVE` when `abs(D(2, 0)) <= F_O1`.
`DECELERATES` is the paper's three-of-five outcome and is recorded under that name rather than tuned away.

### O2: Does routing the verification help?

O2 compares arm 1 with arm 2 at each K for each draft.
K = 2 and K = 10 are checked before the ordinary decider rule: each must have zero routed calls and `abs(D(1, 2)) <= F_O2`, in which case it reads `NULL`, while any other reading invalidates the run.
For K = 4, K = 6, and K = 8, a cell with `ceiling_pct <= F_O2` reads `not-a-decider` and is never interpreted as a null effect.
For a decider cell, O2 is `WIN` when `D(1, 2) > F_O2`, `LOSS` when `D(1, 2) < -F_O2`, and `NULL` when `abs(D(1, 2)) <= F_O2`.

### O3: What is the product number against plain stock decode?

The primary cell is fixed at K = 6 for each draft before the grid is run.
K = 6 is the paper's optimum, the meeting-point spike's cell, and the M = 7 point whose 16.44% step gain sits above both neighbouring binding A/B gains rather than between them.
At that cell O3 compares arm 1 with arm 0, reports `D(1, 0)`, and reports arm 2 against arm 0 and arm 1 against arm 2 beside it so the gain is split between speculation alone and the kernel on top.
The attribution uses `composed_attribution` with `comp_pct = D(1, 0)`, `base_pct = D(2, 0)`, and `noise_pct = F_O3`.

| Exact condition at K = 6 | O3 attribution label |
|---|---|
| `abs(comp_pct) <= F_O3` | `inconclusive` |
| `comp_pct < -F_O3` | `negative` |
| `comp_pct > F_O3` and `base_pct > F_O3` | `artifact-alone` |
| `comp_pct > F_O3` and `base_pct <= F_O3` | `joint` |

The label names come from the earlier A/B even though `base_pct` here measures speculation alone rather than an artifact alone.
The full-grid maximum of `D(1, 0)` over K is reported beside the K = 6 result as `exploratory, selection-biased` and is never the quoted product number.
Any positive O3 claim is stated as an end-to-end batch-1 claim about `mlx_lm`'s speculative path with our kernel against `mlx_lm`'s plain path with stock kernels.
This is the first end-to-end batch-1 claim this repository could make.

### O4: Does the 3-bit path match the speed of speculative stock 4-bit?

O4 is the memory-capacity claim that the 2026-08-15 design document named first.
The primary cell is again fixed at K = 6 for each draft.
O4 reports `ratio_composed_vs_4bit = T_1 / T_3` for arm 1, the speculative 3-bit target with our kernel, against arm 3, the speculative stock 4-bit target.
O4 applies the same attribution arithmetic with `comp_pct = D(1, 3)`, `base_pct = D(2, 3)`, and `noise_pct = F_O4`.
A positive result beyond the floor supports the registered capability wording: "a 3-bit model with our kernel at the speed a user would otherwise get from stock 4-bit".
An inside-floor result is `inconclusive`, a negative result beyond the floor is `negative`, and a positive result is `artifact-alone` when `base_pct > F_O4` or `joint` otherwise.
Arm 3 takes no part in token-identity checks because it is a different model.

**Amendment, 2026-08-18, before any measurement exists: O4's identity exclusion.**
Sections 5 and 6 as first written specified which rounds O1, O2 and O3 exclude and said nothing about O4, so the harness had to choose and the choice was not registered.
Ruled by Vlad before the run: O4 excludes `kernel-diverged` rounds, the same rule O3 uses.

The reasoning, and the counter-argument it had to beat.
O4 claims a user runs the 3-bit model with our kernel and gets the speed of stock 4-bit.
A `kernel-diverged` round is one where our kernel's rounding flipped a near-tie and the model emitted a different token, so what was timed in that round is not the output the claim is about, and a speed number taken from it is precise about the wrong thing.
Against that: the divergence is a mismatch between arms 1 and 2, while O4 compares arms 1 and 3, and arm 3 is a different model whose tokens differ regardless, so token identity is arguably not O4's axis at all.
The first argument wins because the claim is about our kernel rather than about the pair.

What this is expected to cost, stated more carefully than it first was.
An earlier draft of this amendment said the cost is zero because greedy speculative decoding is lossless by construction, and that reason does not support the claim.
Read at `mlx_lm/generate.py:620-634`, the acceptance rule compares each drafted token with the TARGET's own token and, on the first mismatch, emits the target's token instead: `if tn != dtn: break`, then `yield tokens[n]`.
So the algorithm is lossless with respect to the target's decisions, and every emitted token is one the target chose.
That says nothing about `kernel-diverged`, which is arm 1 against arm 2: both run the same lossless algorithm, and what differs is only which kernel computed the target's logits.

The honest expectation is therefore "rare", not "zero".
Our kernel is certified equal to stock within the contract tolerance, which is not bit-equality, and an `argmax` over near-tied logits can flip on a difference far smaller than that tolerance.
The bit-identity test added at the 2026-08-18 merge gate pins the two rank SPELLINGS of our own kernel against each other, not our kernel against MLX's.

If the count is not zero, that is a finding about numerical equivalence at these widths, which is a larger result than anything O4 reports, and the exclusion rule is the least interesting consequence of it.
The same reasoning applies to `mlx-m-dependent`, which can fire for MLX's own stack whenever its M = 1 and M = K + 1 paths round a near-tie differently.

Registered now rather than after the numbers arrive, because a rule chosen once the deltas are visible cannot be shown not to have been chosen to fit them.

No outcome uses a round excluded by its identity rule.
The two drafts retain separate O1, O2, O3, and O4 results.
There is no maximum across drafts because that would introduce the same selection bias one level above the K grid.

## 7. What this does not claim

- This experiment makes no claim about non-greedy sampling.
- This experiment makes no claim about prompts other than the one registered here.
- This experiment makes no new claim about the kernel's correctness beyond the existing certificates and this run's token-identity checks.

## 8. Provenance

The harness verifies and prints the content pins for all four model artifacts before any result is printed.

- The 3-bit target is `qwen3-4b-3bit-g64`.
- The 4-bit target is `qwen3-4b-4bit-g64`.
- The first draft is `qwen3-0.6b-4bit-g64`.
- The second draft is `qwen3-1.7b-4bit-g64`.

The first record also prints the MLX and `mlx_lm` versions and the full machine fingerprint through `provenance()`.
Every result in sections 9 and 10 binds to that four-model manifest, toolchain record, and machine fingerprint.

## 9. Measurements

This section is filled after the registered run.

## 10. Verdicts

This section is filled after the registered run.
