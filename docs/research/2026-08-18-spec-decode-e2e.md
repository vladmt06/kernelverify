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
