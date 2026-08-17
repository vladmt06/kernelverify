# Batched-decode serving on sub-4-bit weights: the four-arm harness

Tickets T4, T5 and T8 of the 2026-08-15 pivot plan (design doc vlad-pivot-kernel-design-20260815.md, rulings D3.1, D3.2, D3.3, D3.6, D5).
The harness is `bench/serve_sub4bit.py`; its tests are `tests/test_serve_sub4bit.py`.
Sections 1 to 7 are the pre-registration and are written and committed before any measurement exists.
Sections 8 to 10 hold the measurements and stay empty until a coordinated quiet window.

Recorded assumptions of this lane, made because the lane brief is the trace available to it:

- The lane brief quotes rulings D3.1, D3.2, D3.3, D3.6 and D5; the design doc itself was not readable from this session, so the brief's wording is what this doc traces to.
- The pinned artifacts live at the absolute paths `/Users/vlad/kernelverify/bench/.models/qwen3-4b-3bit-g64` and `/Users/vlad/kernelverify/bench/.models/qwen3-4b-4bit-g64` (coordinator's correction of 2026-08-15: `bench/.models/` is gitignored and never propagates to worktrees, so the main worktree's copy is canonical across lanes, the same convention as the llama.cpp checkout).
  The harness verifies them against `PINNED-HASHES.txt` in that directory (lines of `<sha256>  <path relative to bench/.models>`) and refuses smoke and timing modes until they verify.
- Cache depth 512 and generation length 128 are this lane's choices, pre-registered here; the brief fixes the B grid and the metrics but not these two numbers.

## 1. The question

Two questions, split per D3.3 and never merged:

- Primary (the pack's contribution): does routing the model's decode-time quantized linear projections through the verified `kv_wide_qmv` kernel change serving throughput on a 3-bit Qwen3-4B, at batch sizes inside the kernel's dispatch zone, against the same 3-bit model on stock MLX.
- Composed (the capability): does 3-bit-with-kernel beat stock 4-bit serving, with attribution pre-stated per outcome cell in section 6.

## 2. The four arms

| Arm | Model | Patch | Routing |
|-----|-------|-------|---------|
| 1 ours | 3-bit g64 | installed | per `should_dispatch`, fused inside the zone |
| 2 stock3 | 3-bit g64 | none | stock MLX everywhere |
| 3 stock4 | 4-bit g64 | none | stock MLX everywhere |
| 4 control | 3-bit g64 | installed | eligibility evaluated then discarded; always stock |

Arm 4 exists because the 2026-08-15 kv_attention spike died from unmeasured interception overhead (E2E findings doc, same date).
It runs the identical wrapper code path as arm 1, including the full eligibility evaluation and its counters, and then takes the stock path regardless of the answer, so arm4 minus arm2 is the patch's host cost with zero kernel effect in it.
Both quantized 3-bit arms and the control share one pinned 3-bit artifact; arm 3 uses the pinned 4-bit artifact.

## 3. Fixed measurement parameters

- Cache depth: prompt length T = 512 tokens, one shared prompt replicated across all B streams, standard fp16 KV cache in every arm; the KV depth during the timed window is 512 plus the step index.
- Generation length: G = 128 tokens per stream; the timed decode window is the 127 steps after the first token, prefill and first token excluded, matching the spike's convention.
- Metrics, BOTH pre-registered: aggregate batch throughput = B x 127 / window seconds, and per-stream throughput = 127 / window seconds.
- B grid: {1, 4, 5, 6, 8, 11, 12, 16} (edge cells per D3.1).
  Cell roles, as amended on 2026-08-15 by the amendment at the end of this section: B = 1 is the interception-cost cell (arm 4 vs arm 2 feeds the MDE); B = 4, B = 11 and B = 12 are the just-outside edge cells where the patch must dispatch nothing; B = 5 is the lower boundary cell of the win zone and B = 8 is the upper one; B = 6 is the interior win-zone cell; B = 16 documents the above-zone regime where MLX's own routing has no re-read defect.
  B = 11 was registered as the upper boundary cell and is now an edge cell, and that is the only role the amendment moves.
- The win zone was pinned here as exactly {5, 6, 8, 11}, the grid cells inside the pack's two-sided dispatch boundary [5, 11] (ticket T1).
  That set is superseded by the amendment at the end of this section and is left standing above because it is what this pre-registration actually claimed.
  Routing itself still delegates to `kernelverify.pack.wide_qmv.should_dispatch`, but the timing modes refuse to run if the boundary at measurement time disagrees with the pinned zone anywhere on the grid, so a later boundary change forces a re-registration instead of silently rescoping the claim.
  At the time of writing `should_dispatch` is still one-sided (the upper bound is T1's ticket, not yet landed), which is one of the two reasons T8 is blocked; the refusal above makes that blocking mechanical.
- Rounds: 5 per B cell, arms interleaved [1, 2, 3, 4] inside every round; separate passes measure the clock, not the kernels (AGENTS.md).
- Canary and withholding: arm 2 is the round canary; any cell whose arm-2 spread exceeds `machine_state.MAX_SPREAD_PCT` (10%) is withheld, not published.
- Quiet window: timing modes refuse to run unless `machine_state.idle_check` is clean before and after, and T8 rules that no timing is recorded outside a coordinated quiet window.
  A busy machine is a TRANSIENT refusal and exits `EXIT_NOT_IDLE` (9), so a detached runner waits and comes back; a pin mismatch, a moved dispatch boundary and a corpus that is not the registered one are PERMANENT and exit `EXIT_PRECONDITION` (8), which no amount of waiting heals.
- Machine discipline, registered 2026-08-16 (task I1 of the audit-amendments plan): every timing mode and the perplexity mode take the one machine-wide `machine_state.MeasurementLock` before they sample anything or load anything, and the timed modes additionally carry a `phys_footprint` budget (`--budget-gb`, default 24.0 decimal GB) and the machine-wide available-memory gate, both checked at the start of every B cell and after every arm.
  Smoke stays lock-free because it times nothing.
  Every refusal code is `bench/memory_guard.py`'s, shared with the serving calibration and the boundary-pricing probe rather than numbered here: per-harness numbering is what let two "protected" runs collapse the machine on 2026-08-15.
- Artifact integrity: every non-hidden file in each model directory must appear in `PINNED-HASHES.txt` and hash-match before any mode runs; a mismatch or an unlisted file is a refusal, not a warning, because an unlisted tokenizer or config would load unverified while the numbers claim to bind to the manifest.
- Smoke never dispatches a shape outside the registered zone: any grid cell where the pack's boundary disagrees with the pinned zone at any intercepted shape is skipped loudly until the two agree, and smoke is rerun after any change that moves either side.
- Sampling is argmax; all streams share one prompt, so decode cost is shape-determined and identical streams change nothing the clock can see.

### Amendment, 2026-08-15: the win zone is re-registered per shape as {5, 6, 8}

This is a re-registration forced by a measurement, and it is made before any A/B timing has been recorded, under this section's own clause that a later boundary change forces a re-registration instead of a silent rescoping.
No number from this harness exists under either the old registration or the new one, so nothing is being reinterpreted after the fact.

What forced it: `bench/price_qmv_boundary.py` priced the six Qwen3-4B decode dispatch shapes at M = 1..16 and 3 bits, and its run is committed at `bench/results/qmv-boundary-pricing-2026-08-15.json` (sha256 `4a7c500f...`), read in ADR 0015.
The uniform 5..11 window the original pin sat inside was a stated default (ruling D3.1), not a measurement.
The recording routes M = 5..9 at all five intercepted shapes, which meets this grid at {5, 6, 8}.

B = 11 is out of the zone at all five, but the reason differs by shape and is worth stating exactly, because "measured loss" and "undecided" are not the same evidence.
It is a measured LOSS at three of them (q_proj 0.958-0.997, o_proj 0.876-0.935, down_proj 0.757-0.773) and REFUSED at two (k_proj/v_proj 0.987-1.018, gate_proj/up_proj 0.925-1.008), where REFUSED means the ratio interval straddles 1.0 and the cell claims no direction.
It is a WIN at none of them, and the probe's pre-registered rule routes a cell only on a WIN, so an undecided cell leaves the zone exactly as a losing one does.

What changes:

- The pinned zone stops being one set over the grid and becomes one set per intercepted shape, keyed (d_out, d_in), because the routing table it is a claim about is keyed that way and no single pair of bounds can express a table that differs between shapes.
- All five intercepted shapes register {5, 6, 8}: q_proj 4096x2560, k_proj/v_proj 1024x2560, o_proj 2560x4096, gate_proj/up_proj 9728x2560, down_proj 2560x9728.
- lm_head 151936x2560 is priced and wins one width further (M = 5..10), and it is deliberately absent from the registration: it is tied embeddings rather than an `nn.QuantizedLinear` leaf, so the patch never wraps it and this harness cannot route it at any B.
- B = 4 stays outside the zone and now has evidence under it rather than a default: a measured LOSS at q_proj (0.966-0.994) and REFUSED at the other four.

What does not change: the arms, both metrics, the grid itself, the rounds, the canary rule, the withholding rule and the quiet-window rule are all untouched.

Section 4's dispatch count is unchanged in value and changes in derivation only.
The five intercepted shapes share one window, so a grid cell still dispatches either 36 x 7 = 252 fused calls per decode step or none at all.
It is now summed per wrapped site instead of multiplied out as one whole-model number, so a future table that routes one projection and not another would be counted correctly rather than rounded to all-or-nothing.

## 4. The interception, and what keeps it honest

- The patch wraps exactly the model tree's `nn.QuantizedLinear` leaves, by swapping each parent attribute for a wrapper object holding the original layer; install and uninstall happen outside every timed window.
  Nothing global is touched: no class attribute, no module function, no MLX namespace edit, so oracle calls, harness calls and any second model instance cannot route through the kernel under test.
  `tests/test_serve_sub4bit.py` proves this: with the patch installed on model A, direct `mx.quantized_matmul` calls and an independent model B must leave the dispatch counter untouched.
- The wrapper is decode-scoped: a 3-D input with L > 1 falls back to stock with reason `prefill-L{L}` by design, because the kernel and its certificates are decode-shaped (one vector per stream).
  For eligible calls the batch M is the product of the leading dimensions, and the routing question is delegated entirely to `should_dispatch(M)`.
- Eligibility beyond shape: affine mode, bits in `SUPPORTED_BITS`, group size 64, fp16 activations and scales, d_in divisible by 64.
  Every fallback is counted by reason; the only whitelisted reasons are `prefill-*` and `m-*-outside-dispatch`, and any other fallback in a fused round invalidates the round.
  That whitelist is superseded by the 2026-08-16 amendment at the end of this section and is left standing here because it is what this pre-registration actually claimed.
- Exclusions, applied to every arm equally: the embedding and the tied lm_head stay stock (they are not `QuantizedLinear` modules, and the vocab-sized output shape sits outside the pack gate's verified sweep); attention stays stock (the KV cache is fp16 and no KV kernel is under test here).
- The expected dispatch count is exact: 36 layers x 7 wrapped projections = 252 fused calls per decode step when B is in the zone, 0 when it is not; smoke asserts equality, not positivity.

### Amendment, 2026-08-16: three degrees of freedom the code always had and this section never named

Found by the merge gate on the `metal-runner` branch and written before any A/B timing exists, under the same rule the win-zone amendment above followed: a measurement harness's freedoms are registered in writing or removed from the code, never left implicit.
None of the three is new; all three are in the harness as merged, and this amendment is the registration that was missing.

**The bias-term exclusion.**
`_RoutedLinear._ineligible` returns the reason `bias-term` when the wrapped layer carries a bias, so that layer runs stock in every arm.
It is there because the kernel computes x @ W.T and has no bias path at all, so routing a biased layer would silently drop the bias and produce a wrong answer; it is a correctness limit, not a tuning choice.
What it could bias: nothing in this experiment, because all seven of Qwen3-4B's wrapped projections are bias-free and the counter stays at zero.
It would matter for a model whose projections carry biases, and the protection there is that `bias-term` is deliberately NOT whitelisted below, so a round in which it fired is INVALID rather than a quietly part-stock arm 1.

**The whitelisted fallback reasons.**
This section originally said the only whitelisted reasons are `prefill-*` and `m-*-outside-dispatch`, and the code whitelists a wider set.
The registration is now the code's tuple, spelled once, here, and `tests/test_serve_sub4bit.py` reads this line: WHITELIST_PREFIXES = ("prefill-", "m-", "forced-stock").
`forced-stock` has to be whitelisted or arm 4 - the control arm section 2 requires - would invalidate every round it ever ran; that is a necessity of the four-arm design this section simply failed to carry over.
The `m-` prefix is wider than `m-*-outside-dispatch`, and the reason the older spelling stopped matching is the per-shape routing amendment above: the reason string became `m-{M}-outside-dispatch-{d_out}x{d_in}`, which no longer ends where the old pattern expected.
What the widening could bias: a future eligibility reason beginning with `m-` would be whitelisted without anyone deciding it should be, and arm 1 could then be part-stock inside a published round.
The check that keeps it honest is a test asserting that the wrapper emits no `m-` reason other than the routing table declining the cell.

**The fp16 cast of both checkpoints.**
`load_model` calls `model.set_dtype(mx.float16)` on both pinned artifacts.
It is there because the artifacts are bf16-headed while the kernel, the eligibility check (`x.dtype != mx.float16`) and the frozen contract are all fp16: without the cast arm 1 would route nothing and the arms would not share a lane width.
What it could bias: not the comparison, since the same cast is applied to all four arms and both artifacts, but it does mean every number this harness reports - timings and the section 7 perplexity pair alike - describes an fp16 cast of the pinned checkpoints rather than the checkpoints as stored.

## 5. The minimum detectable effect, derived before the A/B

Symbols: L = 36 layers; the seven per-layer projection shapes are q 2560x4096, k 2560x1024, v 2560x1024, o 4096x2560, gate 2560x9728, up 2560x9728, down 9728x2560 (d_in x d_out).

Every quantity below is per decode step, never per token: at B > 1 a step produces B tokens, and mixing the two units silently rescales the arithmetic by B.

- Interception cost i = per-step time of arm 4 minus per-step time of arm 2, measured at B = 1 over the same interleaved rounds.
  The wrapper traversal count is 252 per decode step at every B, so i transfers across the grid unchanged.
- Known residual, stated up front: i excludes the fused path's own per-call launch cost (arm 4 never launches the kernel), and the per-op probe amortizes host cost by batching dispatches past 5 ms.
  The expected gain is therefore an upper bound; the error direction is safe for the decision rule below, because an overpredicted cell that runs the A/B can only come back null, while a non-decider cell stays a non-decider under a smaller true effect.
  The prior spike's MDE missed exactly this cost class, which is why it is named here before any number exists.
- Per-op probe: stock `mx.quantized_matmul` vs the fused kernel at each of the seven true shapes, at each in-zone M, 3-bit g64, interleaved within rounds, batched dispatches of at least 5 ms, weight sets rotated past any cache.
- Expected per-step saving at batch B: saving(B) = L x sum over the seven shapes of (t_stock(shape, B) - t_fused(shape, B)).
- Expected fractional gain at B: gain(B) = (saving(B) - i) / t_step(B), where t_step(B) is arm 2's measured per-step time; the same fraction applies to the aggregate and the per-stream metric because they share the denominator.
- Noise floor: arm 2's round-to-round per-step spread at that B.
- Decision rule, pre-stated: a cell whose expected gain is below its noise floor is a pre-declared non-decider; if every in-zone cell is a non-decider and the reshaping arithmetic (longer G, more rounds) cannot bring the floor under the effect, the honest report is that the arithmetic answers the question, and the A/B runs as confirmation only.

### Amendment, 2026-08-16: the decision rule is now enforced, and the MDE is an upper bound for a second reason

Two things about this section, written before any MDE number exists.

The decision rule above was derived and never applied.
`--mde` printed the expected gain and the noise floor one line apart and compared them nowhere, so every A/B cell read as evidence whatever the arithmetic said.
From this amendment the comparison is made per cell, recorded as `decider` in the MDE's own row, carried to `--ab` through `bench/.cache/serve_mde.json` under the pins it was derived on, and stamped on every A/B row the run publishes - valid, invalid or withheld alike.
`--ab` refuses to start when that record is absent or was derived against different artifacts, because defaulting a cell to "decider" is the reading this rule exists to prevent.
A routed shape whose stock arm's own round-to-round spread exceeds `machine_state.MAX_SPREAD_PCT` withholds its whole cell and the cell publishes no expected gain at all; that spread was measured and reported from the first version of the harness and nothing read it.

The known residual above named one reason the expected gain is an upper bound.
There is a second: `saving(B)` comes from `_op_probe`'s own interleaved rounds and `t_step(B)` from a separate decode pass minutes later, so the ratio composes two numbers measured across passes - the composition interleaving exists to forbid, because a power-state excursion between them moves the denominator without touching the numerator.
It is labelled rather than hidden: every MDE row carries `"composition": "cross-pass"`, and the number is to be read as an upper bound for this reason as well.
Computing it from a single interleaved pass would be a redesign of the MDE and is not made silently here.

Measured inputs to the derivation (from `bench/serve_sub4bit.py --mde`, quiet window only):

Measured 2026-08-17 07:37 to 07:41 UTC by `bench/serve_sub4bit.py --mde` at the default 24.0 GB budget, exit 0, in one strong-idle window held by the detached runner.
MLX 0.32.0, mlx_lm 0.31.3, Apple M3 Pro, 12 cores, 38.65 GB, Darwin 26.5.2 build 25F84.

Interception cost, from B = 1 with arms interleaved over the 5 registered rounds: i = -0.0351 ms per step, against arm-4 and arm-2 spreads of 1.04% and 1.01%.
The measured value is negative and an order of magnitude smaller than either arm's own round-to-round spread, so the honest reading is that the wrapper traversal is not resolvable at this precision, not that it is free or that it pays for itself.
It enters the gain arithmetic as measured rather than clamped to zero, which makes every expected gain very slightly larger; the shift is under 0.13% of a step at every cell and moves no decider label.
The wrapper traversal count is 252 per decode step, exactly as this section requires for i to transfer across the grid unchanged.

Per-op probe, stock `mx.quantized_matmul` against the fused kernel at the seven true shapes, 3-bit g64, interleaved within rounds:

| shape (d_in x d_out) | per layer | B = 5 stock / fused us | B = 6 stock / fused us | B = 8 stock / fused us |
|---|---|---|---|---|
| 2560x4096 | x1 | 56.5 / 52.0 | 76.1 / 61.8 | 94.4 / 75.5 |
| 2560x1024 | x2 | 14.5 / 13.7 | 18.7 / 15.5 | 24.0 / 20.2 |
| 4096x2560 | x1 | 56.8 / 51.6 | 72.6 / 58.8 | 95.8 / 74.7 |
| 2560x9728 | x2 | 133.0 / 121.2 | 176.4 / 142.4 | 220.1 / 174.8 |
| 9728x2560 | x1 | 132.5 / 119.3 | 175.7 / 141.6 | 221.4 / 182.3 |

The fused path is faster at every shape and every in-zone B, and the margin widens with B, which is the re-read defect this kernel exists to remove behaving as the design predicted.

Derivation per cell, over L = 36 layers:

| B | saving(B) ms/step | i ms/step | t_step(B) ms, arm 2 | expected gain | noise floor | decider |
|---|---|---|---|---|---|---|
| 5 | 1.7384 | -0.0351 | 27.557 | 6.44% | 0.98% | yes |
| 6 | 4.9218 | -0.0351 | 34.191 | 14.50% | 0.46% | yes |
| 8 | 6.3847 | -0.0351 | 42.486 | 15.11% | 0.59% | yes |

Every in-zone cell's expected gain clears its own noise floor by at least 6x, so all three are deciders under this section's rule and the A/B ran as a real test rather than as confirmation.
Every row carries `"composition": "cross-pass"`, so each expected gain is read as an upper bound for both of the reasons named above.

## 6. Claim split and attribution (D3.3)

Every comparison below is qualified by the cell's noise floor, defined as arm 2's round-to-round spread in that cell; a ratio whose distance from 1 sits inside the floor supports no claim in either direction.

Primary claim, the pack's contribution: median arm1 / arm2 per in-zone cell, both metrics reported.
The win criterion is explicit: arm1 / arm2 > 1 beyond the floor is a win, a ratio inside the floor is a null result, and arm1 / arm2 < 1 beyond the floor is a regression finding and is reported as such.

Patch cost, reported alongside: arm4 / arm2 per cell; a cost beyond the floor at any cell is reported as the patch's price, and arm1 / arm4 is the kernel effect given the patch (diagnostic only, never the headline).

Composed claim, separate and never merged with the primary: arm1 vs arm3 per cell, with attribution fixed now, per outcome:

| Outcome in a cell | Attribution written in the verdict |
|---|---|
| arm1 vs arm3 inside the floor | Inconclusive; no composed claim in either direction. |
| arm1 > arm3 beyond the floor, and arm2 > arm3 beyond the floor | The 3-bit artifact alone already beats 4-bit; the composed win is the artifact's bandwidth, and the kernel's own contribution stays the primary ratio. |
| arm1 > arm3 beyond the floor, arm2 not beyond it | The composed win exists only with the kernel; attribution is joint: the artifact supplies the memory saving, the kernel unlocks it. |
| arm1 < arm3 beyond the floor | No composed capability claim; the cell is reported as a negative result. |

The T5 quality pair (section 7) is printed beside any composed claim, always.

## 7. The perplexity proxy (T5, D5)

- Purpose: one pre-registered quality number pair, 3-bit vs 4-bit artifact, so the capability claim never travels without its quality cost.
- Corpus: one plain-text file at `bench/.corpus/ppl.txt`; its sha256 and token count are recorded in section 8 before either perplexity number is read.
  Preferred source is the wikitext-2-raw-v1 test split; if the network refuses, the coordinator pins the corpus file instead, and either way the hash on record is what the numbers bind to.
- Method, fixed now: both artifacts are scored stock (no patch installed), fp16, on the identical token stream; the two tokenizers must produce byte-identical token ids or the run refuses; the stream is cut into non-overlapping windows of 1024 tokens, the first 96 windows are scored, positions 2..1024 of each window contribute their negative log-likelihood computed in float32, and the reported number is exp of the mean over all contributing positions.
- Output: exactly one pair, ppl_3bit and ppl_4bit, plus the corpus hash and token count.
- Perplexity is not a timing, so it may run outside the quiet window, but never concurrently with any timing mode.

## 8. Measurements: MDE and corpus record

Corpus record, written before either perplexity number exists:

- File: `/Users/vlad/kernelverify/bench/.corpus/ppl.txt`, the wikitext-2-raw-v1 test split (`wiki.test.raw` from the dataset author's mirror at wikitext.smerity.com), 1,290,590 bytes.
- sha256: `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.
- `bench/.corpus/` is gitignored, the same convention as `bench/.models/`, so the file never travels through a merge and each worktree that scores perplexity holds its own copy.
  That is exactly why the harness checks the digest rather than the path (registered 2026-08-16, task I1): the path is per worktree and only the content identifies the corpus.
  A file that does not hash to the sha256 above is refused with `EXIT_PRECONDITION` before either model is loaded.
- Both artifacts tokenize it identically: 299,078 tokens against the 98,304 the 96 registered windows need.
- A plumbing check ran before this record: tokenizer identity plus a 2-window scoring pass on the 3-bit model only; the 4-bit side was deliberately not computed, so no preview of the registered pair exists.

MDE record, written 2026-08-17:

- Run: `bench/serve_sub4bit.py --mde`, detached runner, strong-idle window held from 07:37:48 UTC, 4 minutes wall clock, exit 0, closing idle sample clean.
- Budget: the default 24.0 GB; the footprint never approached it in this mode.
- Decider labels persisted to `bench/.cache/serve_mde.json` under the pinned-artifact manifest they were derived on: B = 5, 6 and 8 all `true`.
  `--ab` reads that record and refuses to start without it, so the A/B below is bound to these labels and to these artifacts.
- The harness printed `DECIDERS: every MDE cell's expected gain cleared its own noise floor`, meaning no cell was pre-declared a non-decider and section 5's confirmation-only branch was not taken.
- The numbers themselves are in section 5, beside the derivation they feed.

Perplexity pair (T5), measured 2026-08-17 after the A/B:

- Run: `bench/serve_sub4bit.py --ppl --corpus /Users/vlad/kv-baseline/bench/.corpus/ppl.txt`, exit 0, machine lock held, no idle gate by section 7's own rule that perplexity is not a timing.
- The corpus read is not the path named above, which holds no file in this worktree, but a surviving copy under the `kv-baseline` worktree.
  It hashes to the registered `173c87a5...` and the harness verifies that digest before loading either model, so the numbers bind to the registered CONTENT exactly as section 7 requires; the path difference is what this section already anticipated when it recorded that only the content identifies the corpus.
- 299,078 tokens, byte-identical ids from both tokenizers, 96 non-overlapping windows of 1024, positions 2..1024 scored in float32.

| artifact | perplexity |
|---|---|
| 3-bit g64 | 22.7069 |
| 4-bit g64 | 15.2355 |

The 3-bit artifact is 49.0% worse in perplexity than the 4-bit artifact on the registered corpus.
That is the quality cost the capability claim in section 10 must always be read against, and it is large.

A benign warning appears twice in the run log and is recorded so it is not mistaken later for a defect: transformers reports that 299,078 ids exceed the model's 131,072 maximum sequence length.
It refers to the one-shot encode of the whole corpus file, not to anything the model is asked to process; every forward pass sees exactly one 1024-token window, so no indexing error is possible and none occurred.

## 9. The A/B grid

Measured by `bench/serve_sub4bit.py --ab --budget-gb 30`, detached runner, 2026-08-17 15:53 to 16:27 UTC, 33 minutes, exit 0, idle clean before and after.
Same pinned artifacts, same MLX and machine fingerprint as the MDE above.
Per-stream throughput in tokens per second, medians over the 5 registered rounds; aggregate is B times per-stream and is reported in the same rows of the raw log.

| B | zone | decider | arm 1 ours | arm 2 stock3 | arm 3 stock4 | arm 4 control | arm1/arm2 | noise floor |
|---|---|---|---|---|---|---|---|---|
| 1 | out | no | 64.745 | 64.681 | 52.285 | 64.649 | 1.0010 | 0.67% |
| 4 | out | no | 43.178 | 43.143 | 43.710 | 43.159 | 1.0008 | 0.10% |
| 5 | IN | yes | 38.554 | 36.476 | 36.935 | 36.501 | **1.0570** | 0.05% |
| 6 | IN | yes | 33.548 | 29.252 | 30.647 | 29.247 | **1.1469** | 0.10% |
| 8 | IN | yes | 27.248 | 23.561 | 24.517 | 23.558 | **1.1565** | 0.07% |
| 11 | out | no | 16.847 | 16.849 | 16.936 | 16.845 | 0.9999 | 0.03% |
| 12 | out | no | 16.458 | 16.462 | 16.572 | 16.462 | 0.9997 | 1.79% |
| 16 | out | no | 16.064 | 15.820 | 15.821 | 16.065 | 1.0154 | 0.25% |

Validity, per cell: no cell was withheld, every arm-2 spread sat far under the 10% canary limit, no hard fallback occurred anywhere, and every in-zone cell dispatched exactly 160,020 fused calls against 160,020 expected across 5 routed shapes.
Every out-of-zone cell dispatched zero, as the edge cells require.

Three runs were needed to produce this one, and all three are recorded because the discarded ones are evidence about the harness:

| Run | Budget | Outcome | Why it is not the binding run |
|---|---|---|---|
| 1, 14:19 UTC | 24.0 GB default | refused at B = 12 arm 1, footprint 25.72 GB, `EXIT_BUDGET_REFUSAL` | The harness discards a run whose budget was crossed; six completed cells count for nothing by its own rule. |
| 2, 14:48 UTC | 30.0 GB | all 8 cells completed, exit 1 | The closing idle sample caught WindowServer at 15% CPU, so `check_idle_after` marked the run non-binding. |
| 3, 15:53 UTC | 30.0 GB | all 8 cells completed, exit 0 | This is the binding run and the table above. |

Deviation from the registered default, recorded because section 3 pins it: the binding run used `--budget-gb 30` rather than the default 24.0.
The reason is a defect in the harness rather than a property of the measurement.
`mde()` and `ab()` never call `mx.clear_cache()`, so MLX's freed-buffer cache is never returned to the allocator, and `phys_footprint` counts those dead buffers; the footprint therefore ratchets upward across cells until it crosses any fixed budget, which it did at B = 12 in run 1.
30.0 GB is the same ceiling the serving calibration already runs under and leaves roughly 8.6 GB of headroom on this 38.65 GB machine, so it widens the guard without disarming it.
The fix is to clear the cache at each cell boundary and restore the 24.0 GB default; it is queued in `TODOS.md` as its own change and is not made here, because editing the harness between the MDE and the A/B would have unbound the two.

### Amendment, 2026-08-17: the cache fix is made, and this re-run is pre-registered before it is taken

The defect named above is now fixed: `mde()` and `ab()` call `mx.clear_cache()` at each cell boundary, immediately BEFORE the guard reads the footprint, and the 24.0 GB default is pinned by a test.
Section 3's registered budget therefore applies again, and this re-run is taken at that default with no flag.

The outcome is pre-registered here, before the run, because a re-run of a grid whose numbers are already published can be rationalised after the fact in either direction.

- The budget must hold. Eight cells complete at the registered 24.0 GB default and the run exits 0. If it refuses at 24.0 GB the fix did not work and the grid below stands unchanged.
- The in-zone ratios must not move beyond the spread this document already records. The binding run's arm1/arm2 at B = 5, 6 and 8 were 1.0570, 1.1469 and 1.1565, and run 2 agreed to within 0.0014. A re-run landing inside that band confirms the fix is numerically inert and section 10's verdicts stand as written.
- A ratio that moves outside that band is a FINDING, not a tuning knob, and it is the one outcome that changes the verdicts. `mx.clear_cache()` discards warmed allocations, so the first dispatch of each cell may pay an allocation cost the published grid did not. The per-cell warm-up already precedes the timed rounds, which is why inertness is the expectation rather than the hope - but if the numbers move, this document records the new grid and section 10 is re-derived from it. No amendment may explain a moved ratio away.
- The verdict on `wide_qmv` staying in the pack is NOT reopened by this run unless a ratio crosses 1.0 or its cell's noise floor. This run tests the harness, not the kernel.

Reproducibility, stated at the grade the evidence actually carries: run 3 is the binding run and the only one, and run 2 is a NON-BINDING run whose in-zone ratios agree with it to 1.0566 vs 1.0570, 1.1455 vs 1.1469 and 1.1558 vs 1.1565.
An earlier draft of this paragraph called the two "independent quiet windows", which contradicted this section's own run table three paragraphs above: the harness marked run 2 non-binding because its closing idle sample caught WindowServer at 15% CPU, and section 3 registers that a timing mode's window must be clean before AND after.
The agreement is therefore corroboration that the harness reproduces itself, not a second measurement, and every verdict below rests on run 3 alone.

## 10. Verdicts

PRIMARY CLAIM, the pack's contribution, arm 1 against arm 2, judged against each cell's own noise floor per section 6:

| B | arm1/arm2 | noise floor | margin over floor | verdict |
|---|---|---|---|---|
| 5 | 1.0570 | 0.05% | 114x | **win** |
| 6 | 1.1469 | 0.10% | 147x | **win** |
| 8 | 1.1565 | 0.07% | 224x | **win** |

`kv_wide_qmv` is a win at every cell of its re-registered dispatch zone, by 5.7%, 14.7% and 15.7% of per-stream throughput, each more than a hundred times its cell's noise floor.
This is the pre-registered outcome under which the kernel STAYS in the pack for the multi-stream regime it was measured in.
The retirement branch, which would have fired on a loss or a within-spread result at every routed batch, did not fire.
Per section 5's own rule this says nothing whatever about batch 1, where the pack routes nothing by construction.

Measured against the forecast, which was written before the A/B ran: expected 6.44 / 14.50 / 15.11%, measured 5.70 / 14.69 / 15.65%.
Section 5 registered the expected gain as an upper bound for two named reasons, and B = 5 lands under its forecast as that framing predicts.
B = 6 and B = 8 land marginally above theirs, by 0.19 and 0.54 points, which the upper-bound framing does not predict.
That is recorded as an open discrepancy rather than explained away here: the two numbers are composed across passes, so a small excursion in either direction is within what the cross-pass composition can produce, and nothing in this document needs the forecast to have been tight.

PATCH COST, arm 4 against arm 2, reported alongside per section 6: 1.0007, 0.9998 and 0.9999 at B = 5, 6 and 8.
The patch's host cost is not resolvable at this precision in any in-zone cell, consistent with the interception cost measured in section 5.
The 2026-08-15 kv_attention failure mode, a kernel win eaten by unmeasured interception overhead, does not occur here.

OUT-OF-ZONE CELLS, none of which is a decider and none of which supports a claim:

- B = 1: arm1/arm2 = 1.0010 inside a 0.67% floor, a null result and the correct one, since the pack routes nothing at B = 1 and arm 1 must therefore measure arm 2.
- B = 4, 11, 12: null, all within their floors, and all dispatched zero fused calls as required.
- B = 16: arm1/arm2 = 1.0154 against a 0.25% floor, which the harness labels a win.
  It is not a kernel effect and must not be read as one.
  The control arm moved with it, arm4/arm2 = 1.0155, and arm 4 routes nothing by construction, so the 1.5% is arm 2 running slightly slow in that cell rather than anything the kernel did.
  The cell is a non-decider and is reported here only so the label in the raw row is not mistaken later for a finding.

COMPOSED CLAIM, arm 1 against arm 3, with the attribution fixed in section 6 before any measurement: ratios 1.0438, 1.0947 and 1.1114 at B = 5, 6 and 8, each far beyond its floor, with arm2/arm3 at 0.9876, 0.9544 and 0.9610, all below 1.
That is section 6's third row exactly: arm1 above arm3 beyond the floor while arm2 is not, so the attribution is JOINT.
The 3-bit artifact alone is slower than stock 4-bit at every in-zone cell; with the kernel it is faster.
The artifact supplies the memory saving and the kernel is what unlocks it.

The quality pair section 6 requires beside this claim, from section 8: perplexity 22.7069 at 3 bits against 15.2355 at 4 bits, the 3-bit artifact 49.0% worse on the registered corpus.
The composed claim is therefore published with its cost attached, and the cost dominates the benefit for any reader who cares about output quality: the capability is a 4.4 to 11.1% throughput win bought with a 49% perplexity regression.
Section 7 registered this pair precisely so the capability claim could never travel without it, and this is the case that rule was written for.
Nothing here argues the trade is worth making; that judgement belongs to whoever chooses the artifact, and this record's job is to make sure they see both numbers at once.

THE BATCH-1 BASELINE, which is the number this plan exists to produce and the only one that speaks to the product's actual target:

| Arm at B = 1 | per-stream tokens/s |
|---|---|
| stock 3-bit | 64.681 |
| stock 4-bit | 52.285 |
| ours, routing nothing | 64.745 |

64.7 tokens per second on the 3-bit artifact is the end-to-end batch-1 figure every future kernel spike must beat, and the pack contributes nothing to it.
The 3-bit artifact alone is 23.7% faster than stock 4-bit at batch 1 (arm 2 against arm 3), which is the bandwidth saving of the narrower weights and not a kernel result.
That sentence compares the two artifacts, so section 6's rule binds it exactly as it binds the composed claim, and the pair travels with it here rather than being left to the reader to fetch: perplexity 22.7069 at 3 bits against 15.2355 at 4 bits, the 3-bit artifact 49.0% worse on the registered corpus.
Read together, the batch-1 picture is that the 3-bit artifact buys 23.7% of throughput with 49.0% of perplexity, and the kernel changes neither number.
This is a baseline, not a claim of speedup: nothing in this document makes single-user decode faster, and the research that intends to is opened by the literature review, not here.
The out-of-zone cells above support no claim about the KERNEL, which is what that sentence means; the artifact-versus-artifact comparison in this table is a different question and is qualified by its own cost.
