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
- Exclusions, applied to every arm equally: the embedding and the tied lm_head stay stock (they are not `QuantizedLinear` modules, and the vocab-sized output shape sits outside the pack gate's verified sweep); attention stays stock (the KV cache is fp16 and no KV kernel is under test here).
- The expected dispatch count is exact: 36 layers x 7 wrapped projections = 252 fused calls per decode step when B is in the zone, 0 when it is not; smoke asserts equality, not positivity.

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

Measured inputs to the derivation (from `bench/serve_sub4bit.py --mde`, quiet window only):

<!-- MDE numbers go here -->

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
- Both artifacts tokenize it identically: 299,078 tokens against the 98,304 the 96 registered windows need.
- A plumbing check ran before this record: tokenizer identity plus a 2-window scoring pass on the 3-bit model only; the 4-bit side was deliberately not computed, so no preview of the registered pair exists.

<!-- results from bench/serve_sub4bit.py --mde and --ppl go here -->

## 9. The A/B grid

<!-- results from bench/serve_sub4bit.py --ab go here -->

## 10. Verdicts

<!-- primary per-cell verdicts, patch cost, composed claim with its pre-stated attribution row, quality pair beside it -->
