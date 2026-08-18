# Batched serving through mlx-lm's own engine, pre-registered

Status: sections 1 to 8 written and committed BEFORE the harness exists and before any measurement.
Sections 9 and 10 are written after the run and record what it produced.

Lane: `lane/batch-decode`.
Plan: the eng-reviewed implementation plan of 2026-08-18, rulings D1 to D7.
Parent evidence: `docs/research/2026-08-15-sub4bit-serve-findings.md` sections 4, 9 and 10; `docs/research/2026-08-18-spec-decode-e2e.md` for the apparatus this reuses.

## 1. The question

For an operator serving B concurrent requests from one MacBook through mlx-lm's own batched engine, does routing our kernel at the decode step make a 3-bit Qwen3-4B faster than stock 3-bit, and faster than stock 4-bit, and does the win the A/B measured in our own loop survive the engine?

The comparison is mlx-lm's `BatchGenerator` with our kernel routed against the same engine with stock kernels, on the same pinned artifacts, and it is spelled that way wherever it is quoted.

Why this question and not the published one.
The A/B of 2026-08-15 measured the kernel at B = 5, 6 and 8 in a decode loop this repository wrote, over B identical copies of one prompt.
That is a kernel measurement and it stands as one.
It is not evidence about a user, because no user runs that loop; the code a user runs is `BatchGenerator`, which mlx-lm's own server drives, and which nothing in this repository has ever exercised.
Risk RR1 of the 2026-08-15 pivot review named exactly this gap and it has been open since.

## 2. The arms

Four arms, every one driven through `BatchGenerator` identically, differing only in which model and which patch state.

| Arm | Model | Patch | What it is |
|---|---|---|---|
| 1, ours | 3-bit g64 | installed, fused | the claim |
| 2, stock 3-bit | 3-bit g64 | none | the kernel question's comparator |
| 3, stock 4-bit | 4-bit g64 | none | the capability question's comparator |
| 4, control | 3-bit g64 | installed, forced stock | eligibility evaluated and discarded, so arm 4 against arm 2 is the interception's own host cost |

Every arm runs once per round, with the starting arm rotated each round: round 1 runs 1, 2, 3, 4; round 2 runs 2, 3, 4, 1; round 3 runs 3, 4, 1, 2; and so on.
The A/B ran a fixed order, and a fixed order gives one arm the first and coolest slot in every round; rotation removes that without disturbing OB3, which compares ratios.

There is no plain-decode arm.
What a user has today at B concurrent streams is stock 3-bit or stock 4-bit through this same engine, and those are arms 2 and 3.

## 3. Fixed parameters

`B_GRID = [1, 4, 5, 6, 7, 8, 9, 11, 12, 16]`, the A/B's grid plus the two in-window widths it skipped, run in that order.
In-zone cells are 5, 6, 7, 8 and 9 by `PINNED_ZONE`, which is the whole routed window; every other cell is a null cell where nothing may route.
B = 7 and B = 9 have no published loop number to replicate against, so OB3 reads `no-loop-number` at those two cells and compares nothing there; their ceilings are substituted in section 5 and the substitution is written.

`PROMPT_T = 512`, `GEN_TOKENS = 128`, `ROUNDS = 5`, budget 24.0 GB, `mx.clear_cache()` before every cell's guard and the guard after every arm.

Prompts are B distinct windows of 512 tokens cut from the A/B's registered token stream, `PROMPT_SEED * 40` encoded by the target's tokenizer, with stream i taking tokens 64i through 64i + 512.
Stream 0 is therefore the A/B's exact prompt, and the stream is 2921 tokens long, which is enough for B = 16 at stride 64.
A stream too short for the requested B refuses before any GPU work.

Sampling is the engine's default argmax.
No stop tokens are given and every stream has `max_tokens = 128`, so no stream finishes early and the width of every decode pass is B, which is the effect `mlx_lm.benchmark` gets by blanking the tokenizer's EOS set.

`BatchGenerator` is configured with `completion_batch_size = 32`, `prefill_batch_size = 16` and `prefill_step_size = 2048`; therefore every stream is admitted in one prefill batch and every decode call has full width B, and this run does not measure the server default `prompt_concurrency = 8`.
The reason is measured from the engine's scheduler rather than assumed: under the default of 8, a request of more than 8 streams has its first 8 admitted, moved into generation, and decoded at width 8 for at least one step before the rest join, and width 8 is inside the routed window, so B = 11, 12 and 16 would route during their first steps under the default.
That behaviour belongs to the server-path question and is registered out of scope in section 7.

Warm-up is one complete 128-token generation per arm per cell before the rounds, registered here because the K = 4 follow-up showed an unspecified warm-up is a degree of freedom.

The two targets are the pinned `qwen3-4b-3bit-g64` and `qwen3-4b-4bit-g64`; both stay resident for the run as they did in the A/B, under the 24.0 GB budget with no flag.
A budget refusal at any cell discards the run: the budget is not raised, the grid is not shrunk and neither target is unloaded in response.

## 4. What keeps the interception honest

The counted seam is `Model.__call__` on the 3-bit target, wrapped once at class level with a `self is target` filter and restored in `finally`, the instrument the spec-decode lane built and its tests pin.

A decode pass is a rank-2 call whose width, `prod(shape)`, equals B.
The only other call the registered configuration produces is exactly one prefill of width `B * (PROMPT_T - 1)`, because the engine holds the last prompt token back.
A second prefill, a call of any other width, or a call of another rank is a typed refusal that names the cell.

The initial `(B, 1)` call on the held-back final prompt token happens at `GenerationBatch` construction, is a decode pass, and enters the exact routed-call count.
The engine also samples one step ahead, so the number of decode calls may exceed the number of tokens returned by one.
The number of decode calls is therefore observed and recorded, never asserted to be 128, and must be equal across the four arms of a round or the round is invalid, because the clock's denominator must be the same work.

`generation_tokens` must equal `128 * B`, every stream must return exactly 128 tokens, and every final response must say `length`; any of these failing invalidates the run.

Expected routed calls for arm 1 are the sum over the observed DECODE passes only of the number of sites whose routing table admits width B.
This is the rule the spec-decode harness registered for its verification passes and for the same reason: a sum over every call would agree with the counter by construction, because the same table drives both, whereas this sum lets a routed prefill call surface as a mismatch.
Arm 1 must match that sum exactly and arm 4 must route zero; a mismatch is a typed error that stops the run and names the cell, so a routed prefill is a finding about the interception rather than noise.

Zero hard fallbacks in arms 1 and 4 per round, or the round is invalid.

Token identity is per stream in insertion order.
`kernel-diverged` is recorded when arm 1's tokens differ from arm 2's, reported by (B, round, stream, position), and that round is excluded from the kernel and capability outcomes.
Arm 4 must equal arm 2 in every stream of every round or the run is invalid.
Arm 3 is a different model and is never compared for identity.
At most one round per cell may be excluded for divergence; a second exclusion marks the cell `identity-unstable`, which is not read for OB1 or OB2 and counts against GO, so the exclusion can never hide a pattern in the very thing being validated.

## 5. Metrics, floors, ceilings

Per arm per round: `generation_tps` from the engine's own `BatchStats` around the drain, which is aggregate over streams; `generation_tokens` and `generation_time` recorded beside it.
Per-stream throughput is the aggregate divided by B, exact under section 3 because no stream stops early.

Per arm per cell: median and `spread_pct` over the eligible rounds.

The noise floor is per comparison and is the larger spread of the two arms compared: `F12 = max(S1, S2)` for the kernel question, `F13 = max(S1, S3)` for the capability, `F14 = max(S1, S4)` for the null-cell control.
A delta equal to its floor does not clear it.

The ceiling at an in-zone cell is the kernel gain already measured at that width, because routing inside the engine can only recover what the loop measured and the engine adds cost on top.
At B = 5, 6 and 8 those are the A/B binding run's +5.70%, +14.69% and +15.65%.
At B = 7 the ceiling is the spike's +16.44% at M = 7, recorded in `docs/research/2026-08-18-spec-verify-meeting-point-spike.md`.
At B = 9 the ceiling is the A/B's +15.65% at M = 8 standing in for M = 9, the same substitution the spec-decode lane registered for its own M = 9 cell.
Both substitutions are written here, before the run.
Null cells have ceiling 0.
A cell is a decider if and only if its ceiling is strictly greater than `F12`.

A NO-GO caused only by `not-a-decider` cells is a statement about this design's power at those cells and is recorded as such, exactly as the K = 4 follow-up recorded its own; it is not evidence that the kernel does not help there.

B = 1 is measured through the engine like every other cell, and because the engine's batch cache builds a mask even for one stream, B = 1 validates the engine and claims no parity with the loop.

The interception host cost at every cell is arm 4 against arm 2, reported.

## 6. Pre-registered outcomes

**OB1, the kernel question**, per cell, arm 1 against arm 2.
On an in-zone decider cell: WIN if the delta clears `F12` upward, LOSS if it clears downward, NULL otherwise.
A non-decider in-zone cell reads `not-a-decider`, never NULL.
On a null cell: NULL if and only if routed calls were 0 by the table AND arm 1 equals arm 4 within `F14`; otherwise `NULL-uncontrolled`, which is a finding about the interception, reported, and does not void the run.
B = 16 supports only the null-control reading; arm 1 against arm 2 there is interception host cost and is never a kernel verdict, whatever its size, because the A/B's 1.0154 at that cell moved with the control.

**OB2, the capability**, per in-zone cell, arm 1 against arm 3, floor `F13`.
`composed_attribution` reports how much is the artifact alone (arm 2 against arm 3) and how much the kernel on top (arm 1 against arm 2).
The perplexity pair 22.7069 against 15.2355, quoted from the A/B on the same artifacts, is printed beside every OB2 row and decides nothing.

**The ZONE verdict.**
GO if and only if at every in-zone cell B in {5, 6, 7, 8, 9}: OB1 reads WIN and OB2's delta clears its floor upward.
Anything else is NO-GO: one LOSS, one NULL, one `not-a-decider`, one `identity-unstable` cell, or one OB2 inside its floor is enough.
The quoted numbers are the ranges of the OB1 and OB2 deltas across the five cells, never a single favourite cell.

**OB3, the engine against the loop**, reported BEFORE the zone verdict is read.
Per in-zone cell with a published loop number, so B = 5, 6 and 8, this run's arm 1 to arm 2 ratio and arm 1 to arm 3 ratio are reported beside the A/B's published 1.0570, 1.1469, 1.1565 and 1.0438, 1.0947, 1.1114; B = 7 and B = 9 read `no-loop-number`.
The registered expectation is that the engine's ratios sit at or below the loop's, because non-matmul engine cost dilutes them, and that reading is `AT-OR-BELOW-LOOP`.
With `ratio_delta_pct = 100 * (engine_ratio / loop_ratio - 1)`, a value strictly greater than `max(engine F12, published loop floor)` is the finding `ENGINE-ABOVE-LOOP`, reported before the zone verdict; it is not a GO condition and it is not a win.

No outcome may be read from a round its identity check excluded.

A NO-GO closes the 3-bit batched claim through this engine on this stack.
It does not reopen the loop-level A/B, which stands as a kernel measurement of what it measured.
It does not decide the 4-bit question, which routes nowhere today and has its own queued pricing run.

## 7. What this does not claim

Nothing about prompts of different lengths or the engine's left-padding path.
Nothing about continuous arrival or a B that varies within a request.
Nothing about the server's default prompt concurrency of 8, under which a burst of more than 8 streams decodes its first steps at width 8, inside the routed window.
Nothing about the HTTP server or client-side latency.
Nothing about non-greedy sampling.
Nothing about 4-bit routing, which routes nowhere today.
Nothing about any prompt other than the registered stream.
Nothing about llama.cpp, whose batched baseline is its own deferred block.

At B up to 8 this configuration matches what `mlx_lm.benchmark -b B` drives, so a reader can reproduce the arms 2 and 3 numbers with the stack's own tool on the same artifacts; above 8 that benchmark's default admits in eights, which is the section 3 finding.

## 8. Provenance

Printed first in every run log, as `provenance()` does: the sha256 pins of both models, the MLX and mlx-lm versions, the machine fingerprint, the routed-window pins, and one sha256 digest of the B prompt token lists per cell.

The engine's decode call line and its construction-time first step are pinned by tests against the installed mlx-lm source, so an upgrade that moves the seam turns the suite red rather than silently changing what the counter counts.
