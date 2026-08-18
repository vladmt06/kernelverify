# The K=6 meeting point: does routing a verification step buy anything?

Status: PRE-REGISTERED 2026-08-18, before the spike was written or run.

## 1. The question

A speculative decoder drafts K tokens and verifies them in one pass.
At K = 6 that pass presents 7 tokens of one stream to every projection, shape (1, 7, d_in).
This repo's `wide_qmv` kernel routes M = 5..9 and was measured on 2026-08-17 at 14.61% faster than stock at M = 6 and 15.64% at M = 8, so M = 7 sits inside the window and between two wins.

The gate refuses that shape today.
`_RoutedLinear._ineligible` returns `prefill-L{n}` for any 3-D input whose sequence length is not 1, before it computes M or consults `should_dispatch`, so a verification step falls back to stock while the A/B's B = 7 cell - the same (7, d_in) matmul, spelled (7, 1, d_in) - routes.

The question this spike answers is narrow and it is NOT "is speculative decoding faster".
It is: **if the gate let the verification shape through, would the step get faster?**

## 2. What is already known, and must not be re-derived

- The kernel does identical work for both spellings. `_fused` flattens with `x.reshape(-1, d_in)` before dispatch, so (1, 7, d_in) and (7, 1, d_in) produce the same kernel call on the same array. There is therefore NO kernel-level question here and this spike must not pretend to measure one.
- What differs is everything around the projection: a 7-token step has 7 query positions against the KV cache where a 7-stream step has one each. The projections are the same; attention and cache handling are not.
- So the only honest question is at the step level, and the answer could be a win, a null, or a loss even though the kernel is faster in isolation.

## 3. Method

One model (the pinned 3-bit artifact), one stream, a cache primed by a prompt, then repeated 7-token forward passes - the shape a K = 6 verification produces.

Two arms, interleaved within every round so drift cannot land on one of them:

- arm S (spec-routed): projections routed through `wide_qmv` when the flattened M is in the routed window, sequence spelling allowed.
- arm T (stock): the same model, unpatched.

The spike installs its OWN interception, not `serve_sub4bit`'s.
`bench/serve_sub4bit.py` is the harness whose grid was re-measured on 2026-08-17 and it is left untouched, so this spike cannot move a published number by editing the thing that produced it.

Fixed before running: 5 rounds, the same round count the A/B registers; median over rounds; per-arm spread reported and a cell withheld if either arm's spread exceeds the win it claims.

## 4. Pre-registered outcomes

- **GO**: arm S beats arm T by more than both arms' spread. The eligibility rule is worth changing under its own pre-registration, and a draft/verify loop becomes worth building.
- **NO-GO**: arm S is at or below arm T beyond spread. The meeting point is refuted at the step level despite the kernel winning in isolation, the finding is recorded, and the K = 6 branch is closed for a measured reason rather than an argued one.
- **INCONCLUSIVE**: the difference sits inside the spread. Recorded as inconclusive and NOT read as either outcome; the honest next move would be more rounds, not a re-reading of these.

The kernel's isolated win at M = 5..9 does NOT license reading a null here as a win.
If the step does not get faster, the step does not get faster, and no amendment may explain that away by pointing at the projection-level number.

## 5. What this spike does not claim

It does not measure acceptance rates, drafting cost, or end-to-end speculative-decode throughput.
Those need a draft model and a draft/verify loop, neither of which exists here, and both of which are only worth building if this comes back GO.

## 6. Result

Measured by `bench/spike_spec_verify.py`, detached runner, 2026-08-18 00:21 UTC, exit 0, AC power, display asleep, five consecutive clean idle samples before the start.
Same pinned 3-bit artifact and machine fingerprint as the 2026-08-17 A/B.

**Verdict: GO.**

| quantity | value |
|---|---|
| arm S, spec-routed, median | 33.711 ms |
| arm T, stock, median | 39.252 ms |
| ratio stock/spec | 1.1644 |
| gain | 16.44% |
| arm S spread | 0.327% |
| arm T spread | 0.280% |
| noise floor | 0.327% |

The gain is 50 times the noise floor, so the pre-registered INCONCLUSIVE branch does not apply and the NO-GO branch is refuted.

The premise held too, and it was checked before the timing rather than assumed.
One verification pass routed 504 calls across 252 wrapped sites, and every fallback in that pass was the 64-token prompt prefill, refused by name as `m-64-outside-dispatch-<shape>`.
That is the relaxed rule behaving as TODOS argued it would: the verification shape opens, and real prefill is still refused - not by a blunt sequence check, but by `should_dispatch` on the merits, because a 64-token prefill is far outside the routed window of 5..9.

The number also agrees with evidence collected independently and earlier.
The 2026-08-17 A/B measured 14.61% at M = 6 and 15.64% at M = 8 in the BATCH spelling; this run measures 16.44% at M = 7 in the SEQUENCE spelling.
Three points, two harnesses, one monotone trend, and M = 7 lands exactly where its neighbours bracket it.
That is what the meeting-point argument predicted, and it is now measured rather than argued.

### What this does NOT establish

It does not say speculative decoding is faster end to end.
Drafting costs time and acceptance is fractional; arXiv 2607.17283's own measurement is that three of five configurations DECELERATE, and nothing here contradicts that.
This measures one half of the ledger - the verification step, 16.44% cheaper when routed - and the other half needs a draft model and an acceptance rate.

The timed block is also a fixed 7-token pattern replayed against a growing cache, where a real decoder verifies different drafted tokens each round.
The projections see the same shapes either way, which is what the routing decision turns on, but a full loop is what would settle the end-to-end question.

### What it licenses next, per section 4

The eligibility rule is worth changing under its own pre-registration: `_RoutedLinear._ineligible` should compute the flattened M and let `should_dispatch` decide, rather than refusing every sequence step before it asks.
That change re-opens a harness whose grid was re-measured on 2026-08-17, so it needs its own pre-registration and a re-run of the A/B to show the published grid is unmoved - the same discipline the `mx.clear_cache()` fix went through, and for the same reason.

A draft/verify loop is now worth building, which it was not before this run.
