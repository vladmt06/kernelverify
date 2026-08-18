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

Filled after the run.
