# ADR 0018: Serving dispatch uses flattened row count at every input rank

Date: 2026-08-18
Status: Accepted

## Context

The serving interception refused every 3-D input whose sequence length exceeded one before it computed M or consulted the measured routing table.
It called that refusal `prefill-L{L}` and justified it by saying the kernel and its certificates were decode-shaped.

That distinction does not exist at the kernel boundary.
`_fused` applies `x.reshape(-1, d_in)` before dispatch, so `(1, 7, d_in)`, `(7, 1, d_in)` and `(7, d_in)` present the same seven rows to the same M = 7 specialization.
Measured on 2026-08-18, the batch and rank-2 spellings routed while the sequence spelling alone fell back with `prefill-L7`.

The separate speculative-verification spike then admitted the sequence spelling locally and measured the full seven-token verification step 16.44% faster than stock against a 0.327% noise floor.
Its committed record is `docs/research/2026-08-18-spec-verify-meeting-point-spike.md`.
Section 4 of `docs/research/2026-08-15-sub4bit-serve-findings.md` registers the serving-rule change and its exact scope.

## Decision

`_RoutedLinear._ineligible` computes M as the flattened row count for every input rank and leaves `should_dispatch` as the only judge of M.
The rank-specific prefill refusal and its whitelist prefix are removed.
The speculative-verification spike uses the shared serving patch instead of maintaining a second interception with the same rule.

## Consequences

A sequence step with flattened M = 5 through 9 now routes, including a speculative decoder's K = 4 through 8 verification pass.
A prefill of exactly 5 through 9 tokens also routes because the measured table admits those flattened projection calls.
Every M outside the routed window still falls back with `m-{M}-outside-dispatch-{d_out}x{d_in}`, including real prefill chunks of tens to hundreds of tokens.

The two M = 7 rank spellings are checked for bit-identical routed output, so one M = 7 kernel certificate covers both presentations of the same rows.
This ADR does not reinterpret or update the published A/B grid in Sections 9 and 10 of the findings document.
That grid is re-measured under its own pre-registration before this lane merges; that pre-registration is not written yet, so this ADR asserts the plan for it rather than the existence of it.
The spike's 16.44% is the only measurement behind this decision, and the code that produced it is deleted by the same change, so its reproduction under the shared patch is a separate check and this ADR is provisional until that check and the A/B re-run land.
