# ADR 0011: What a certificate asserts, and what it refuses to

Date: 2026-08-14
Status: Accepted

## Context

The certificate is the product's trust artefact: it ships beside a kernel and lets anyone re-check the claim.
Its failure mode is not being wrong, it is being *read as stronger than the verification behind it*, which is precisely the failure this project exists to criticise in other people's benchmarks.

Three measurements from this week's lanes forced the design, and each is load-bearing rather than precautionary.

## Decision 1: byte-bound and protocol-bound are separate blocks, not separate sentences

A certificate carries hashes (which translation unit, which catalogue, which harness) and it carries assertions (these verdicts hold).
These are different kinds of claim and a reader will conflate them if the document lets them.

- The byte-bound block identifies WHAT was verified and explicitly asserts nothing about output values.
- The protocol-bound block asserts that re-running the published protocol reproduces the VERDICTS.

Byte-exactness is not achievable on a GPU: reduction order, compiler version and power state all move the last bits legitimately, and the runner lane measured one fixed shape drifting 131.7 to 93.1 us between runs minutes apart.
A certificate claiming bit-identical output would therefore be false on its face, so the structure prevents the claim rather than disclaiming it in prose someone can skim past.

## Decision 2: every contract clause names the mechanism that established it

The admissible-implementation contract has six clauses.
They are not all enforceable the same way, and a certificate that listed them uniformly as "verified" would be claiming a check it never ran.

C1 is the measured case, and the boundary follows accumulation topology rather than dtype alone:

| Kernel shape | fp32 activations | fp16 activations | Mechanism at fp16 |
|---|---|---|---|
| Sequential accumulation (naive) | 280/280 separated | 192/280 separated, up to 23x over tolerance | numeric battery |
| Tree-reduced accumulation (simdgroup) | 280/280 separated | 0/280 separated | source attestation |

The physics: tree error grows with reduction DEPTH and stays inside the legitimate fp16 rounding envelope the conditioned floor must allow, while sequential error grows with reduction LENGTH and punches past it.
So at fp16 activations the tree class has zero input-space signal at every case, meeting the ADR 0008 bar, and is attested by reading the kernel's accumulator dtype from source.
That is trivially checkable for pack-authored kernels and honestly labelled for anyone else's.

The consequence generalises: `mechanism` is a required field per clause, one of `numeric-battery` or `source-attestation`, and an attestation without evidence is rejected at construction.

## Decision 3: extraction provenance has three tiers, and the top one does not claim byte identity

MLX generates the real translation unit at run time; the runner lane established that it can be captured through the framework's own documented hook, with fd-level redirection because the dump is written by the C++ layer straight to the file descriptor.

| Tier | Meaning |
|---|---|
| `extracted-behaviorally-validated` | captured from the framework's generator and reproduces its outputs under this protocol on the pinned toolchain |
| `extracted-unvalidated` | captured, but the reproduction check has not run on this toolchain version |
| `wrapper-assembled` | we assembled the unit ourselves; it attests our wrapper, not the shipped program |

The top tier deliberately stops short of byte identity, for two reasons the runner lane raised and I accepted: numerically equivalent but non-identical programs pass a behavioural comparison by construction, and compile options never appear in generated source at all.
Compile options are therefore recorded as certificate inputs rather than implied by the source hash, and the mirroring decision (Metal defaults fast, MLX documents safe) is recorded with them.

What a source hash structurally cannot cover is listed in the certificate itself: host-side contiguity copies the framework may insert before launch, launch configuration beyond the declared spec, and those compile options.

## Decision 4: the audit gates emission

`audit()` runs before any certificate is written and refuses on: a failed case, fewer cases run than the stated budget, an unvalidated extraction claiming the top tier, a wrapper-assembled unit not declared plainly, a speed claim with no scope, a binding speed claim with no sampling group, and a missing C1 attestation.
`emit()` raises rather than writing, and writes nothing at all on refusal.

Speed claims carry their dispatch range, never the operator: the pack lane measured a 2-bit kernel losing at batch 1-2 while winning at batch 5-11, so a claim scoped to "quantized matvec" would be false at the low end of its own operator.

## Consequences

The certificate can be re-checked by anyone: the byte-bound block says exactly what to fetch, the protocol-bound block says exactly what to re-run, and the validity block says where the result binds (matching chip generation and toolchain) and where it is advisory.
An advisory failure is a new matrix row to investigate, not a refutation - which keeps contributed rows useful without letting them overturn a measurement we control.

Deliberately not decided here: whether a certificate should carry the pack kernel's `should_dispatch` boundary as machine-readable policy rather than prose scope.
That waits until the binding run gives the profitability boundary a measured value instead of a placeholder.
