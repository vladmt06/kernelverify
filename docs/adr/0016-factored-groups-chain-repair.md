# ADR 0016: The factored-groups member sums each group as a chain; ten shipped false positives close; K stays 4

Date: 2026-08-15
Status: Accepted

## Context

ADR 0014 left one thing unresolved and said so: after the out-of-contract cells were withheld, four batch-1 cells kept an admissible-only demand above the shipped K = 4, every one bound by the device member `device-factored-simd` on constant-rows float32.
The same day's outside review of the contract lane found what that residual actually was.
It was not a membership gap and not a K gap; it was one CPU member being more accurate than any real kernel of its kind.

`factored-groups` stands for the int-accumulate quantized GEMV kernel: per group it forms `s * sum(x*q) + b * sum(x)`.
It formed both per-group sums with numpy's pairwise reduction.
On a constant row that reduction is EXACT: 64 identical fp32 values halve down a power-of-two tree with no rounding at all.
Every real int-accumulate kernel sums the group as a chain and rounds 63 times.
So on constant rows the member was most accurate exactly where the device kernels are least accurate, the leave-one-out floor it feeds was too tight there, and the shipped tolerance `max(base, 4 * floor)` flagged a correct, C1-admissible device kernel.

The count, from the committed serving evidence (`bench/results/quant_serving_adequacy.json`, sha256 `54d0ad5c…`): 10 of 1,536 records, all at batch 1, constant-rows, float32, worst overshoot 1.743x.
Those are shipped false positives against the one thing the verifier promises never to flag.

An earlier ruling on this residual (a companion member added beside `factored-groups`) was withdrawn before implementation when the review showed it would drive the demand statistic to ~1 without touching the false positives; the ruling that stands is this repair.

## The repair

Both per-group sums in `member_factored_groups` are now explicit fp32 chains, accumulated left to right the way the kernel's inner loop runs; only the cross-group combination stays vectorized, which is the freedom `factored-serial` varies.
The member is checked against a scalar-loop rewrite in `tests/test_quant_contract_members.py` that shares no numpy reduction with the code it checks, at 3 and 4 bits, both activation dtypes, three input modes.

The chain costs time the vectorized reduction did not: at lm_head each of the 64 steps touches a 389 MB working array.
The serving harness runs it in output-row blocks (1 MB measured best: 24.5 s whole-width, 6.4 s blocked, 20 s again at 16 MB).
The block is sound where the memory amendment's row-chunked matmul was not: the chain is elementwise in the row index with no reduction across it, proven bit-identical at every block size (`test_chunked_factored_groups_is_bit_identical`), and the member's own BLAS term stays whole.

Two invariants are pinned by test.
The five untouched CPU members recompute bit-identically on a 64-record sample of the committed serving evidence, so the floor moved for this reason and no other.
And the verdict cache's fingerprint now carries `QUANT_ENSEMBLE_VERSION` beside the member labels, because a member's arithmetic changing under an unchanged name is exactly the change a label-keyed cache cannot see; without it `score_oracles` would have read back verdicts computed against the pairwise member.

## The ten false positives, closed

The shipped tolerance against the held-out device member, on the committed serving records, recomputed under the repaired member with every other member's value taken from the record:

| Cell (batch 1, constant-rows, float32) | Before | After |
|---|---|---|
| 1024x2560 normal-0.02 s0 | 1.743 | 0.256 |
| 9728x2560 heavy-tailed s1 | 1.246 | 0.263 |
| 1024x2560 normal-0.02 s1 | 1.205 | 0.225 |
| 1024x2560 heavy-tailed s1 | 1.201 | 0.247 |
| 4096x2560 heavy-tailed s0 | 1.141 | 0.235 |
| 4096x2560 normal-0.02 s0 | 1.088 | 0.222 |
| 9728x2560 normal-0.02 s1 | 1.078 | 0.218 |
| 9728x2560 normal-0.02 s100 | 1.016 | 0.223 |
| 4096x2560 heavy-tailed s100 | 1.007 | 0.265 |
| 2560x4096 normal-0.02 s101 | 1.000 | 0.229 |

Every one is inside tolerance with a margin of about 4x.
Over all 1,536 serving records the held-out device member's worst ratio to the shipped tolerance falls from 1.743 to 0.328.

## ADR 0014's residual, closed

The four in-contract cells ADR 0014 recorded as still demanding more than K = 4, re-read under the repaired member (the whole 1,536-record grid was recomputed for this, one `(shape, draw, seed)` block at a time, with every untouched member checked bit-identical against its recorded value):

| shape | ADR 0014 demand | Under the repair |
|---|---|---|
| 1024x2560 | 6.973 | 1.022 |
| 9728x2560 | 4.947 | 1.052 |
| 4096x2560 | 4.563 | 0.941 |
| 2560x4096 | 4.001 | 0.915 |

All four were bound by `device-factored-simd` on constant rows, and all four now sit at roughly 1, which is what a member that is one legitimate rounding order among several should read.
The admissible-only pooled demand over the whole serving grid falls from 6.973 to 3.120.

## The device grid, re-run

`bench/calibrate_quant_device.py` was re-run under an amendment to its pre-registration, committed before the run (`f98c2e2`), stating what each outcome means when a member has deliberately been made less accurate: K is expected to fall and a fall is adopted only if the gates pass at it; K rising is a stop; the detection price is named cell by cell.
The records are committed at `bench/results/quant_device_adequacy.json`, sha256 `c7197a1af4d76c6f741ada7271a981f85d9472f0ad649223c93df668eba70b3f`.

G0 held: `canonical_quantize` bit-exact against `mx.quantize` at every width.
The trigger stayed clear: 0 of 192 exceedances at every width against the CPU-calibrated K = 3 floor.

| bits | ADR 0012 calibration / independent | This run calibration / independent | Binding member, this run |
|---|---|---|---|
| 2 | 2.792 / 2.755 | 2.766 / 2.604 | factored-groups, heavy-tailed corpus-scale |
| 3 | 2.087 / 3.156 | 1.476 / 1.907 | dequant-reversed |
| 4 | 2.667 / 2.712 | 1.783 / 1.840 | dequant-reversed |
| 8 | 3.901 / 3.500 | 1.394 / 2.061 | factored-groups, heavy-tailed constant-rows |

The demand fell at every width, and fell most exactly where ADR 0012 said `device-factored-simd` on constant rows was binding.
The two routes that landed K = 4 in ADR 0012 - the bits = 8 calibration demand of 3.901 and the bits = 3 independent-draw miss at 3.156 - are gone: 1.394 and 1.907.
Peak demand across the device grid is 2.766, and the harness therefore printed `shipped K: 3.0` and evaluated its gates there:

| bits | records | G1 false positives | G2 worst margin | G3 distinct per class | verdict |
|---|---|---|---|---|---|
| 2 | 192 | 0 | 161.4x | 3 / 2 / 3 | ADEQUATE |
| 3 | 192 | 0 | 278.5x | 3 / 2 / 3 | ADEQUATE |
| 4 | 192 | 0 | 266.4x | 3 / 2 / 3 | ADEQUATE |
| 8 | 192 | 0 | 272.1x | 3 / 2 / 3 | ADEQUATE |

## The decision: K stays 4.0

Two quantities live on the serving grid and they do not agree, so both are recorded.

**The K-derivation demand, over the nine-name membership** (ADR 0012's own rule: any member's error against the floor of the OTHER members, full membership) falls from 6.973 to **3.120**, bound by `factored-groups` at 2560x9728, batch 2, normal-0.02 weights, constant-rows float32.
The K grid is `(1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)`, so the smallest value covering 3.120 is 4.0.

**The device harness's `shipped K: 3.0` is a reading over its own three shapes** - 512x512, 1024x4096, 4096x4096 - and 2560x9728 is not among them.
Taking that line as authority would have shipped a K that the next serving run refuses, at a cell this ADR could have named in advance.
The gates are not the check that catches it: they pass at K = 3 on the serving grid, all 24 cells ADEQUATE, zero G1 false positives, zero G2 failures, worst fault margin 260.6x.

K stays 4.0. `K_QUANT` and `K_SHIP` are unchanged, and this ADR is now the reason they are 4.0; ADR 0012's reason (the device class demanding 3.901) no longer holds at 1.394.

## What the repair fixed, measured the way the verifier judges

The shipped tolerance is `max(base_tol, K_QUANT * floor)` where the floor runs over the SIX CPU members in `quant_contract.ENSEMBLE`; the device members are calibration instruments and need a GPU, so they are not in the shipped floor.
Judged that way, over all 1,536 serving records, counting every in-contract candidate (the three device members and the held-out implementations, with ADR 0014's out-of-contract cells withheld):

| | live false positives at K = 4 |
|---|---|
| Before the repair | **10** (worst 1.743x) |
| After the repair | **0** |

That is the product claim of this ADR, and it is the whole reason the repair was made.

## The open question the repair exposes

The same records carry a second reading that moved the other way.
The leave-one-out spread over the six CPU members alone - the floor the shipped tolerance actually divides by - rises from **4.076 to 7.561**, bound by `factored-groups` itself at 9728x2560, batch 1, constant-rows float32.

This is an ensemble-adequacy statistic, not a live flag, and the distinction is exact: the shipped floor CONTAINS `factored-groups`, so a candidate that rounds like it is judged against a floor that already holds its own error and lands at 0.25 of tolerance by construction.
What the number says is that the int-domain class is now carried by one member that sticks far out from the rest - `factored-serial`, the next int-domain member, reads 1.692 - because the repair moved `factored-groups` from the most accurate member on constant rows to the least.

No K in the grid covers 7.561, and raising K to cover it would undo the tightening ADR 0009 and ADR 0012 fought for.
The standing repair order says membership before K, and this is a membership question: whether the int-domain class needs a second member that rounds like a real kernel, and if so which real kernel it stands for.
This ADR records it as OPEN with its numbers rather than resolving it, because it is a contract-population decision and belongs to a renegotiation, not to the tail of a repair.
It is not blocking: at K = 4 the shipped verifier flags no correct kernel on any record measured.

## The detection price, named

The pre-registration required every fp16-dequant boundary cell that flips caught -> not-caught to be named, not counted.
The count is three, out of 384 float32 cells across the four widths, and every one is at the smallest shape on heavy-tailed constant rows - the same regime the false positives lived in:

| bits | shape | draw | seed | mode | ratio before | ratio after |
|---|---|---|---|---|---|---|
| 2 | 512x512 | heavy-tailed | 100 | constant-rows | 1.248 | 0.953 |
| 3 | 512x512 | heavy-tailed | 0 | constant-rows | 1.640 | 0.835 |
| 4 | 512x512 | heavy-tailed | 0 | constant-rows | 1.963 | 0.980 |

The "before" column was recomputed rather than remembered: the device grid overwrites its cache on every run and ADR 0012's records were never committed, so the pre-repair member was re-run verbatim (from `0052f7b`) over the same rng stream, with the untouched member and the repaired member both cross-checked bit-exactly against this run's 768 records first.
The boundary table moves from ADR 0012's K = 3 CPU-floor column of 39 / 96 / 96 / 96 to 38 / 95 / 95 / 96 at bits 2 / 3 / 4 / 8; float16 activations stay 0 / 96 everywhere.

This is the honest shape of the trade.
The member stopped being unrealistically exact on constant rows, so on constant rows the tolerance is wider both for correct kernels (ten false positives closed) and for one boundary implementation (three of 384 cells stop being flagged, each now sitting at 0.84-0.98 of tolerance).

## The battery, re-scored

`score_oracles` rebuilt its verdicts under the new fingerprint (the old cache was refused, which is the fingerprint fix doing its job).
The shipped policy holds 96.6% / 98.3% / 100.0% / 100.0% at B = 4 / 8 / 16 / 32 over the 58 viable faults, the 100.0% cells exact (zero misses over 40 seeded runs), 100% on the corpus subset from B = 4, the same 7 equivalent mutants and the same two small-budget survivors as ADR 0008.
The repair changed nothing the headline claim rests on.

## Consequences

- `K_QUANT` = 4.0 and `K_SHIP` = 4.0 are unchanged; this ADR replaces ADR 0012 as the reason.
- The shipped K is the maximum demand over every harness's shapes, not any one harness's own reading. Nothing in the code enforces that today: `calibrate_quant_device` prints its `shipped K` from three shapes with no knowledge of the serving grid. Recorded in TODOS.md.
- The int-domain class rests on one outlying member (leave-one-out 7.561 against the shipped floor); open, recorded above, not blocking.
- ADR 0012's freeze line is amended: reruns of the device grid must reproduce THIS run's tables; ADR 0012's tables are the reading under the pairwise member and no longer reproduce.
- ADR 0013's records are unchanged (they are measurements); their interpretation under the repaired member is derived, not re-measured, and the derivation is in `tests/test_quant_contract_members.py`.
- ADR 0014's residual is closed by this ADR; its out-of-contract rulings (B1 fp16, B16 fp16) are untouched, since they concern MLX's kernels rather than this member.
- The standing repair order (membership before K) is unchanged, and this ADR followed it: nothing was added to the membership because nothing was missing from it; one member was wrong and was corrected.

## What was NOT re-run, and why

`calibrate_k` measures the unquantized contract (`kernelverify/tolerance/contract.py`) and never evaluates the quantized ensemble; nothing in it depends on this member.
It was on the plan out of caution and is struck from it by reading the code.
The cold serving grid (~45 min of GPU) was not re-run either: every quantity it would re-measure is either a device dispatch that did not change or a CPU member that either did not change (proven bit-identical) or is the repaired member, recomputed on all 1,536 records from the same rng stream.
The harness's checkpoint fingerprint hashes `quant_contract.py`, so it will refuse the old checkpoint whenever it next runs cold, by design.
