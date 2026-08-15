# ADR 0015: The wide-tile qmv routes per shape and per bit width, from measurement, not from a default

Date: 2026-08-15
Status: Accepted

## Context

The kernel pack's wide-tile quantized GEMV (`kernelverify/pack/wide_qmv.py`) beats MLX's own quantized matvec over some range of tile widths M and loses outside it.
Until this ADR the range was one pair of numbers, `MIN_PROFITABLE_M = 5` and `MAX_PROFITABLE_M = 11`, applied to every shape at every bit width.
That pair was a stated default (ruling D3.1), not a measurement: it was chosen to be conservative while the pricing probe did not yet exist.

`bench/price_qmv_boundary.py` is that probe.
Its pre-registered rule lives in its module docstring, committed before any number existed: a cell counts as WIN only when the whole ratio interval clears 1.0 from below (`ratio_lo > 1`) on a round whose reference arm held steady, LOSS only when the whole interval sits under it, and REFUSED otherwise, which means undecided and never means measured-bad.
This ADR is the reading of its run.

## The run

`bench/results/qmv-boundary-pricing-2026-08-15.json`, 96 points, committed, sha256 `4a7c500fce9c…`.
Six dispatch shapes times M = 1..16, at 3 bits only, 7 rounds per cell, 5 ms minimum sample, canary spread capped at 1.5.
Shapes came from the pinned artifact's own config (`bench/.models/qwen3-4b-3bit-g64/config.json`), not from a hand-written list.
Machine: Apple M3 Pro, 12 cores, 36 GB, on AC, load1 1.34 before and 1.48 after, no thermal warning, under the machine-wide measurement lock and the 26 GB footprint budget.
Every timed cell was verified first by the pack's own correctness gate, so nothing here is a speed claim about an unverified kernel.

## The reading

Ratio is ours over MLX, so above 1.0 is a win.
`W` = WIN, `L` = LOSS, `R` = REFUSED (interval straddles 1.0).

| shape | site | M1-M3 | M4 | M5-M9 | M10 | M11 | M12-M16 |
|---|---|---|---|---|---|---|---|
| 4096x2560 | q_proj | L | L 0.966-0.994 | W 1.019-1.312 | L 0.728-0.813 | L 0.958-0.997 | L |
| 1024x2560 | k_proj/v_proj | L | R 0.992-1.058 | W 1.029-1.323 | L 0.820-0.846 | R 0.987-1.018 | L/R |
| 2560x4096 | o_proj | L | R 0.966-1.066 | W 1.027-1.381 | L 0.758-0.779 | L 0.876-0.935 | L/R |
| 9728x2560 | gate_proj/up_proj | L/R | R 0.980-1.009 | W 1.025-1.271 | L 0.764-0.976 | R 0.925-1.008 | L |
| 2560x9728 | down_proj | L/R | R 0.993-1.012 | W 1.035-1.301 | L 0.933-0.986 | L 0.757-0.773 | L |
| 151936x2560 | lm_head | L | W 1.018-1.028 | W 1.058-1.377 | W 1.029-1.050 | L 0.895-0.925 | L |

Three findings, in order of consequence.

**The shipped default was routing measured losses.**
M = 10 is a LOSS at all five non-lm_head shapes, as low as 0.728x at q_proj, and M = 11 is a LOSS or REFUSED at every shape including lm_head.
The 5..11 window was therefore shipping a regression at two tile widths per shape, on the strength of a number nobody had measured.

**The winning set is not the same at every shape.**
lm_head wins one width further than the rest (M = 10, 1.029-1.050), because at 151936 rows MLX's second weight pass costs more than our single pass for longer.
No single pair of bounds can say that, which is why the routing decision is now a table keyed by shape rather than a pair of integers.

**The wins are real and large where they exist.**
M = 7 peaks at 1.27x-1.35x across the six shapes; the whole 5..9 band clears 1.02x at every shape.

## Decision

Routing is now a table, `kernelverify/pack/routed_windows.py`, keyed `(bits, d_out, d_in)` and holding the SET of winning tile widths.

| key | site | routed M |
|---|---|---|
| (3, 4096, 2560) | q_proj | 5, 6, 7, 8, 9 |
| (3, 1024, 2560) | k_proj/v_proj | 5, 6, 7, 8, 9 |
| (3, 2560, 4096) | o_proj | 5, 6, 7, 8, 9 |
| (3, 9728, 2560) | gate_proj/up_proj | 5, 6, 7, 8, 9 |
| (3, 2560, 9728) | down_proj | 5, 6, 7, 8, 9 |
| (3, 151936, 2560) | lm_head | 5, 6, 7, 8, 9, 10 |

Five properties make the table a record of evidence rather than a second opinion about it.

The cells are derived from the committed recording at import, so the shipped routing and the recorded evidence cannot drift apart; nothing is hand-written except the exclusions and the pins.
The recording is pinned by sha256, so re-running the probe over the same filename cannot silently re-derive shipped routing - the re-pin has to be deliberate.
The kernel body is pinned by sha256 (`WIDE_QMV_MSL`), because a window is a claim about the kernel that was priced and says nothing about an edited one.
The launch config is pinned as data (`ROWS_PER_SIMDGROUP`: R = 4 through M = 10, R = 2 from M = 11) and every point of the recording is checked against it, because the same M at a different R was never measured.
`should_dispatch(m, bits, d_out, d_in)` takes required positional arguments and an absent key routes nowhere, so a stale one-argument caller fails loudly and an unpriced shape gets the honest answer (no evidence) rather than a default.

Two rulings shape what the table contains.

**D1: keyed by bit width.**
The run priced 3-bit only.
4-bit is unpriced, so it routes nowhere and MLX serves it, until a 4-bit pricing run lands (queued in TODOS.md).

**D2: lm_head M = 4 is measured WIN and is still not routed.**
The pre-registration this run was read under permits the routed window to NARROW only: an unevidenced cell is dropped, a new cell is not added.
Every window above is strictly inside the old 5..11 default, so adopting them is pure narrowing and needs no new authority.
M = 4 is outside it, so adopting that one cell would be reading a widening out of a rule that pre-registered narrowing - a post-hoc rule change worth 2.3% at one shape.
It is recorded in `EXCLUDED_CELLS` with the ruling, and becomes routable only through the widening rule now pre-registered in the probe's docstring, dated before any run that could adopt it: WIN again in a later run, contiguous with the existing window, and adopted together with gate coverage of the cell.

## What this ADR does not claim

The batch-10 loss and the batch-11 loss are not the same phenomenon, and neither is diagnosed here.
M = 10 loses at R = 4, the same launch config as the M = 5..9 wins, so it is a cliff inside one configuration.
M = 11 is the first width at R = 2, so its loss is confounded with the config switch and cannot be attributed to tile width alone.
Untangling both is a separate measurement (an R x M sweep), and until it runs, no cell at M >= 10 may be re-argued from these numbers.

The run priced one bit width, one model's dispatch shapes, and one machine.
Nothing here licenses routing at another width, another shape, or another chip.

## Consequences

`kernelverify/pack` now reads a committed file under `bench/results/` at import time, which is a dependency from the library package onto the evidence directory.
This is accepted deliberately: the alternative is a hand-copied table, and a hand-copied table is exactly the drift this ADR exists to prevent.
The file is committed, so the dependency travels with the repo; a missing or altered recording fails loudly at import rather than routing on stale numbers.

Two deviations from the approved plan, both deliberate, both recorded here so the plan and the code do not disagree silently.
The plan called for a generator script emitting a committed table; the table is derived at import instead, which removes the failure mode where the generator and its output disagree.
The plan called for the derivation to assert each window contiguous; the table stores sets, which can represent a hole, and a test asserts today's windows are in fact contiguous - so a future hole reaches a person as a finding rather than being either crashed on or shipped.

The certificate emitter no longer hardcodes its dispatch-scope prose; it asks the table, so a certificate cannot describe a boundary the pack does not enforce.
Certificates already on disk still carry the old prose and are stale until the next gate run re-emits them.
