# ADR 0017: G1 is scored out of sample, K comes from calibration, and the held-out draw did not fire

Date: 2026-08-16
Status: Accepted

## Context

The audit of 2026-08-15 found that the verifier's headline gate could not fail.

G1 says the shipped tolerance flags no correct implementation, and every ADR since 0009 reported `G1 = 0` as evidence that it holds.
It was not evidence.
`k_demand` maximised `heldout / floor` over a set of records, `main` then chose `k >= demand`, and `gates()` scored G1 over those same records at that k.
A member cannot exceed a tolerance that was just set to cover its worst case, so G1 = 0 was algebraically guaranteed and carried no information about the verifier at all.

This is the second time this project has found a self-scoring gate.
The first was ADR 0009's leave-one-class-out, which as a gate fails every ensemble whose classes are genuinely distinct, and hardest when they are most distinct.
Both were caught the same way: by asking what result the check is capable of producing, rather than by reading its output.

## The amendment, pre-registered before the code moved

`bench/calibrate_quant_device.py`'s docstring carries the amendment, committed alone at `66a4147` with no code in the same commit, on the standing rule that a measurement's rule is fixed before its run.

- K is chosen from the CALIBRATION draw alone (seeds 0 and 1).
- The INDEPENDENT draw (seeds 100 and 101) tests that K and never sets it.
- If the independent demand exceeds K, that is an INDEPENDENT MISS: the cell is named, the records are persisted, and the run exits 1 for a human to renegotiate.
- G1 is scored on the independent draw only. G2 and G3 stay on all records.
- Value-duplicate members are collapsed before any demand is taken, and both readings are printed and persisted.

The miss branch REPLACES ADR 0012's pre-registered rule of taking the next covering grid value and reporting the miss.
Rolling K up to cover the held-out draw would make G1 true by construction on that draw, which is the exact defect being removed.

## The run

`bench/calibrate_quant_device.py`, 2026-08-16, records committed at `bench/results/quant_device_adequacy.json`, sha256 `00545b2aa1a2ff889f67ec24...`.
G0 held: `canonical_quantize` is bit-exact against `mx.quantize` at every width.

| bits | calibration demand | independent demand | independent > calibration | miss at K = 3.0 |
|---|---|---|---|---|
| 2 | 2.766 | 2.604 | no | no |
| 3 | 1.476 | 1.907 | YES | no |
| 4 | 1.783 | 1.840 | YES | no |
| 8 | 1.394 | 2.061 | YES | no |

The device-grid K is 3.0, being the smallest grid value covering the calibration demand of 2.766, bound by `factored-groups` at 1024x4096 on heavy-tailed weights, corpus-scale float32.
The verifier continues to ship `native_ops.K_QUANT = 4.0`, for the reason ADR 0016 records: the serving grid demands 3.120 at 2560x9728, a shape this harness does not measure.
Those two numbers are now separate keys, `k_grid` and `k_shipped`, because reading one as the other is the mistake ADR 0016 nearly shipped.

Value-duplicate collapse changed no decision: `all` and `distinct` agree at every width, since `lut-gather == dequant-pairwise` and `device-dequant-loop == dequant-pairwise` are pinned at a leave-one-out ratio of 1.0 and can never bind.
The reading is kept because a duplicate that stopped being a duplicate would show up here first.

Gates at K = 3.0: ADEQUATE at every width, zero G1 false positives on the independent draw, G2 worst fault margin 161.4x to 278.5x, G3 three distinct classes at 3/2/3 members, zero equivalent faults.

## The result, stated as what it is

At three of four widths the held-out draw demanded MORE than calibration.
Under the old rule those three widths would have had K set by the held-out draw, and G1 would then have been guaranteed to pass on it.
Under the amendment K was set by calibration alone and the held-out draw's higher demands - 1.907, 1.840 and 2.061 - still fell under it.

So the check that can now fail, did not.
That is the first evidence this project has that its K rule survives a draw it did not fit, and it is worth exactly as much as the draw is independent.

## What this ADR does NOT claim

`G1 = 0` is still not the interesting number, and this ADR does not report it as one.
The miss branch exits before `gates()` is reached, so once a run gets far enough to score G1 on the independent draw, no independent record can exceed the tolerance: the quantity `k_demand` maximises and the quantity `gates` thresholds are the same quantity.
G1 as computed remains true by construction.
The falsifiable check is the miss branch, and the headline of any future run under this rule is `INDEPENDENT MISS: none` or the named cell, never `G1 = 0`.
The code says this at the call site rather than in a docstring, so a reader meets it where the value is produced.

The held-out draw is also weaker than it reads.
It differs from calibration by SEED only: same three shapes, same two weight distributions.
A draw that varied shape and weight distribution could disagree in ways this one structurally cannot, and TODOS.md carries that as queued work with its cost.
Until it runs, "the rule survived a held-out draw" means "survived two different seeds".

## Consequences

- The device harness prints `device-grid K` and persists `k_grid` beside `k_shipped`; the key `k_ship` is gone.
- `cover()` lives once, beside `K_GRID` in `bench/calibrate_quant_bits.py`, and raises above the grid rather than clamping, so a demand no K can cover is a KILL rather than a silent ceiling.
- `choose_k({})` returns the CPU-calibrated K with no misses, and a run where every width fails G0 prints `no usable width` and returns 1 rather than raising.
- ADR 0012's roll-up rule is superseded for this harness.
- The 2026-08-15 device records this ADR replaces were themselves preserved; the pre-repair comparison in ADR 0016 was re-derived against these records and still reproduces every number.
