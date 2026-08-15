# ADR 0013: Batch 1 at every serving shape demands more than K = 4; interpretation stops at the pre-registered branch

Date: 2026-08-15
Status: Accepted

## Context

ADR 0012 shipped the quantization tolerance `max(base, 4 * floor)` over the three-class, nine-name floor, measured on the calibration shapes.
`bench/calibrate_quant_serving.py` asks whether that tolerance covers the six shapes Qwen3-4B actually dispatches under mlx-lm, per (shape, batch) cell, at bits = 3.
The pre-registered rule lives in that harness's docstring, committed before any measurement (f563bce); this ADR is the reading of its run.

The run: `bench/calibrate_quant_serving.py --resume`, completed 2026-08-15 06:49 from this worktree.
The log is `bench/.cache/calib_serving3.log`; the records are `bench/.cache/quant_serving_adequacy.json` (1.67 MB; 1,536 grid records plus the per-shape probe records).
The harness exited with code 1 through its pre-registered refusal branch, which is the correct exit for this run, not a malfunction.

The shapes came from the pinned artifact itself, present and verified: provenance `pinned-artifact:/Users/vlad/kernelverify/bench/.models/qwen3-4b-3bit-g64`, uniform bits = 3, group_size = 64 on every projection including the tied-embedding lm_head path, hashes recorded in the JSON.
The docstring's conditional-attestation clause is therefore discharged: this reading binds to the verified artifact, not to the derived-config fallback.
ADR 0012's step-2 trigger quantity (device members against the retired CPU-calibrated K = 3) was deliberately not re-run; the records carry every member and held-out error, so it stays computable post hoc from the JSON.

## Memory safety: the run that survived

Three earlier attempts died to Jetsam inside STEP 2, at 66.7, 69.4 and 39.5 GB Python footprints on the 36 GB machine.
This run executed under this lane's commit 58209b8: every measurement iteration in its own child process, the 24 GB phys_footprint budget with its distinct refusal exit, the machine-global single-instance lock with the available-memory gate, and the Metal buffer-pool fix in `kernelverify/runners/device.py` beneath them.
The parent closed the run at 0.05 GB peak (its final self-report; per-cell parent footprint readings stayed at 0.03-0.04 GB), the largest measurement child peaked at 22.42 GB under the budget, and none of the distinct memory-refusal exits fired.

## One step was resumed, and it is named: `continuity`

`resumed_steps` in the JSON is `['continuity']`.
The checkpoint fingerprint (provenance plus serving shapes) validated, so `--resume` reused STEP 0 from the pre-crash checkpoint: the continuity anchor's 64 records, which reproduced the cached ADR 0012 records at (1024, 4096) exactly, were measured by the earlier process and were NOT re-measured in this one.
Every other step (the batch-regime probe, the full main grid, and everything computed from them) was measured fresh in this process.

## Clean preliminaries

G0 held everywhere: `canonical_quantize` stayed bit-exact against `mx.quantize` on every probe artefact and on all 48 main-grid checks.
The batch-regime probe found three arithmetic-regime components at every shape, with batch 1 alone in its own component at all six shapes; the split falls at 11/12 for q_proj, kv_proj and o_proj and at 9/10 for gate_up_proj, down_proj and lm_head.
No batch joined the grid, so the grid batches stayed {1, 2, 8, 16}; batches 9-11 (q_proj, kv_proj, o_proj) and batch 9 (the other three shapes) are weak-prefix covered, attested by measured row-invariance rather than direct sampling, reported per the pre-registered visibility line.

## STEP 4: the K demand, and the miss

Per-cell K demand at full nine-name membership against the shipped K = 4, shown as calibration draw / independent draw.
A cell misses when either draw exceeds 4.

| shape | B1 | B2 | B8 | B16 |
|---|---|---|---|---|
| 1024x2560 | 78.479 / 3.806 MISS | 2.200 / 1.863 | 2.228 / 2.176 | 1.879 / 1.614 |
| 151936x2560 | 105.021 / 93.925 MISS | 1.990 / 2.096 | 2.053 / 1.942 | 1.845 / 1.676 |
| 2560x4096 | 97.497 / 195.198 MISS | 1.532 / 1.983 | 1.803 / 1.711 | 1.574 / 1.876 |
| 2560x9728 | 91.841 / 160.037 MISS | 2.471 / 2.388 | 2.048 / 2.329 | 1.746 / 1.700 |
| 4096x2560 | 170.510 / 4.007 MISS | 1.758 / 1.833 | 1.503 / 1.965 | 1.435 / 1.657 |
| 9728x2560 | 78.767 / 84.753 MISS | 1.611 / 2.005 | 2.462 / 1.802 | 1.509 / 1.835 |
| pooled | 170.510 / 195.198 | | | |

The verdict, exactly as the harness printed it:
DEMAND MISS at [('1024x2560', 1), ('151936x2560', 1), ('2560x4096', 1), ('2560x9728', 1), ('4096x2560', 1), ('9728x2560', 1)].
Batch 1 misses at all six E2E serving shapes: five shapes miss on both draws (at 4096x2560 the independent draw only just, at 4.007), and 1024x2560 misses on the calibration draw alone (independent 3.806).
The pooled binding cases are `heldout mlx-on-device @ 4096x2560 heavy-tailed s1 constant-rows float16` (calibration) and `heldout mlx-on-device @ 2560x4096 heavy-tailed s100 constant-rows float16` (independent).

## STEP 5: gates at K = 4, width-pooled fault equivalence

| cell | records | G1 FP | G2 worst margin | G3 classes | verdict |
|---|---|---|---|---|---|
| 1024x2560 B1 | 64 | 2 | 268.6x | 3/2/3 | INADEQUATE |
| 1024x2560 B2 | 64 | 0 | 304.7x | 3/2/3 | ADEQUATE |
| 1024x2560 B8 | 64 | 0 | 314.2x | 3/2/3 | ADEQUATE |
| 1024x2560 B16 | 64 | 0 | 290.0x | 3/2/3 | ADEQUATE |
| 151936x2560 B1 | 64 | 4 | 302.1x | 3/2/3 | INADEQUATE |
| 151936x2560 B2 | 64 | 0 | 300.9x | 3/2/3 | ADEQUATE |
| 151936x2560 B8 | 64 | 0 | 323.5x | 3/2/3 | ADEQUATE |
| 151936x2560 B16 | 64 | 0 | 335.2x | 3/2/3 | ADEQUATE |
| 2560x4096 B1 | 64 | 8 | 286.3x | 3/2/3 | INADEQUATE |
| 2560x4096 B2 | 64 | 0 | 279.5x | 3/2/3 | ADEQUATE |
| 2560x4096 B8 | 64 | 0 | 301.2x | 3/2/3 | ADEQUATE |
| 2560x4096 B16 | 64 | 0 | 288.7x | 3/2/3 | ADEQUATE |
| 2560x9728 B1 | 64 | 6 | 285.1x | 3/2/3 | INADEQUATE |
| 2560x9728 B2 | 64 | 0 | 270.5x | 3/2/3 | ADEQUATE |
| 2560x9728 B8 | 64 | 0 | 282.1x | 3/2/3 | ADEQUATE |
| 2560x9728 B16 | 64 | 0 | 301.5x | 3/2/3 | ADEQUATE |
| 4096x2560 B1 | 64 | 2 | 280.1x | 3/2/3 | INADEQUATE |
| 4096x2560 B2 | 64 | 0 | 302.3x | 3/2/3 | ADEQUATE |
| 4096x2560 B8 | 64 | 0 | 330.8x | 3/2/3 | ADEQUATE |
| 4096x2560 B16 | 64 | 0 | 289.8x | 3/2/3 | ADEQUATE |
| 9728x2560 B1 | 64 | 6 | 260.6x | 3/2/3 | INADEQUATE |
| 9728x2560 B2 | 64 | 0 | 297.5x | 3/2/3 | ADEQUATE |
| 9728x2560 B8 | 64 | 0 | 310.7x | 3/2/3 | ADEQUATE |
| 9728x2560 B16 | 64 | 0 | 312.8x | 3/2/3 | ADEQUATE |
| pooled | 1536 | 28 | 260.6x | 3/2/3 | INADEQUATE |

All 28 pooled false positives sit in the six batch-1 cells and carry one signature: the held-out `mlx-on-device` implementation (MLX's own `quantized_matmul`, a correct implementation the floor must cover, held out of the ensemble), on constant-rows inputs at float16 activations.
G2 caught 768 of 768 on every fault with a pooled worst margin of 260.6x and no width-level equivalent faults; G3 passed 3 / 2 / 3 everywhere; the known value-duplicate `lut-gather == dequant-pairwise` is unchanged.
The largest-error example per shape, verbatim from the log tail (seeds 0/1 are the calibration draw, 100/101 the independent draw; all 28 lines are in the log and the JSON):

| shape | seed | error | tolerance |
|---|---|---|---|
| 1024x2560 | 0 | 0.298 | 0.0382 |
| 151936x2560 | 1 | 0.414 | 0.0477 |
| 2560x4096 | 100 | 0.763 | 0.0614 |
| 2560x9728 | 0 | 1.3 | 0.231 |
| 4096x2560 | 1 | 0.0825 | 0.00765 |
| 9728x2560 | 101 | 0.677 | 0.108 |

## STEP 6: the class-necessity diagnostic (never a gate)

Worst member-to-tolerance ratio with a class dropped, pooled over the serving records; ratios above 1 mean the dropped class is load-bearing.

| drop | members left | worst ratio | cases exceeding |
|---|---|---|---|
| dequant-domain | 5 | 0.439 | 0 |
| int-domain | 7 | 1.253 | 1 |
| device-arithmetic | 6 | 1.743 | 10 |

Device-arithmetic is load-bearing at the serving shapes: dropped, ten cases exceed the tolerance, at 1.743x worst.
Per the ADR 0009 amendment this is a necessity diagnostic, never a pass or fail gate.

## The fp16-dequant boundary under both floors

Cases outside tolerance, CPU floor -> device-joined floor, out of 768 per activation dtype:

| activations | CPU floor | device-joined floor |
|---|---|---|
| float32 | 766/768 | 753/768 |
| float16 | 0/768 | 0/768 |

## The pre-registered branch: interpretation stops here

Step 4 of the committed rule: a cell above the shipped K = 4 fires the branch - STOP interpretation, report the cell, renegotiate membership with the coordinator, membership before K, always.
That branch fired, and this ADR obeys it.
Per the standing repair order (ADR 0005, 0009, 0012), a demand above K is answered by membership first - further device variants joining the floor - and K moves only if membership cannot close it; either branch is a fleet decision taken with the coordinator, not inside this harness and not inside this ADR.
Accordingly: no member joins here, no K moves here, no cause is assigned here, and no fix is proposed here.

## Scope of the miss

The miss names batch 1 only.
Cells at batches 2, 8 and 16 are not named by the miss.
No claim about those cells is made in this ADR, in either direction: the tables above record their numbers, and any reading of them - an adequacy reading included - belongs to the renegotiation, not here.

## Consequences

K = 4 over the nine-name floor is not attested at the Qwen3-4B serving shapes; the run's operative outcome is the refusal, exit code 1.
The renegotiation belongs to the coordinator; until it lands, the pre-registered rule, the harness and the shipped tolerance stay exactly as committed.
Reruns must reproduce this ADR's tables.

## Amendment (2026-08-15, same day): the resumed step has been re-measured under the merge-review fixes

The checkpoint that supplied the resumed `continuity` step was written by pre-pool code, and the fingerprint of that era carried no code identity, so the resume validated silently across the buffer-pool change in `kernelverify/runners/device.py`.
The merge review caught this; commit c5b508a added a code-identity component to the checkpoint fingerprint and a `--continuity-only` mode.
Under c5b508a, STEP 0 was then re-measured in the foreground twice (once by the lane, once independently by the coordinator): both runs reproduced ADR 0012's 64 cached device records exactly, exit 0.
The pool therefore did not change device arithmetic, and this ADR's verdict stands on a re-measured anchor rather than a resumed one.
The pre-pool checkpoint is preserved at `bench/.cache/quant_serving_partial.prepool-backup.json`.
