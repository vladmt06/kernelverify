# ADR 0014: MLX's batch-1 and batch-16 fp16 quantized-matmul kernels are out of contract; held-out eligibility is per cell

Date: 2026-08-15
Status: Accepted

## Context

ADR 0013 recorded a DEMAND MISS: batch 1 at all six Qwen3-4B serving shapes demanded 78-195 against the shipped K = 4, with every pooled binding case the held-out `mlx-on-device` on constant-rows inputs at float16 activations.
Per the pre-registered branch, interpretation stopped and the renegotiation went to the coordinator.
This ADR is that renegotiation's ruling, taken with the coordinator on 2026-08-15 (rulings D1-D3 of the contract-lane plan), informed by the read-only investigation at `bench/.cache/b1-cliff-investigation.md` (a working-cache document; every number it derived comes from the committed evidence below and from the mlx 0.32.0 wheel source, sha-verified).

The measured evidence is committed at `bench/results/quant_serving_adequacy.json`.
Its sha256 is `54d0ad5ca4f28f00a401f0481a8b1c69c42cadcc01e3c4a21cca7728ced01833`, and every derived claim in this ADR traces to those records.

## The ruling (D1)

The out-of-contract judgement is recorded: MLX's own batch-1 quantized-matmul kernel at float16 activations violates contract clause C1, so the binding held-out was not an admissible implementation in the missing cells, and the miss is not a membership gap.
K = 4 and the nine-name membership stay unchanged; C1's wording stays unchanged; no measured record changes.
The finding is citable: the verifier's tolerance did its job and flagged silent precision loss in the stock serving stack.

## The evidence bar is structural (D2)

An exclusion requires a VERIFIED SOURCE READING showing a reduced-precision intermediate: the shipped kernel text, hash-verified against the installed wheel, exhibiting an intermediate narrower than IEEE binary32 on the cell's dispatch path.
Mechanism proofs - emulations, ablations, error decompositions - are corroboration at whatever strength they measure, never a prerequisite.
The bar is structural because an outcome-driven bar ("exclude it once it fires") would let the labelling chase the numbers, which is the hole the outside review named.

By that bar, two cells are excluded now:

- `mlx-on-device` at (batch 1, float16): `affine_qmv` / `affine_qmv_fast` compute a per-thread sub-sum of eight activations where every operand is `T = half`, so the seven additions round in fp16 before the fp32 accumulator.
- `mlx-on-device` at (batch 16, float16): `affine_qmm_t` stores its dequantized weight tile in `threadgroup T*`, i.e. half at fp16 activations.

`block-tiled` remains eligible everywhere.

## The B1 mechanism, as corroboration, at its measured strength

A CPU emulation of `affine_qmv_fast` reproduces the recorded GPU output bit-exactly on five independent cases including the pooled binding case.
The ablation isolates the whole deviation to the half-precision sub-sum: with the sub-sums evaluated in fp32 the emulated kernel lands exactly on the ensemble floor (0.00380085 vs floor 0.00380085 on the probe case), so the integer-domain algebra is harmless and fully covered by the existing membership.
On the pooled binding case the deviation costs 78 fp16 ulps of the largest output, a 3.4% relative error on that output.
Those are the measured strengths; nothing beyond them is claimed here.
On constant-rows inputs every chunk sums the same eight numbers, so the rounding error accumulates coherently (measured 12.65x over the incoherent baseline), which is why constant-rows is the binding mode.
At float32 activations the same kernel sits below the floor (worst demand 0.688 over all 192 batch-1 records): `T` is what makes the intermediate narrow, so the exclusion is fp16-only.

## The B16 numbers

At float16 activations the batch-16 `mlx-on-device` error is 1.9x the floor in the median and up to 5.3x, where batches 2 and 8 sit at exactly 1.0.
It never fired `k_demand` because it stays under `base_tol` (max 0.34x), so it was masked, not clean.
The structural bar (D2) excludes it now, with these numbers recorded; a B16 mechanism demonstration is optional corroboration work (TODOS.md).

## The reporting clarification: k_demand is a K demand, not an error magnitude

At float16 activations every contract member rounds its output to the storage dtype, so all eleven implementations report the IDENTICAL max error and the ensemble floor collapses to the output's own fp16 rounding.
`base_tol` is 11-14x that floor in the median, so the shipped tolerance `max(base, 4 * floor)` is `base_tol` in 768 of 768 float16 records: the floor term is inert at fp16.
`k_demand` divides by that inert floor, which is how a 12.43x overshoot of the tolerance actually shipped was reported as a demand of 195.2 (195.2 = 12.43 x 15.71, the base_tol-to-floor ratio at the binding case).
`k_demand` keeps its definition - it IS the K demand, and the shipped K = 4 derivation chain depends on it - and the tolerance-overshoot reading `e / max(base, K * floor)` is now printed beside it, both-readings style per the ADR 0009 precedent.
The root repair (computing the floor before the storage-dtype rounding) is recorded in TODOS.md and needs its own pre-registered pass.

## The mechanism: eligibility as versioned, hash-guarded contract data

The exclusions live in `kernelverify/schemas/heldout_eligibility.py`, keyed (held-out name, batch, dtype), each entry carrying the ADR reference, the kernel it names, and the sha256 of the kernel source it was ruled against (`4da52bf4ee688165a65b84c52a5f4e82efcae7f69e8c74d9ee3e00bef463c99f`, the mlx_metal 0.32.0 wheel's `quantized.h`, matching the wheel RECORD).
Consumers verify the installed wheel before applying any exclusion and FAIL LOUDLY on mismatch: a changed source hash is a stale ruling even at an unchanged version string, and an unreadable source falls back to requiring the exact ruled MLX version.
An MLX fix can therefore never be waved through on a stale label.
`tests/test_heldout_eligibility.py` pins the table's exact contents.

## The derived reading of the ADR 0013 records

`bench/reinterpret_serving_adequacy.py` re-runs the STEP 4/5 interpretation from the committed evidence (pure CPU re-read, sha256-checked) into `bench/results/quant_serving_reinterpretation.json`, whose header carries the evidence sha256, a sha256 over the interpretation code, and the eligibility version.
Invariance is enforced: in-contract quantities must be bit-identical to the tables the measuring run recorded, or the write refuses; only labelling differs.

The reading, per cell (k_demand unchanged | overshoot of the shipped tolerance | admissible-only demand):

| cell | k_demand (cal / indep) | overshoot | admissible |
|---|---|---|---|
| 1024x2560 B1 | 78.479 / 3.806 | 7.81x / 0.79x | 6.973 / 3.806 |
| 151936x2560 B1 | 105.021 / 93.925 | 8.67x / 8.27x | 1.721 / 2.972 |
| 2560x4096 B1 | 97.497 / 195.198 | 8.73x / 12.43x | 3.445 / 4.001 |
| 2560x9728 B1 | 91.841 / 160.037 | 7.65x / 12.34x | 3.405 / 2.165 |
| 4096x2560 B1 | 170.510 / 4.007 | 10.79x / 0.81x | 4.563 / 4.007 |
| 9728x2560 B1 | 78.767 / 84.753 | 6.98x / 6.28x | 4.947 / 4.065 |
| pooled | 170.510 / 195.198 | 10.79x / 12.43x | 6.973 / 4.065 |

Batches 2, 8 and 16 are unchanged from ADR 0013's tables at every shape; the B16 cells carry the exclusion label with nothing withheld, because their fp16 `mlx-on-device` evidence stays under `base_tol` either way.
The 28 pre-ruling false positives are re-labelled as out-of-contract flags - the verifier flagging an implementation C1 entitles it to flag - and every cell's gate verdict is ADEQUATE under eligibility-aware G1.

## The honest residual: an in-contract demand above K = 4 remains

Withholding the out-of-contract contributions does not clear STEP 4.
Four B1 cells keep an admissible-only demand above the shipped K = 4: 1024x2560 (6.973 calibration), 2560x4096 (4.001 independent), 4096x2560 (4.563 / 4.007) and 9728x2560 (4.947 / 4.065).
Every one is bound by the ensemble member `device-factored-simd` at float32 constant-rows - a leave-one-member-out spread inside the admissible class, present in the recorded tables all along and shadowed by the 78-170x out-of-contract readings until now.
This ADR records the reading and resolves nothing about it: per the standing repair order (ADR 0005, 0009, 0012) it is a membership-before-K renegotiation that belongs to the coordinator.
The derived artifact reports these cells as in-contract misses, and a rerun of the harness would fire the pre-registered DEMAND MISS branch on exactly them.

## The certificate consequence

At the excluded cells the attestation is ADEQUATE-BY-EXCLUSION: there is no admissible device held-out evidence there, and adequacy follows from the exclusion rather than from a covered measurement.
Every certificate whose tolerance model divides by the K_QUANT floor now states this in its validity domain (`bench/emit_pack_certificates.py`): admissible class only, MLX's own B1/B16-fp16 kernels out of contract, those cells evidence-free.

## Consequences

> Two of the lines below are superseded by ADR 0016; see the 2026-08-16 amendment at the end of this file.

- K = 4, the nine-name membership, C1's wording, the measured records and the in-contract stop-at-miss semantics are all unchanged.
- ADR 0013's freeze line is amended by its own amendment pattern: reruns must reproduce the measured records bit-identically, and their interpretation follows this ADR; the exit-1 refusal was the pre-ruling reading.
- The four-cell in-contract residual is open coordinator business; nothing here closes it.
- The inert-fp16-floor repair and the optional B16 mechanism demonstration are recorded in TODOS.md.

## Amendment (2026-08-16): both Consequences lines above are superseded by ADR 0016

Two lines in the Consequences section were true when written and are not true now.
The merge review of the contract lane found them; they are corrected here rather than edited in place, so the record of what this ADR concluded stays readable.

"The four-cell in-contract residual is open coordinator business; nothing here closes it."
ADR 0016 closes it.
The residual was never a K or membership gap: the CPU member `factored-groups` summed each group with numpy's pairwise reduction, which is EXACT on a constant row, so the leave-one-out floor was too tight in exactly the regime the four cells lived in and a correct device kernel read as a demand above K.
Under the repaired member the four cells read 1.022, 1.052, 0.941 and 0.915, recomputable from `bench/results/quant_serving_repaired_member.json` on every test run.

"reruns must reproduce the measured records bit-identically"
That freeze line no longer holds as stated, for the same reason ADR 0013's own third amendment narrows it: ADR 0016 deliberately changed the `factored-groups` column, so reruns reproduce every column bit-identically EXCEPT that one, and everything derived from it moves with it by design.
