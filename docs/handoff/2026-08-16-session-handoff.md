# kernelverify: state of the project and what is left

Written 2026-08-16 at commit `3a0acb9`, pushed to `origin/main`.
Suite: 883 passed, 0 failed, ~87 s, in a venv with `mlx==0.32.0` and `pyobjc`.

This document exists so the next session starts from measured facts rather than from memory.
Every number in it was produced by a command in the session that wrote it.

---

## 1. What the product is

One person, one MacBook, one open-weight model, generating tokens one at a time.
They want it FASTER and using LESS MEMORY than Apple's stock path, and they want every kernel that gets them there to be provably correct rather than plausibly correct.

The project has two halves and they are at very different stages.

| half | what it does | state |
|---|---|---|
| the verifier | decides whether a kernel is correct, and never flags a correct one | built, and as of this session honest about its own limits |
| the optimiser | produces faster kernels for the verifier to gate | DOES NOT EXIST - zero code generates a candidate kernel |

The strategy is verifier first, then optimiser.
That order is deliberate: an optimiser without a trustworthy gate produces kernels nobody can ship.

## 2. The one number that matters and does not exist yet

Nothing in this repository has ever measured end-to-end token generation speed with our kernels installed.

- `bench/serve_sub4bit.py` is the harness that would.
- It is built, pre-registered, merged, and has never been run.
- Running it is task M2 below, and it is the highest-value thing left.

What IS measured, and constrains everything:

- Dense 4-bit decode already runs at ~92% of this machine's memory-bandwidth ceiling (ADR 0007). There is ~8% left from writing a better kernel at the same byte count. That road is closed.
- Fixed non-kernel cost is 0.99 ms per token (ADR 0007).
- Our one live kernel, `wide_qmv` at 3 bits, is a measured LOSS at batch 1 to 3 at every shape (0.937x at M = 1, from `bench/results/qmv-boundary-pricing-2026-08-15.json`) and wins only at batch 5 to 9.
- Batch 1 is the target user. So the kernel we have does not help the user we are building for, and we have known this since 2026-08-15.

The consequence: speed at batch 1 comes from reading FEWER BYTES PER WEIGHT, not from better code.
MLX affine 3-bit reads 3.5 bits/weight (3 bits + one fp16 scale and bias per 64-group).

## 3. What this session changed

### The verifier's headline gate could not fail

G1 says "the verifier flags no correct kernel". Every ADR since 0009 reported `G1 = 0` as evidence.
It was not evidence: `k_demand` maximised `heldout/floor` over a set of records, `main` set `k >= demand`, and `gates()` then scored G1 over those same records. Algebraically guaranteed.

Fixed (ADR 0017): K now comes from the CALIBRATION draw alone; the independent draw TESTS it; a held-out demand above K is an INDEPENDENT MISS that names the cell and exits 1.

M1 ran it. Result: **no miss fired**, and at three of four bit widths the held-out draw demanded MORE than calibration (1.907 vs 1.476 at 3 bits, 1.840 vs 1.783 at 4, 2.061 vs 1.394 at 8). Under the old rule those would have set K themselves.

**Stated honestly in ADR 0017:** `G1 = 0` is still true by construction, because the miss branch exits before `gates()` runs. The falsifiable check is the miss branch. The headline of any future run is `INDEPENDENT MISS: none`, never `G1 = 0`. And the held-out draw differs from calibration by SEED only - same shapes, same weight distributions - so what survived is two seeds, not a genuinely independent draw.

### Ten shipped false positives, closed

A CPU ensemble member (`factored-groups`) summed each group with numpy's pairwise reduction, which is EXACT on a constant row, while every real int-accumulate kernel chains and rounds 63 times. So it was most accurate exactly where device kernels are least accurate, the tolerance floor was too tight there, and the shipped verifier flagged a CORRECT device kernel on 10 of 1,536 records.

Fixed (ADR 0016). Worst overshoot 1.743x becomes 0.328. K stays 4.0, and the reason is load-bearing: the device grid alone would license 3.0, but the serving grid demands 3.120 at a shape the device harness does not measure.

### Every ADR number now has committed evidence

ADR 0016's four headline readings rested on computations nobody could repeat. Two generators now produce them and tests recompute them every run:

- `bench/derive_repaired_member_column.py` (~70 min, CPU, budget 30 GB) - the serving column.
- `bench/derive_prerepair_device_records.py` (~7 min, GPU) - the detection-price "before" half.

Both artifacts carry a whole-file hash of the code that produced them. That guard has already fired three times on real staleness, including twice on merges. It is deliberately over-firing: over-firing costs a re-run and says so, under-firing is the bug this repo already had.

### The A/B harness can be trusted to run

Twelve defects found by the merge gate, all fixed: it takes the machine lock, imports the shared memory guard, has numbered refusals, tests for its decision functions (losing branches first), implements the non-decider rule it had only printed, checks its corpus digest, and matches its own pre-registration.

### The record matches the tree

AGENTS.md had eight stale facts. Fixed and now guarded by `tests/test_docs.py` - ADR numbering, catalogue count, every path named, empty-package list, and the 100% claim's record compared against the live catalogue.

---

## 4. Findings that are open, in priority order

| # | label | finding |
|---|---|---|
| F1 | BLOCKER for the product | The optimiser does not exist. Zero code produces a candidate kernel. `kernelverify/runners/specialize.py` only substitutes into a fixed template. |
| F2 | BLOCKER for the product | The one live kernel loses at batch 1 to 3, which is the target user's regime. It has never been tested end to end. |
| F3 | INVESTIGATION | 55 of 65 faults in the catalogue are this project's own seams in its own kernels. The verifier is largely pointed at itself. |
| F4 | HEADS UP | Two of the three top-ranked speed techniques are out of contract as published (they accumulate in fp16). Adopting them means taking the packing and rewriting the reduction in fp32 - unpublished work on any backend. |
| F5 | HEADS UP | arXiv 2608.12700 (this month) is a contract-grade verifier for LLM-generated kernels whose twelve gates include our exact fp32-accumulate line, audited over 2,638 kernels with 39.5% broken. Our thesis, published by someone else. |
| F6 | HEADS UP | 4 of 9 heavy harnesses still do not take the machine lock: `calibrate_quant_device`, `measure_baselines`, `emit_pack_certificates`, `spike_mlx_e2e`. The other five do. I2 closes it. |
| F7 | HEADS UP | The held-out draw for K differs from calibration by seed only. Queued in TODOS. |
| F8 | HEADS UP | The int-domain arithmetic class rests on ONE outlying member (leave-one-out 7.561 vs next 2.166). A membership question, recorded not resolved. |
| F9 | DEFERRAL | 26 committed certificates predate the ADR 0016 repair and identify the ensemble by member NAME only. M3 re-emits them. |

---

## 5. What is left to do

### Immediately valuable

**M2 - the end-to-end A/B.** The measurement that answers whether any of this makes the laptop faster.
- Command: `bench/serve_sub4bit.py --mde` FIRST, then `--ab`. The `--ab` arm now refuses with exit 8 without the MDE cache, by design.
- Needs: a quiet window, the machine lock (it takes it itself), ~30 min, `python -u`, output to `bench/.cache/serve_ab_<date>.log`.
- Its outcomes are PRE-REGISTERED in the plan and must not be renegotiated after seeing them:
  - The B = 1 arms pin the batch-1 baseline every future spike is measured against. This exists whatever else happens.
  - `wide_qmv` wins at B in {5,6,8}: it stays in the pack for the multi-stream regime, and that says nothing about batch 1.
  - `wide_qmv` loses or is within spread at every routed batch: it is RETIRED for the target model, the pack is empty, and research must produce the first kernel. No task may explain the loss away.

### Sequenced work

| task | what | cost | blocked by |
|---|---|---|---|
| V1-serving + I4 + I5 | The `K_SHIP` import (stashed at `kv-infra` `stash@{0}`), progress lines leading with phys_footprint, the orphan-child hole | one ~70 min regeneration for all three, batched | nothing |
| I2 | Generalise the detached runner; add the lock to the 5 harnesses that lack it | ~half day | nothing |
| I3 | Fix the order-dependent footprint test; add `slow`/`gpu` markers so a lane can run tests while a measurement holds the machine | ~1 h | nothing |
| M3 | Re-emit the 26 certificates with the ensemble version and hash | minutes, GPU | after M2's verdict, since a retired kernel means fewer certificates |
| plan branch | Fold `plan/audit-amendments` into main | trivial | nothing |

### The next plan, opened by the research lane

`docs/research/2026-08-16-faster-honest-decode-kernels-literature.md` section 7 names what to build, in order:
1. the candidate store (without it every spike leaves a number in a terminal)
2. one hand-written candidate through the whole generate-gate-keep loop
3. the generator

Three techniques are ranked with M = 1 spikes and three-outcome go/no-go (GO / NO-GO / NEW CLASS) pre-registered before any spike runs. Sections 5 and 6 are binding on that plan.

---

## 6. State of the repository

- Branch: `main` at `3a0acb9`, pushed, in sync with `origin/main`.
- Unmerged branches worth knowing: `plan/audit-amendments` (the approved plan, fold it in), plus backup and scratch branches that can be pruned.
- Worktrees: `kv-baseline` (holds the only ppl corpus - do not remove), `kv-infra`, `kv-verif`, `kv-hygiene`, `kv-research`. The four lane trees are merged and can be pruned when convenient. **`kv-infra` holds `stash@{0}` with V1's serving half - do not remove that tree before landing it.**
- ADRs 0001-0017, contiguous, guarded by test.
- TODOS.md carries 22 entries.

## 7. Rules that bind any session continuing this work

- Never commit on main directly; branch, then merge through the two-axis `code-review` gate, then Vlad approves.
- Never push, merge, force-push, rebase published commits, or discard a dirty tree without asking.
- Stage files by name. `git add -A` and `git add .` are banned.
- One heavy measurement on the machine at a time. Harnesses take the lock themselves; a launcher must never take it on their behalf, because a child cannot acquire the flock its parent holds.
- A changed pre-registration is amended IN WRITING before the run, never silently in code.
- Every ADR number gets committed evidence a test can recompute.
- No AI attribution in commits. No em dashes. Markdown one sentence per line.
