# Audit amendments and research redirection - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-aim kernelverify at its actual product - one person on a MacBook running an open-weight model faster than Apple's stock path, with every AI-produced kernel provably correct - by (1) starting the literature research that decides how the fast kernels get made, and (2) amending the verifier, the infra and the records so the safety net is honest while that happens.

**Architecture:** Five workstreams that do not depend on each other except where stated: R (research), V (verification integrity), I (infrastructure), H (hygiene), M (measurements the coordinator runs one at a time). Each V/I/H task lands on its own short branch off main after the two pending merges (contract, runner) go through the code-review gate. R produces documents, not code, and its findings open the NEXT plan (kernel spikes and the optimiser).

**Tech Stack:** Python 3.12 (`/Users/vlad/kernelverify/.venv`), numpy, MLX 0.32.0, pyobjc (Metal), pytest 9.1.1; git worktrees per lane.

**Spec:** the four audit reports of 2026-08-15 (strategy, verification design, infra, claims-vs-reality), verified claim by claim in the coordinator session, plus Vlad's statement of the vision quoted in Context.

## A note on task shape

R1, R2, H1, H2, H5 and H6 are prose tasks (documents, records, a fleet decision), so "bite-sized steps with real code" applies to their commands and file edits, not to code they do not contain. Every code task carries real test code and real implementation text.

## Global Constraints

- Stage files by name; `git add -A` and `git add .` are banned.
- Never commit on main; never push, merge, force-push, or discard a dirty tree without Vlad's word.
- Every merge into main runs the `code-review` skill on the merge diff, then waits for Vlad's approval.
- One heavy measurement on the machine at a time; the coordinator (or the detached runner after I2) runs them, never a headless worker.
- Ad-hoc analysis over the serving records is a heavy harness: one block in memory at a time, phys_footprint printed per block, watchdog on the child pid via `proc_pid_rusage`, never RSS.
- No AI attribution in commits; commit messages are sentence-per-line prose that explain WHY.
- Markdown: one sentence per line; no em dashes (use "-").
- Full suite green before every commit, quoted from a venv with mlx AND pyobjc.
- Report format after any run: BUG NEEDS FIXING / HEADS UP / BUG FIXED / BLOCKER / INVESTIGATION / DEFERRAL / WHATS NEXT, one sentence, one label each.

---

## Context

Vlad's vision, verbatim: "this is meant for someone running an open weight model on their macbook who wants it to be faster and more efficient while also not producing fucked up kernels", where "not fucked up" means "the kernels that are ai produced are not fucked up which is the current problem we've been building the verification layer for", and the route to faster is "research using research papers on arxiv and find how to make a product that is both faster and more accurate than the stock".

So the product is an AI kernel optimiser for Apple silicon whose every output is verified, aimed at batch-1 decode on a laptop.

What the four audits found, each claim verified in the coordinator session before it went into this plan:

| Audit | Verified headline |
|---|---|
| Strategy | 11% of 151 commits and 1 of 16 ADRs are about kernel speed; the pack's one live kernel is a measured LOSS at batch 1-3 at every shape (worst 0.39x) and wins only at batch 5-9; the end-to-end serving A/B (`serve_sub4bit.py --ab`) is built, pre-registered, and has never been run; two of three pack kernels are dead for the target model (kv_attention demoted by measurement, moe_dispatch because Qwen3-4B has no experts) |
| Verification design | G1 ("zero false positives") cannot fail by construction: `k_demand` maximises `heldout/floor` over the same records `gates()` then scores at `k >= demand`; the shipped tolerance divides by 6 CPU members while K was derived over 9; three of nine members are value-duplicates whose leave-one-out ratio is pinned at 1.0; the ensemble has 4 effective points; the quant ensemble is not drawn from any contract module; kv_attention uses K=4.0 over an ensemble no harness ever calibrated; `K_SHIP` is a hand-copied literal; moe/kv fingerprints cannot see arithmetic changes |
| Infra | `serve_sub4bit.py` takes no lock, no budget, no memory gate (grep count 0) - the same class of harness that produced the 39.5 GB kill; `detached_run.py` already survives session teardown but is hardcoded to one harness; the measurement queue is ~2.5 h wall-clock, so capacity is not the shortage; a different-chip Mac adds a column, not throughput (SCHEMA.md forbids cross-group comparison); one order-dependent test; RSS still printed on the progress lines that narrate the step that killed the machine three times |
| Claims vs reality | The optimiser does not exist (zero code generates candidate kernels); 55 of 65 faults are the project's own seams in its own kernels; the 26 committed certificates predate today's repair and identify the ensemble by member NAME only; AGENTS.md has eight stale facts and omits ~5,500 of ~8,000 package lines from its Layout |

Two things this plan deliberately does NOT do: it does not tune or redesign the current kernel (that waits for R2's evidence), and it does not move K or rebuild the ensemble (that is a contract renegotiation with its own ADR, listed under Deferred with the numbers).

## Sequencing (ruling OV-6: the premise is tested first)

```
now:    merge contract-b1-ruling -> main   (gate + Vlad)      heals the ADR record (0014, 0016 join 0015)
then:   merge metal-runner -> main         (gate + Vlad)
FIRST:  I1 EXPANDED (lock + budget + exits, PLUS the decision-surface tests, the MDE rule, the corpus pin, the registration amendment, one timing discipline - see the task)  ->  M2 the E2E four-arm A/B, coordinator-run, one quiet window
        M2's outcome is PRE-REGISTERED here, before it runs:
          - ALWAYS: the B = 1 arms (stock 3-bit, stock 4-bit, ours-routed which routes nothing at B = 1) pin the BATCH-1
            end-to-end baseline - tokens/s per stream at B = 1 - that every research spike must beat. This is the number the
            product is measured against; it exists after M2 whatever wide_qmv does.
          - wide_qmv WINS end-to-end at B in {5,6,8} (per-stream tokens/s beyond spread, per the findings doc's own gates):
            the kernel stays in the pack for the multi-stream regime it was measured in; that says nothing about batch 1.
          - wide_qmv LOSES or is within spread at every routed batch: the kernel is RETIRED from the pack for the target model
            (routing entries dropped by a new recording, certificates withdrawn), the pack is empty, and R's synthesis
            must produce the first kernel. No task in this plan may explain the loss away; the kv_attention precedent applies.
then, in parallel on short branches off main:
        R1..R2  research (a worker with the `research` skill; docs only)
        V1..V6  verification integrity     I2..I5 infra          H1..H6 hygiene
GPU, one at a time, coordinator or (after I2) the detached runner:
        M1 device grid under V5's rule (7 min)  ->  M3 certificate re-emission after V4
```

Branch ownership (ruling OV-5): every edit to `bench/calibrate_quant_serving.py` (V1's one import line, I4, I5) lands on the infra branch in that order; the verif branch never touches that file.

---

# Workstream R - Research: how to make AI-produced kernels beat stock at batch 1, provably

Executor: one worker using the `research` skill, on branch `research/faster-honest-decode`, worktree of choice. Docs only; no code, no measurements. Everything is pre-registered before searching, the way every harness in this repo pre-registers before measuring.

### Task R1: Pre-register the question, criteria and rubric; then run the search and fill the candidates table

**Files:**
- Create: `docs/research/2026-08-16-faster-honest-decode-kernels-literature.md`

**Interfaces:**
- Produces: the document R2 reads; section names below are load-bearing.

- [ ] **Step 1: Write the skeleton with the question and the two sub-questions**

```markdown
# Faster and honest decode kernels on Apple silicon: what the literature says

Status: PRE-REGISTERED 2026-08-16, before any paper was read for this document.

## 1. The question
One person, one MacBook, one open-weight model, batch-1 decode.
We want that person's model to run FASTER and in LESS MEMORY than Apple's stock path, and every kernel that gets it there to be produced by a generator whose every candidate the verifier gates.

What this repo has already measured, and this review must not re-derive (ruling D1 of the 2026-08-16 plan review):
- Dense 4-bit decode already runs at ~92% of the memory-bandwidth ceiling on this machine (ADR 0007); the remaining ~8% is the most any dense-4-bit GEMV kernel can ever win at batch 1. That is not where "faster" comes from.
- The fixed cost outside the kernel is 0.99 ms per token (ADR 0007, "the small-model gap is dispatch cost").
- Apple's own sub-4-bit (3-bit, 2-bit) path is inefficient by construction (pivot design E3, PolyQ's 2^b table wall), and 3-bit reads fewer bytes than 4-bit, so an honest 3-bit kernel near the ceiling would be BOTH faster than stock 4-bit AND smaller in memory. Our current 3-bit kernel loses to stock at batch 1 (ADR 0015: 0.937x-0.98x at M=1); the headroom is real and unclaimed.
- Apple's stock batch-1 kernel rounds in half precision internally (ADR 0014); an honest kernel accumulates in fp32 or better, and that is our contract's line.

Sub-question A1 (sub-4-bit kernels): what published kernel designs bring 2/3-bit weight-only GEMV close to the bandwidth ceiling at batch 1 on a unified-memory GPU without narrowing intermediate precision - packing layouts, dequant-on-the-fly schemes, lookup-table methods and their limits, register/occupancy strategies?
Sub-question A2 (around the kernel): for a single-user decode loop on a MacBook, where does the non-GEMV time go and which published techniques reclaim it honestly - dispatch and launch overhead (our 0.99 ms/token), speculative and parallel decoding, prefill, kernel fusion of the per-token tail?
Sub-question B (generator architecture): what published approaches to generating or searching GPU kernels (LLM-authored, autotuned, search-space) have a correctness gate, what did the gate catch, and how did they keep candidates honest?

## 2. Inclusion criteria (a paper is IN only if all hold)
- Addresses quantized matvec/GEMV or the decode step of an LLM, on a GPU.
- Reports speed against a named strong baseline (not against an unoptimised reference).
- States its accumulation precision, or gives enough kernel detail to infer it.
- For sub-question B: the generated kernels were checked for correctness by something more than "the benchmark ran".

## 3. Rubric (0-3 each; recorded per paper in the table)
- Evidence: 3 = open code + reproduced by others; 2 = open code; 1 = numbers only; 0 = claims.
- Applicability to Metal/unified memory: 3 = demonstrated on Apple silicon; 2 = memory-bound technique with no CUDA-only dependency; 1 = needs a hardware feature Metal lacks (tensor-core int8, warp shuffle semantics differ); 0 = CUDA-specific.
- Honesty by our contract: 3 = fp32 accumulate throughout; 2 = int accumulate then fp32 (our int-domain class); 1 = mixed with a narrower intermediate somewhere; 0 = fp16 accumulate.
- Bytes read per weight at M = 1, relative to MLX affine 3-bit (3 bits + one fp16 scale and one fp16 bias per 64-group = 3.5 bits/weight): 3 = under 3.25 bits/weight with the group parameters counted; 2 = 3.25-3.5; 1 = 3.5-4.0; 0 = at or above 4-bit dense. This is the mechanism at batch 1 (bandwidth-bound) and it is derivable from any paper's packing description; the paper's own reported gain and the hardware it was measured on go in a NOTE column, never in the score (ruling OV-7: CUDA-measured speedups do not transfer).

## 4. Candidates (filled by R1 step 4)
| paper | year | sub-q | technique in one line | evidence | Metal | honest | bytes/weight | reported gain (hardware) | why it might beat stock at batch 1 |
|---|---|---|---|---|---|---|---|---|---|

## 5. Synthesis (filled by R2)
## 6. What the verifier would need (filled by R2)
## 7. The optimiser's shape (filled by R2)
```

- [ ] **Step 2: Commit the skeleton before reading anything**

```bash
git add docs/research/2026-08-16-faster-honest-decode-kernels-literature.md
git commit -m "Pre-register the decode-kernel literature review: question, criteria, rubric, before any paper is read"
```

- [ ] **Step 3: Search arxiv (and the linked code repos) with at least these queries, recording every hit that passes section 2**

Sub-question A1 queries: `sub-4-bit weight-only quantization GPU kernel`, `3-bit 2-bit GEMV kernel bandwidth`, `lookup table quantized inference GPU` (LUT-GEMM, FLUTE, "any-precision LLM", PolyQ), `bit-plane packing low-bit matvec`, `weight-only quantization kernel Apple silicon Metal`, `MLX quantized matmul`, `llama.cpp Metal K-quants kernel` (the incumbent at sub-4-bit), `dequantize-on-the-fly kernel bandwidth roofline`.

Sub-question A2 queries: `LLM decode kernel launch overhead fusion`, `per-token dispatch overhead GPU inference`, `speculative decoding` (Medusa, EAGLE, lookahead - what the acceptance rate does on a memory-bound single-user loop), `prefill decode split optimization`, `KV cache quantization decode speed` (we measured a loss at every setting - papers that claim a speed win are the interesting ones, and their setting must be recorded).

Sub-question B queries: `LLM generated GPU kernels benchmark` (KernelBench and successors), `AI CUDA engineer`, `correctness of LLM generated kernels` (the vendored "Correctness Illusion" corpus paper and everything that cites it), `kernel autotuning search space GEMV`, `Triton autotuner correctness`, `verified kernel generation`.

- [ ] **Step 4: For each IN paper, add one row to the section 4 table with all four rubric scores and the one-line mechanism**

- [ ] **Step 5: For each OUT paper that a reader would expect to see, add a one-line "excluded because <criterion>" list under the table**

- [ ] **Step 6: Commit**

```bash
git add docs/research/2026-08-16-faster-honest-decode-kernels-literature.md
git commit -m "Literature review: candidates table filled under the pre-registered criteria"
```

### Task R2: Synthesise and rank; name what the verifier would need; the optimiser's shape

**Files:**
- Modify: same document, sections 5, 6 and 7
- Modify: `TODOS.md` (one entry: "Optimiser: first spike plan", pointing at section 7)

- [ ] **Step 1: Section 5 - rank the sub-question A1 candidates by (honest x bytes-per-weight), evidence as tie-break; keep the top three; for each write: the mechanism, why it should still win on Metal at batch 1, the risk that it does not, and the cheapest spike on THIS machine that would tell (verify with `bench/pack_wide_qmv.py`'s gate, time with `bench/price_qmv_boundary.py`'s protocol, one shape, M = 1 only)**

- [ ] **Step 2: Section 6 - for each of the three, name the arithmetic class it lands in (dequant-domain, int-domain, device-arithmetic, or NEW), and if NEW, what member the contract ensemble would need before the verifier could gate it without a false positive; cite ADR 0012's membership-before-K order**

- [ ] **Step 3: Pre-register the go/no-go for the spikes, three outcomes (CEO outside-voice ruling OV2-6): (i) GO - the technique's M = 1 spike measures WIN under the pricing protocol (`ratio_lo > 1`) AND passes the gate; (ii) NO-GO - it measures LOSS or REFUSED, or it fails the gate while its arithmetic class already has ensemble members - recorded and stopped; (iii) NEW CLASS - section 6 named its class as NEW, so the gate's tolerance has no member of that class and a flag would be the ten-false-positives mechanism again: its spike's gate result is READ ONLY AFTER a membership block (a member of that class added under its own pre-registration, K re-derived per the standing membership-before-K order, ADR 0012), then (i)/(ii) apply. Which outcome a technique is eligible for is fixed by section 6 before any spike runs.**

- [ ] **Step 4: Commit**

```bash
git add docs/research/2026-08-16-faster-honest-decode-kernels-literature.md
git commit -m "Literature review: top three techniques ranked, spikes and go/no-go pre-registered, contract needs named"
```

- [ ] **Step 5: From the sub-question B papers, write section 7: the generate -> gate -> keep loop as it would sit on THIS repo: candidates enter through `kernelverify/runners/specialize.py` (template substitution) or as raw MSL; every candidate runs `bench/pack_wide_qmv.py`'s gate (or the operator's) BEFORE any timing; timing under `price_qmv_boundary`'s protocol; a kept candidate gets a certificate from `bench/emit_pack_certificates.py`. Name what is missing (the generator, the search space, the candidate store) and what exists (the gate, the pricing protocol, the certificate emitter, the extraction/report machinery).**

- [ ] **Step 6: Add the TODOS entry and commit**

```bash
git add docs/research/2026-08-16-faster-honest-decode-kernels-literature.md TODOS.md
git commit -m "Literature review: the optimiser's shape on this repo, and what is missing"
```

---

# Workstream V - Verification integrity

Executor: one worker per task or one worker for all, on branch `verif/integrity` off main AFTER the contract merge (V3 needs main's `MOE_MEMBERS` and the branch's `QUANT_ENSEMBLE_VERSION` together).

### Task V1: One source for the shipped K

Branch note (rulings OV-5, OV2-pkg): this task is split by file. The `calibrate_quant_serving.py` import line AND its identity test (`tests/test_serving_adequacy.py`) commit together as the INFRA branch's first commit - before I1 - so no branch ever carries that test red; the device-harness half and its test land on the VERIF branch. The lane table below reflects this order.

**Files:**
- Modify: `bench/calibrate_quant_serving.py:322` (`K_SHIP = 4.0` literal) - on the infra branch
- Modify: `bench/calibrate_quant_device.py` around lines 400, 428-437, 461-462, 508 as of the pre-merge branch (labels and JSON keys) - re-locate by symbol after the merge
- Test: `tests/test_serving_adequacy.py`, `tests/test_quant_device_members.py`

**Interfaces:**
- Produces: `calibrate_quant_serving.K_SHIP` IS `kernelverify.schemas.native_ops.K_QUANT` (same object); the device harness's JSON carries `k_grid` (its own reading) and `k_shipped` (native_ops), never a key called `k_ship`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_serving_harness_ships_the_verifier_k_not_a_copy():
    import calibrate_quant_serving as h
    from kernelverify.schemas.native_ops import K_QUANT
    assert h.K_SHIP is K_QUANT
```

- [ ] **Step 2: Run it: `.venv/bin/python -m pytest -q tests/test_serving_adequacy.py -k ships_the_verifier_k` - expected FAIL. (Identity is the right check: two `4.0` literals in two modules are different objects and an imported alias is the same object - verified in the 2026-08-16 plan review, ruling 4A.)**

- [ ] **Step 3: Replace the literal**

```python
# bench/calibrate_quant_serving.py, replacing line 322
from kernelverify.schemas.native_ops import K_QUANT as K_SHIP  # the verifier's K, never a copy (ADR 0016)
```

- [ ] **Step 4: In `calibrate_quant_device.py`, rename the printed and persisted reading: the line that prints `shipped K: {k_ship}` becomes `device-grid K: {k_ship} (this harness's shapes only; the verifier ships native_ops.K_QUANT = {SHIPPED_K})` where `SHIPPED_K` is imported from `kernelverify.schemas.native_ops`; the JSON at line ~508 writes `{"k_grid": k_ship, "k_shipped": SHIPPED_K, ...}`; add a test asserting both keys are present and `k_ship` is absent**

- [ ] **Step 5: Run the suite; commit - two commits on two branches**

```bash
# infra branch, first commit (serving line + its test together, so the test is never red on any branch):
git add bench/calibrate_quant_serving.py tests/test_serving_adequacy.py
git commit -m "The serving harness ships the verifier's K, never a copy"
# verif branch:
git add bench/calibrate_quant_device.py tests/test_quant_device_members.py
git commit -m "The device grid prints its own K as its own reading, beside the one that ships"
```

### Task V2: Say plainly that kv_attention's K is borrowed and uncalibrated

**Files:**
- Modify: `kernelverify/schemas/native_ops.py:305-316` (`KV_MEMBERS`, `kv_tolerance`)
- Modify: `bench/emit_pack_certificates.py:227-228` (`contract_version_for` kv branch)
- Modify: `TODOS.md` (entry: calibrate kv_attention's K over KV_MEMBERS with a pre-registered harness modelled on calibrate_quant_bits; CPU-only)
- Test: `tests/test_native_ops.py`

- [ ] **Step 1: Failing test: `assert "borrowed" in inspect.getdoc(native_ops.kv_tolerance)` and the certificate prose for kv contains "K borrowed from quantized_matmul, uncalibrated"**
- [ ] **Step 2: Add the docstring to `kv_tolerance` stating: K_QUANT = 4.0 is BORROWED from the quantized_matmul calibration (ADR 0012/0016); no harness has derived a K over KV_MEMBERS; ADR 0008's "its own calibrated K" does not hold for this operator. Change the certificate prose accordingly.**
- [ ] **Step 3: Suite; commit**

```bash
git add kernelverify/schemas/native_ops.py bench/emit_pack_certificates.py TODOS.md tests/test_native_ops.py
git commit -m "kv_attention's K is borrowed and says so; calibrating it is queued"
```

### Task V3: Version every ensemble the verdict cache fingerprints

Written pre-merge, lands post-merge: main already derives the moe labels from `MOE_MEMBERS` (native_ops:229) and this branch already has `QUANT_ENSEMBLE_VERSION`; the merge of contract-b1-ruling into main brings both together and `kernelverify/battery/core.py::_oracle_member_labels` is a CONFLICT SITE in that merge (both sides edited it). Resolve it by keeping main's derived moe labels plus this branch's version token, then this task adds the kv/moe versions on top. The "moe labels are derived" assertion below is therefore a regression guard on main's behaviour, not new work.

**Files:**
- Modify: `kernelverify/schemas/native_ops.py` (add `KV_ENSEMBLE_VERSION = "kv-ensemble-v1"`, `MOE_ENSEMBLE_VERSION = "moe-ensemble-v1"` beside `KV_MEMBERS` / `MOE_MEMBERS`)
- Modify: `kernelverify/battery/core.py:44-67` `_oracle_member_labels()`
- Test: `tests/test_quant_contract_members.py` (beside `test_verdict_cache_fingerprint_sees_the_ensemble_version`)

- [ ] **Step 1: Failing test**

```python
def test_verdict_cache_fingerprint_versions_every_native_ensemble():
    from kernelverify.battery.core import _oracle_member_labels
    from kernelverify.schemas.native_ops import KV_ENSEMBLE_VERSION, MOE_ENSEMBLE_VERSION, MOE_MEMBERS
    labels = _oracle_member_labels()
    assert f"kv-ensemble={KV_ENSEMBLE_VERSION}" in labels
    assert f"moe-ensemble={MOE_ENSEMBLE_VERSION}" in labels
    assert all(name in labels for name in MOE_MEMBERS), "moe labels are derived, not literals"
```

- [ ] **Step 2: Implement: `labels += sorted(MOE_MEMBERS); labels.append(f"moe-ensemble={MOE_ENSEMBLE_VERSION}"); labels += sorted(KV_MEMBERS); labels.append(f"kv-ensemble={KV_ENSEMBLE_VERSION}")` replacing the two literal strings**
- [ ] **Step 3: Suite; commit**

```bash
git add kernelverify/schemas/native_ops.py kernelverify/battery/core.py tests/test_quant_contract_members.py
git commit -m "Every native ensemble carries a version the verdict cache can see"
```

### Task V4: Certificates identify the ensemble by version, and state the moe tolerance correctly

**Files:**
- Modify: `bench/emit_pack_certificates.py:233-249` (`tolerance_model_for`)
- Test: `tests/test_pack_certificates.py`

- [ ] **Step 1: Failing tests: wide_qmv's tolerance_model has `ensemble_version == QUANT_ENSEMBLE_VERSION` and `ensemble_source_sha256` (sha256 of `kernelverify/schemas/quant_contract.py` bytes); the moe numeric family's tolerance_model has `K_native == K_NATIVE` and no `K_quant` key**
- [ ] **Step 2: Implement: import `QUANT_ENSEMBLE_VERSION`, `K_NATIVE`; add the two keys for wide_qmv; for the kv/moe branch use `"K_native": K_NATIVE` when `operator == "moe_dispatch"` and keep `"K_quant": K_QUANT` for kv (borrowed, per V2)**
- [ ] **Step 3: Suite; commit. Re-emission of the 26 committed certificates is M3 (needs the GPU).**

```bash
git add bench/emit_pack_certificates.py tests/test_pack_certificates.py
git commit -m "Certificates name the ensemble version and hash, and state the moe tolerance they actually apply"
```

### Task V5: Make G1 falsifiable and the K estimator honest (pre-registered amendment, then M1)

**Files:**
- Modify: `bench/calibrate_quant_device.py` docstring (AMENDMENT block, written BEFORE the code change), `k_demand()` (pre-merge lines 270-289), `main()` (pre-merge lines 416-440) - re-locate by symbol after the merge
- Test: `tests/<device calibration test file>`

**Interfaces:**
- Produces: `k_demand(records, distinct_only: bool)`; `main()` chooses `k_ship` from CALIBRATION records only, then reports `independent_miss` when `k_demand(indep) > k_ship`, and evaluates G1 on the independent records only. Both readings (with and without value-duplicate collapse) are printed and persisted.

- [ ] **Step 1: Write the pre-registration amendment into the docstring first, and commit it alone**

```markdown
AMENDMENT, 2026-08-16: G1 was not a test
--------------------------------------
Step 3 chose k_ship as max(calibration demand, independent demand) and step 4 then scored G1 over the union of both draws at that k.
Since the demand IS the maximum of heldout/floor over those records, G1 = 0 was true by construction and every ADR since 0009 reported it as evidence.
From this amendment: k_ship is chosen from the CALIBRATION draw alone; the independent draw TESTS it and never sets it.
If the independent draw's demand exceeds k_ship, that is an INDEPENDENT MISS: the cell is named, the run exits 1, and the rule is renegotiated by a human - the same stop-at-miss semantics as the serving harness's DEMAND MISS branch.
This REPLACES ADR 0012's pre-registered "take the next covering grid value and report the miss": rolling K up to cover the held-out draw would make G1 on that draw true by construction again, and the whole point of this amendment is that G1 can fail (ruling 1A of the 2026-08-16 plan review).
G1 is evaluated on the independent draw only.
Value-duplicate members (leave-one-out ratio pinned at 1.0 by a bit-identical twin) are collapsed before the demand is taken; both readings are printed.
K rising under this amendment is a stop, as before.
```

Data flow, before and after (this diagram goes into the docstring beside the amendment):

```
BEFORE (ADR 0012 as implemented)                AFTER (this amendment)
records ──┬── cal ──► k_demand ─┐                records ──┬── cal ──► k_demand ──► cover() ──► k_ship
          └── indep ► k_demand ─┴► max ► k_ship            └── indep ► k_demand ──► > k_ship ? ──► INDEPENDENT MISS, exit 1
records (both) ──► gates(k_ship) ► G1 = 0 always            indep ──► gates(k_ship) ► G1 can fail
                                                             records (both) ──► gates(k_ship) ► G2, G3
```

```bash
git add bench/calibrate_quant_device.py
git commit -m "Pre-register: G1 evaluated out of sample, duplicates collapsed, before the code moves"
```

- [ ] **Step 2: Failing tests (unit, no GPU: build records by hand with the harness's own record shape)**

```python
import json
import calibrate_quant_device as dev

def _record(seed, member_errs, heldout_err, base=1e-9):
    names = list(dev.ALL_MEMBERS)
    return {"bits": 3, "shape": "512x512", "draw": "normal-0.02", "seed": seed,
            "heldout_draw": seed in dev.HELDOUT_SEEDS, "mode": "unit", "dtype": "float32",
            "base_tol": base, "members": dict(zip(names, member_errs)),
            "heldout": {"block-tiled": heldout_err, "mlx-on-device": heldout_err},
            "faults": {name: 1.0 for name in dev.FAULTS},     # gates() reads every FAULTS name (device:325-329)
            "boundary_fp16_dequant": 0.0}

def _measured(*records):
    """What measure() returns: main() reads bit_exact FIRST (device:411)."""
    return {"records": list(records), "bit_exact": True}

def test_k_is_one_value_across_widths_chosen_from_calibration_alone():
    cal3 = [_record(0, [1.0] * 9, heldout_err=2.5)]     # demands 2.5
    cal4 = [_record(0, [1.0] * 9, heldout_err=1.2)]
    indep3 = [_record(100, [1.0] * 9, heldout_err=3.5)] # 3.5 > 3.0: a miss at bits 3
    indep4 = [_record(100, [1.0] * 9, heldout_err=1.0)]
    k_grid, misses = dev.choose_k({3: (cal3, indep3), 4: (cal4, indep4)})
    assert k_grid == 3.0, "one value across widths: max(K_CPU = 3.0, cover(2.5) = 3.0); this is the harness's reading, not native_ops.K_QUANT"
    assert set(misses) == {3} and misses[3][0] == 3.5, "the held-out excess is REPORTED per width, never folded into K"

def test_an_independent_miss_stops_the_run(monkeypatch, capsys):
    # main(argv) with a miss returns 1 and names the cell, exactly like the serving DEMAND MISS branch
    monkeypatch.setattr(dev, "choose_k", lambda per_bits: (3.0, {3: (3.5, "heldout mlx-on-device @ 512x512 ...")}))
    # gates() reads r["faults"][name] for every FAULTS entry and main() reads measured["bit_exact"] first
    monkeypatch.setattr(dev, "measure", lambda *a, **k: _measured(_record(0, [1.0] * 9, 0.0),
                                                                    _record(100, [1.0] * 9, 3.5)))
    monkeypatch.setattr(dev, "DeviceMemberSession", lambda: object())   # never touch Metal in a unit test
    assert dev.main(["--bits", "3"]) == 1
    assert "INDEPENDENT MISS: bits=3" in capsys.readouterr().out

def test_cover_is_the_one_grid_lookup():
    from calibrate_quant_bits import cover, K_GRID
    assert cover(2.766) == 3.0 and cover(3.120) == 4.0 and cover(0.5) == K_GRID[0]
    with pytest.raises(ValueError):
        cover(9.0)          # above the grid: the KILL branch, never a silent clamp

def test_choose_k_with_no_usable_width_is_a_named_verdict_not_a_crash(monkeypatch, capsys):
    assert dev.choose_k({}) == (dev.K_CPU, {})
    monkeypatch.setattr(dev, "measure", lambda *a, **k: {"records": [], "bit_exact": False})   # G0 fails
    monkeypatch.setattr(dev, "DeviceMemberSession", lambda: object())
    assert dev.main(["--bits", "3"]) == 1
    assert "no usable width" in capsys.readouterr().out

def test_g1_is_scored_on_the_independent_draw_only():
    # a calibration record with heldout at 10x floor exists in the run, but G1 never sees it:
    indep = [_record(100, [1.0] * 9, heldout_err=1.5)]
    report = dev.gates(indep, k=3.0)                    # G1 sees indep only
    assert report["gates"]["G1_no_false_positives"] is True
    assert report["false_positives"] == []

def test_the_old_reading_keys_are_still_persisted(monkeypatch, tmp_path):
    """REGRESSION (iron rule): tests/ and ADR 0016 read k_needed_calibration and
    k_needed_independent from the device JSON; both-readings continuity keeps them."""
    monkeypatch.setattr(dev, "OUT_PATH", tmp_path / "out.json")
    monkeypatch.setattr(dev, "measure", lambda *a, **k: _measured(_record(0, [1.0] * 9, 2.5),
                                                                    _record(100, [1.0] * 9, 2.0)))
    monkeypatch.setattr(dev, "DeviceMemberSession", lambda: object())
    dev.main(["--bits", "3"])
    out = json.loads((tmp_path / "out.json").read_text())["reports"]["3"]
    for key in ("k_needed_calibration", "k_needed_independent",
                "k_needed_calibration_distinct", "k_needed_independent_distinct", "independent_miss"):
        assert key in out, key

def test_k_demand_collapses_value_duplicates():
    errs = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # members 0 and 1 bit-identical
    recs = [_record(0, errs, heldout_err=0.0)]
    needed_all, _ = dev.k_demand(recs, distinct_only=False)
    needed_distinct, binding = dev.k_demand(recs, distinct_only=True)
    assert needed_all == 1.0, "a duplicate is capped at 1.0 by its twin and looks harmless"
    assert needed_distinct == 2.0 and "member" in binding, "collapsed, the pair binds at its true ratio"
```

- [ ] **Step 3: Implement (ruling OV-A: `main()` today takes no argv and builds `DeviceMemberSession()` before `measure()`, and K is decided ACROSS bit widths at line 430 - both shape the code below). `main(argv=None)` with `parser.parse_args(argv)`; the session is built inside the per-bits loop AFTER `measure` is looked up from the module, so a test can monkeypatch `measure` and `DeviceMemberSession` and never touch Metal. In `bench/calibrate_quant_bits.py` beside `K_GRID`: `def cover(demand: float) -> float: "The smallest grid value at or above the demand; above the grid is the KILL branch, raised, never clamped." return next(k for k in K_GRID if k >= demand)` wrapped to raise `ValueError` on exhaustion. In `calibrate_quant_device.py`: `choose_k(per_bits: dict[int, tuple[list, list]]) -> tuple[float, dict[int, tuple[float, str]]]` where each value is `(cal_records, indep_records)` for that width: `needed_cal = max((k_demand(cal, distinct_only=True)[0] for cal, _ in per_bits.values()), default=0.0)`; `k_grid = max(K_CPU, cover(needed_cal))` - so `choose_k({})` returns `(K_CPU, {})` (CEO ruling 3A: every width failing G0 must be a named verdict, not a ValueError; `main()` prints `no usable width: G0 failed everywhere` and returns 1 when `usable` is empty) - ONE value across widths, as today. `K_CPU` is this harness's existing import of `calibrate_quant_bits.K_QUANT` (= 3.0, Phase 0's CPU-calibrated starting K), renamed at the import for clarity; it is the floor of the harness's OWN reading, and it is deliberately NOT `native_ops.K_QUANT` (4.0, what ships): V1 already separates the two as `k_grid` and `k_shipped` in the JSON, and ADR 0016 records why the shipped value is decided across every harness's records, not here. `misses = {bits: (needed, binding) for bits, (_, indep) in per_bits.items() for needed, binding in [k_demand(indep, distinct_only=True)] if needed > k_grid}`; return `(k_grid, misses)`. In `main()`: `k_grid, misses = choose_k({b: (cal_b, indep_b) for b in usable})`; if `misses`: print one `INDEPENDENT MISS: bits={b} held-out demand {needed:.3f} > device-grid K {k_grid} at {binding}` line per width and `return 1` after persisting the records (the stop-at-miss semantics); else per width G1 from `gates(indep_b, k_grid)`, G2/G3 from `gates(records_b, k_grid)`; persist `k_needed_calibration_distinct`, `k_needed_independent_distinct`, `independent_miss`, and keep the old two keys for both-readings continuity. In `k_demand(records, distinct_only=False)`: when `distinct_only`, build `signature = {m: tuple(round(r["members"][m], 17) for r in records) for m in ALL_MEMBERS}`, keep the first name per distinct signature, and run the existing leave-one-out over the kept names only. Replace the inline `next(k for k in K_GRID ...)` at the old line 433 with `cover`.**
- [ ] **Step 4: Suite green (CPU); commit. Then M1 re-runs the grid (7 min) and the reading goes into ADR 0017.**

```bash
git add bench/calibrate_quant_device.py tests/test_quant_device_members.py
git commit -m "G1 is scored on the held-out draw; K comes from calibration; duplicates cannot bind"
```

### Task V6: The kv_attention reference is cross-checked against something it does not share code with

**Files:**
- Modify: `tests/test_native_ops.py` (`test_kv_kernel_agrees_with_reference` and its neighbours)
- Read first: `kernelverify/reference/native_kernels.py` (`kv_attention`, `_cache_dequant`), `kernelverify/schemas/native_ops.py` (`kv_reference`)

The claims audit found the kv reference's cache-dequant half is checked only against the project's own numpy kernel, which shares `_cache_dequant` with it - a shared misreading of the quantized cache passes.
The attention half is already cross-checked against `mx.fast.scaled_dot_product_attention`; the dequant half needs the same independence.

- [ ] **Step 1: Failing test. `_cache_dequant(cache, bits, fault=None)` takes the RAW cache and quantizes then dequantizes internally (ruling OV-A; group size fixed inside), so the independent check is MLX's own quantize-then-dequantize on the same raw cache: this checks both halves against code that shares nothing with `_cache_dequant`.**

```python
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_cache_dequant_matches_mlx_quantize_dequantize_bit_for_bit(bits):
    mx = pytest.importorskip("mlx.core")
    from kernelverify.reference.native_kernels import _cache_dequant
    rng = np.random.default_rng(3)
    cache = (rng.standard_normal((4, 128, 256)) * 0.05).astype(np.float16)   # (heads, positions, dim)
    ours = _cache_dequant(cache, bits)
    wq, scales, biases = mx.quantize(mx.array(cache.reshape(-1, 256)), group_size=64, bits=bits)
    theirs = np.array(mx.dequantize(wq, scales, biases, group_size=64, bits=bits)).reshape(cache.shape)
    assert np.array_equal(ours.astype(np.float32), theirs.astype(np.float32))
```

- [ ] **Step 2: Run; if `_cache_dequant`'s internal group size is not 64, read it and match it in the `mx.quantize` call - the assertion is the point; if it fails on VALUES, that is a finding: stop, report, do not "fix" the test**
- [ ] **Step 3: Suite; commit**

```bash
git add tests/test_native_ops.py
git commit -m "The kv cache dequant is checked against MLX's own dequantize, not against code it shares"
```

---

# Workstream I - Infrastructure

Executor: one worker on branch `infra/measurement-discipline` off main after the runner merge (I1 edits `serve_sub4bit.py`, which lives on metal-runner).

### Task I1: Lock, budget and memory gate on `serve_sub4bit.py`, copying `price_qmv_boundary.py`'s pattern - EXPANDED 2026-08-16

> **Scope change, on record before any lane runs.** The merge gate on `metal-runner` (merge `0609088`) found twelve defects in this harness, all verified in the coordinator session and all recorded in that merge's commit body. Vlad ruled option A: merge as-is (the file is inert on main; nothing imports it) and grow THIS task to cover the lot in one pass, rather than a second task that can slip or a pre-merge fix the task would rewrite anyway. Steps 1-3 below are the original scope (the machine); steps 4-7 are the expansion (the experiment, the registration, the discipline). **M2 does not run until every step here has landed** - the untested branches of the decision surface are exactly the ones a losing kernel triggers.

**Files:**
- Modify: `bench/serve_sub4bit.py` `main()` (838-859), `mde()` (642-702), `ab()` (730-811), `primary_verdict()` (705), `composed_attribution()` (716), `_op_probe()` (594-640), `ppl()` (~815-830), `_WHITELIST_PREFIXES`, the eligibility check
- Modify: `docs/research/2026-08-15-sub4bit-serve-findings.md` (a WRITTEN amendment, step 6)
- Test: `tests/test_serve_sub4bit.py`

**Interfaces:**
- Consumes: `bench/memory_guard.py` (`MeasurementLock` from machine_state; `BudgetGuard`, `require_available_memory`, `budget_gb_arg`, `EXIT_LOCK_HELD`, `EXIT_BUDGET_REFUSAL`, `EXIT_LOW_MEMORY`)
- Produces: `--budget-gb` (default `AB_BUDGET_GB = 24.0`, the same value as the serving calibration's `DEFAULT_BUDGET_GB`; serve_sub4bit does not import from that module), lock name `"serve_sub4bit"`, typed refusals `NotIdle` / `PreconditionFailed`; smoke stays lock-free.

- [ ] **Step 1: Failing tests**

```python
import pytest
import serve_sub4bit as s

TIMED = pytest.mark.parametrize("mode", ["--ab", "--mde"])   # ruling 7A: both timed modes, same guards

def _never_load(*a, **k):
    raise AssertionError("a model was loaded before the gate refused")

@TIMED
def test_timed_modes_refuse_when_the_machine_lock_is_held(monkeypatch, mode):
    monkeypatch.setattr(s, "verify_pins", lambda *a, **k: {"pins": "ok"})
    monkeypatch.setattr(s.MeasurementLock, "acquire", lambda self: (False, "held by pid 1 (test)"))
    monkeypatch.setattr(s, "load_model", _never_load)
    assert s.main([mode]) == s.EXIT_LOCK_HELD

def test_smoke_never_takes_the_lock(monkeypatch):
    monkeypatch.setattr(s, "verify_pins", lambda *a, **k: {"pins": "ok"})
    def _boom(self):
        raise AssertionError("smoke must not take the machine lock")
    monkeypatch.setattr(s.MeasurementLock, "acquire", _boom)
    monkeypatch.setattr(s, "smoke", lambda: 0)          # the wiring check itself is covered elsewhere
    assert s.main(["--smoke"]) == 0

def _raise(exc):
    def _f(*a, **k):
        raise exc
    return _f

@TIMED
def test_permanent_preconditions_exit_with_their_own_code(monkeypatch, mode):
    monkeypatch.setattr(s, "verify_pins", _raise(RuntimeError("hash mismatch")))
    assert s.main([mode]) == s.EXIT_PRECONDITION
    monkeypatch.setattr(s, "verify_pins", lambda *a, **k: {"pins": "ok"})
    monkeypatch.setattr(s.MeasurementLock, "acquire", lambda self: (True, "acquired"))
    monkeypatch.setattr(s.MeasurementLock, "release", lambda self: None)
    monkeypatch.setattr(s, "require_idle", lambda label: {"idle": True})
    monkeypatch.setattr(s, "require_pinned_zone", _raise(s.PreconditionFailed("zone disagrees")))
    monkeypatch.setattr(s, "load_model", _never_load)
    assert s.main([mode]) == s.EXIT_PRECONDITION

@TIMED
def test_a_busy_machine_is_transient(monkeypatch, mode):
    monkeypatch.setattr(s, "verify_pins", lambda *a, **k: {"pins": "ok"})
    monkeypatch.setattr(s.MeasurementLock, "acquire", lambda self: (True, "acquired"))
    monkeypatch.setattr(s.MeasurementLock, "release", lambda self: None)
    monkeypatch.setattr(s, "require_idle", _raise(s.NotIdle("WindowServer at 30%")))
    monkeypatch.setattr(s, "load_model", _never_load)
    assert s.main([mode]) == s.EXIT_NOT_IDLE

@TIMED
def test_timed_modes_refuse_when_the_budget_is_crossed(monkeypatch, mode):
    monkeypatch.setattr(s, "verify_pins", lambda *a, **k: {"pins": "ok"})
    monkeypatch.setattr(s.MeasurementLock, "acquire", lambda self: (True, "acquired"))
    monkeypatch.setattr(s.MeasurementLock, "release", lambda self: None)
    monkeypatch.setattr(s, "require_idle", lambda label: {"idle": True})
    monkeypatch.setattr(s, "require_pinned_zone", lambda: None)
    monkeypatch.setattr(s.BudgetGuard, "check", lambda self, cell: (_ for _ in ()).throw(
        s.BudgetExceeded(cell, 30.0, 24.0)))
    monkeypatch.setattr(s, "load_model", _never_load)   # the budget is checked before the first load
    assert s.main([mode, "--budget-gb", "24"]) == s.EXIT_BUDGET_REFUSAL
```

- [ ] **Step 2: Implement exactly the `price_qmv_boundary` ordering: parse -> `verify_pins` (a mismatch returns `EXIT_PRECONDITION`, ruling 3A) -> if smoke: return smoke() (no lock) -> `lock = MeasurementLock("serve_sub4bit"); acquired, detail = lock.acquire(); if not acquired: print refusal; return EXIT_LOCK_HELD` -> `try:` idle gate INSIDE the lock, then the pinned-zone check, then `mde`/`ab`. Today `require_idle` and `require_pinned_zone` `raise SystemExit(2)` (serve_sub4bit:424, 453) and are called INSIDE `mde()`/`ab()` (:646-647, :734-735); change them to raise typed refusals `NotIdle(RuntimeError)` and `PreconditionFailed(RuntimeError)`, move the two calls out of `mde()`/`ab()` into `main()`'s locked block so they run once, and let `main()` catch: `except NotIdle -> EXIT_NOT_IDLE` (transient, the runner waits - ruling OV-1: exit 2 stays the interpreter's, as `memory_guard.py` lines 32-40 already ruled), `except PreconditionFailed -> EXIT_PRECONDITION` (permanent, the runner gives up); `verify_pins`' RuntimeError also maps to `EXIT_PRECONDITION`. Then `mde`/`ab` with `guard = BudgetGuard(args.budget_gb)` checked at the start of every B cell and after every arm, `require_available_memory(remaining, cell)` beside it; `except BudgetExceeded -> EXIT_BUDGET_REFUSAL`, `except LowMemoryRefusal -> EXIT_LOW_MEMORY` -> `finally: lock.release()`. In `bench/memory_guard.py` add `EXIT_PRECONDITION = 8   # a permanent precondition failed (pins, pinned zone): retrying cannot help` and `EXIT_NOT_IDLE = 9   # the machine is not quiet: transient, come back` to the exit table. One docstring line in serve_sub4bit names the vocabulary: 9 transient (idle), 3/4/5 memory, 8 permanent, 2 the interpreter's (argparse). Amend the findings doc's T8 line from "exit 2" to `EXIT_NOT_IDLE`.**
- [ ] **Step 3: Suite; commit**

```bash
git add bench/serve_sub4bit.py tests/test_serve_sub4bit.py
git commit -m "The A/B harness takes the machine lock and a footprint budget before it times anything"
```

- [ ] **Step 4 (expansion): the decision surface gets tests, both failing branches first**

`primary_verdict` and `composed_attribution` are the entire decision surface of the experiment and have zero test coverage across 460 lines of tests. Write these BEFORE touching the functions, and write the losing branches first, because wide_qmv is a measured LOSS at batch 1-3 and those are the branches M2 will actually take:

```python
import pytest
import serve_sub4bit as s

# primary_verdict(prim_pct, noise_pct): arm 1 vs arm 2, qualified by arm 2's own spread.
@pytest.mark.parametrize("prim, noise, expected", [
    (-3.0, 1.0, "regression"),   # the branch wide_qmv is expected to hit at B = 1..3
    (-0.5, 1.0, "null"),         # a loss inside the noise floor is NOT a regression
    ( 0.5, 1.0, "null"),         # nor is a gain inside it a win
    ( 3.0, 1.0, "win"),
    ( 1.0, 1.0, "null"),         # the boundary belongs to null: a gap the noise could produce proves nothing
    (-1.0, 1.0, "null"),
])
def test_primary_verdict_reads_section_6(prim, noise, expected):
    assert s.primary_verdict(prim, noise) == expected

# composed_attribution(comp_pct, base_pct, noise_pct): arm 1 vs arm 3, with arm 2 vs arm 3
# deciding whether a win was the artifact's alone or one the kernel unlocked.
@pytest.mark.parametrize("comp, base, noise, expected", [
    (-3.0, 0.0, 1.0, "negative"),        # composed loses outright
    (-0.5, 0.0, 1.0, "inconclusive"),    # inside the floor
    ( 3.0, 2.5, 1.0, "artifact-alone"),  # arm 2 already beat arm 3 beyond noise: the kernel added nothing
    ( 3.0, 0.5, 1.0, "joint"),           # arm 2 did not; the kernel unlocked it
    ( 0.0, 5.0, 1.0, "inconclusive"),    # comp inside the floor: base_pct is never consulted
])
def test_composed_attribution_reads_section_6(comp, base, noise, expected):
    assert s.composed_attribution(comp, base, noise) == expected

def test_the_verdict_can_say_loss():
    """The check this repo learned to write after G1: a decision function that
    can only confirm is not a decision. Both losing outcomes must be reachable."""
    assert s.primary_verdict(-10.0, 0.1) == "regression"
    assert s.composed_attribution(-10.0, 0.0, 0.1) == "negative"
```

Run: `.venv/bin/python -m pytest -q tests/test_serve_sub4bit.py -k "verdict or attribution"` - expected PASS against the current functions (they are correct; they were untested). If any case FAILS, that is a finding about the harness, not about the test: stop and report before changing either.

- [ ] **Step 5 (expansion): the pre-registered rules the code dropped**

  - **The non-decider rule.** The findings doc, section 5: *"a cell whose expected gain is below its noise floor is a pre-declared non-decider"*. `mde()` prints `arm2_noise_floor_pct` (:699) and `expected_gain_pct` (:701) adjacently and never compares them, so the MDE is derived and then ignored. Add to the `--mde` row: `"decider": gain_pct > noise_floor_pct` and a per-cell `NON-DECIDER` label on stdout; make `ab()` READ the MDE's output (a JSON file `bench/.cache/serve_mde.json` written by `--mde`, consumed by `--ab`, refusing with `EXIT_PRECONDITION` if absent or stale by pins) and mark every non-decider cell's rows `"decider": false` so the reader cannot mistake its verdict for evidence. Test: a synthetic MDE file with one non-decider cell -> that cell's `--ab` rows carry `"decider": false` and the run's summary line names it.
  - **The corpus pin.** Section 7 registers sha256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`; `ppl()` computes a digest (:822), prints it (:829) and never compares. Add `PPL_CORPUS_SHA256 = "173c87a5..."` beside the other pins and refuse with `EXIT_PRECONDITION` on mismatch, BEFORE any perplexity is computed. The pinned file `bench/.corpus/ppl.txt` does not exist in this worktree (the only copy is under `/Users/vlad/kv-baseline`); copy it in, gitignored like `bench/.models`, and record in the findings doc that it lives in the main worktree only. Test: a tmp corpus with the wrong bytes -> `EXIT_PRECONDITION`, no model loaded.
  - **The sample count.** `_op_probe` takes `range(7)` samples (:634) where section 3 registers `ROUNDS = 5`. Use `ROUNDS`. If 7 was deliberate, that is a registration change and goes through step 6, not through a literal.
  - **The reference-arm spread gate in `--mde`.** `_op_probe` returns `stock_spread_pct` and nothing rejects on it, while `ab()` does gate. AGENTS.md: *"reject any round whose reference samples exceed the class spread limit"*. Apply the same rejection in `--mde`; its per-op numbers feed `expected_gain_pct`, which feeds the non-decider rule above.

- [ ] **Step 6 (expansion): the registration is amended IN WRITING, never silently**

Three things the code does that section 4 of the findings doc does not register: a `bias-term` exclusion in eligibility, a `forced-stock` entry in `_WHITELIST_PREFIXES` (the doc says *"the only whitelisted reasons are `prefill-*` and `m-*-outside-dispatch`"*, and the code whitelists any `m-` reason plus `forced-stock`), and `model.set_dtype(mx.float16)` casting both checkpoints. Each is an unregistered degree of freedom in a measurement harness. For each one, EITHER remove it from the code OR add an amendment paragraph to the findings doc, dated, stating what it is, why it is there, and what it could bias - the way every changed pre-registration in this repo is handled (ADR 0012, 0013, 0014 amendments are the pattern). Then make the code and the doc agree exactly: `_WHITELIST_PREFIXES` becomes `("prefill-", "m-", "forced-stock")` ONLY IF the doc now says so, and the `m-*-outside-dispatch` narrowing is either enforced (`reason.startswith("m-") and reason.endswith("-outside-dispatch")`) or the doc is widened. Test: `test_whitelist_matches_the_registration` reads the doc's whitelist line by regex and asserts it equals `_WHITELIST_PREFIXES` - so the two cannot drift again.

- [ ] **Step 7 (expansion): one copy of the timing discipline**

`MIN_SAMPLE_MS = 5.0` (:128), `spread_pct` (:465) and the interleave-and-batch loop in `_op_probe` (`while copies < 4096 ...`, :631) re-implement `bench/interleave.py`'s `dispatch` / `calibrate_copies` and `machine_state.spread_pct`. AGENTS.md: *"One copy of the discipline, so a timing rule amended in one gate cannot silently stay old in another."* Import them; delete the copies. `_op_probe` composes `saving_us` from its own interleaved probe with `t_step_ms` from a SEPARATE decode pass minutes apart - the cross-pass composition the interleave rule exists to forbid - so `expected_gain_pct` must either be computed from a single interleaved pass or be labelled in the row as `"composition": "cross-pass"` with the doc's section 5 amended to say the MDE is an upper bound for that reason too. Test: `inspect.getsource(s)` contains no `def spread_pct` and no `MIN_SAMPLE_MS =`.

- [ ] **Step 8: Suite; commit (one commit per step is fine; the last one carries the doc amendment)**

```bash
git add bench/serve_sub4bit.py tests/test_serve_sub4bit.py docs/research/2026-08-15-sub4bit-serve-findings.md
git commit -m "The A/B harness can say loss, obeys its own MDE, checks its corpus pin, and matches its registration"
```

### Task I2: Generalise the detached runner so any long harness can run overnight

**Files:**
- Modify: `bench/detached_run.py` (HARNESS at 44; `newest_run_rows` 100-109; `main` 112-170; STATUS 43)
- Modify: `bench/start_binding_run.sh` (LABEL 15, LOG 17, ProgramArguments 28-34)
- Modify: `bench/measure_baselines.py`, `bench/spike_mlx_e2e.py` (add MeasurementLock, ruling 2A)
- Modify: `bench/OPERATOR-CARD.md`
- Test: `tests/test_detached_run.py` (create)

**Interfaces:**
- Produces: `detached_run.py --harness <path> [--protocol baselines|exit-code] [-- <harness args>]`; status file `bench/.baselines/detached_status-<harness-stem>.json`; `start_binding_run.sh <harness> [args]` derives `LABEL=com.kernelverify.detached-<stem>`.
- The runner NEVER takes the machine lock (ruling 2A): a child cannot acquire the flock its parent holds (verified live), so a runner that locked would make every self-locking harness refuse with EXIT_LOCK_HELD and loop to the deadline. One owner per lock: every heavy harness locks itself. This task therefore also adds `MeasurementLock` to the two harnesses that gate on idleness without one - `bench/measure_baselines.py` and `bench/spike_mlx_e2e.py` - using the `price_qmv_boundary.py` ordering (lock, then idle gate inside it, `finally: release`).

- [ ] **Step 1: Failing tests using a fake harness script written by the test into tmp_path that exits with a chosen code**

```python
import json, sys
from pathlib import Path
import detached_run as dr
from memory_guard import EXIT_BUDGET_REFUSAL, EXIT_LOCK_HELD, EXIT_NOT_IDLE, EXIT_PRECONDITION

def _fake_harness(tmp_path, code: int):
    p = tmp_path / "fake_harness.py"
    p.write_text(f"import sys; sys.exit({code})\n")
    return p

def test_exit_code_protocol_maps_lock_held_back_to_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "wait_for_strong_idle", lambda deadline: True)
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    outcome = dr.judge_exit_code(EXIT_LOCK_HELD)
    assert outcome == ("waiting", False), "a held lock is not an attempt consumed"

def test_exit_code_protocol_gives_up_on_budget_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    assert dr.judge_exit_code(EXIT_BUDGET_REFUSAL) == ("gave-up", True)
    assert dr.judge_exit_code(EXIT_PRECONDITION) == ("gave-up", True), "pins/zone mismatch cannot heal by waiting"
    assert dr.judge_exit_code(EXIT_NOT_IDLE) == ("waiting", False)
    assert dr.judge_exit_code(0) == ("done", True)
    assert dr.judge_exit_code(1) == ("stopped", True), "a deliberate return 1 is a verdict"
    assert dr.judge_exit_code(1, stderr_tail="...\nTraceback (most recent call last):\n  File x\nValueError: boom") == ("crashed", True), \
        "an uncaught exception also exits 1; the traceback tells them apart (CEO ruling 1A)"

def test_a_crashing_harness_leaves_its_traceback_in_the_status_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "wait_for_strong_idle", lambda deadline: True)
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    p = tmp_path / "boom.py"; p.write_text("raise ValueError('boom')\n")
    assert dr.main(["--harness", str(p), "--protocol", "exit-code"]) == 4
    status = json.loads((tmp_path / "detached_status-boom.json").read_text())
    assert status["state"] == "crashed" and "ValueError: boom" in status["stderr_tail"]
    assert dr.judge_exit_code(2) == ("crashed", True), "argparse's 2 is a typo, not a busy machine (memory_guard's ruling)"
    assert dr.judge_exit_code(99) == ("crashed", True)

def test_status_file_is_per_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    dr.write_status("waiting", harness_stem="serve_sub4bit")
    assert (tmp_path / "detached_status-serve_sub4bit.json").exists()
    assert json.loads((tmp_path / "detached_status-serve_sub4bit.json").read_text())["state"] == "waiting"

def test_the_runner_never_takes_the_machine_lock(tmp_path, monkeypatch):
    """Ruling 2A: harnesses own the lock. A runner that locked would make every
    self-locking harness refuse with EXIT_LOCK_HELD and loop to the deadline."""
    monkeypatch.setattr(dr, "wait_for_strong_idle", lambda deadline: True)
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    assert "MeasurementLock" not in Path(dr.__file__).read_text()
    code = dr.main(["--harness", str(_fake_harness(tmp_path, 0)), "--protocol", "exit-code"])
    assert code == 0

def test_a_harness_holding_the_lock_makes_the_detached_run_wait(tmp_path, monkeypatch):
    # the fake harness exits EXIT_LOCK_HELD, as a real one would when a foreground run holds the lock;
    # the strong-idle wait returns True once, then False (deadline) - the deadline lives inside that function
    idle = iter([True, False])
    monkeypatch.setattr(dr, "wait_for_strong_idle", lambda deadline: next(idle))
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    code = dr.main(["--harness", str(_fake_harness(tmp_path, EXIT_LOCK_HELD)), "--protocol", "exit-code"])
    assert code == 3                                     # gave up waiting; the lock-held attempt was never consumed
    status = json.loads((tmp_path / "detached_status-fake_harness.json").read_text())
    assert status["state"] == "gave-up" and status["attempts_consumed"] == 0

def test_baselines_protocol_is_the_default_and_unchanged(monkeypatch):
    args = dr.parse_args([])
    assert args.protocol == "baselines"
    assert args.harness == dr.ROOT / "bench" / "measure_baselines.py"
```

- [ ] **Step 2: Implement. `parse_args(argv) -> Namespace(harness: Path = ROOT/"bench"/"measure_baselines.py", protocol: str = "baselines", harness_args: list[str])` (everything after `--`). `STATUS_DIR = BASELINES`; `write_status(state, harness_stem, **extra)` writes `STATUS_DIR / f"detached_status-{harness_stem}.json"` and always carries `attempts_consumed`. `judge_exit_code(code, stderr_tail: str = "") -> tuple[str, bool]` returns `(state, attempt_consumed)`: `0 -> ("done", True)`; `1 -> ("crashed", True)` if `"Traceback" in stderr_tail` else `("stopped", True)` (CEO ruling 1A: Python exits 1 on an uncaught exception AND memory_guard's vocabulary uses 1 for measured-and-stopped, so the traceback is the only thing that tells a crash from a verdict); the runner captures the harness's stderr (`subprocess.run(..., stderr=PIPE, text=True)`, streaming stdout to the log as today) and writes its last 40 lines into the status file as `stderr_tail` on every non-zero exit; `EXIT_NOT_IDLE | EXIT_LOCK_HELD | EXIT_LOW_MEMORY -> ("waiting", False)`; `EXIT_BUDGET_REFUSAL | EXIT_PRECONDITION -> ("gave-up", True)`; else including bare `2` -> `("crashed", True)` (ruling OV-1: an argparse typo fails in one attempt, never a 12-hour wait). The BASELINES protocol path is kept verbatim EXCEPT that it consults `judge_exit_code` first for the waiting codes, because `measure_baselines.py` gains the lock in this task and its `EXIT_LOCK_HELD` must read as "wait", not as the "no rows appended -> CRASHED" branch at detached_run:143-149. `main(argv)` keeps today's baselines path verbatim (`newest_run_rows`, `binding` rows) when `protocol == "baselines"`, and for `"exit-code"` runs `subprocess.run([sys.executable, str(harness), *harness_args], cwd=ROOT).returncode` through `judge_exit_code`, returning 0 for done, 3 for gave-up, 4 for crashed. The runner takes NO lock (ruling 2A). Add `MeasurementLock` to `bench/measure_baselines.py` (around `main()`'s measuring body, after argparse, before its idle gate) and to `bench/spike_mlx_e2e.py` the same way, each with a test asserting the timed path returns `EXIT_LOCK_HELD` when `acquire` is monkeypatched to fail. `start_binding_run.sh <harness> [-- args]` sets `LABEL=com.kernelverify.detached-<stem>`, `LOG=bench/.baselines/detached-<stem>.log`, and passes `--harness --protocol exit-code -- args` unless the harness is `measure_baselines.py`. Update OPERATOR-CARD with `bench/start_binding_run.sh bench/calibrate_quant_serving.py` and `bench/start_binding_run.sh bench/serve_sub4bit.py -- --ab`.**
- [ ] **Step 3: `start_binding_run.sh` gains `--print-plist` (write the plist to stdout, do not bootout/bootstrap), ruling 5A, and XML-escapes every interpolated value (`&` `<` `>` `"` `'`) before it lands in a `<string>` (CEO ruling 2A: an unescaped `&` in an argument makes launchd reject the plist silently). Test in `tests/test_detached_run.py`:**

```python
import subprocess

def test_start_script_derives_label_log_and_args_from_the_harness():
    root = Path(dr.ROOT)
    default = subprocess.run(["zsh", str(root / "bench/start_binding_run.sh"), "--print-plist"],
                             capture_output=True, text=True, cwd=root).stdout
    assert "com.kernelverify.detached-measure_baselines" in default
    assert "detached_run.py" in default and "--protocol" not in default   # baselines is the default protocol
    ab = subprocess.run(["zsh", str(root / "bench/start_binding_run.sh"), "--print-plist",
                         "bench/serve_sub4bit.py", "--", "--ab"], capture_output=True, text=True, cwd=root).stdout
    assert "com.kernelverify.detached-serve_sub4bit" in ab
    assert "<string>--harness</string>" in ab and "<string>--protocol</string>" in ab and "<string>exit-code</string>" in ab
    assert "<string>--ab</string>" in ab and "detached-serve_sub4bit.log" in ab
    escaped = subprocess.run(["zsh", str(root / "bench/start_binding_run.sh"), "--print-plist",
                              "bench/serve_sub4bit.py", "--", "--corpus", "a&b<c>"], capture_output=True, text=True, cwd=root).stdout
    assert "<string>a&amp;b&lt;c&gt;</string>" in escaped, "arguments are XML-escaped (CEO ruling 2A)"
```

- [ ] **Step 4: Suite; commit. Record in AGENTS.md (H1) that the detached path is now the way every long run executes, replacing the unwritten "coordinator runs everything" rule.**

```bash
git add bench/detached_run.py bench/start_binding_run.sh bench/measure_baselines.py bench/spike_mlx_e2e.py bench/memory_guard.py bench/OPERATOR-CARD.md tests/test_detached_run.py
git commit -m "Any long harness can run detached: per-harness status, exit-code protocol; harnesses own the lock, the runner never does"
```

### Task I3: Fix the order-dependent footprint test and give the suite a fast subset

**Files:**
- Modify: `tests/test_serving_survival.py:48-56`
- Create: `pyproject.toml` (pytest config only)
- Modify: `tests/test_quant_contract_members.py` (mark the two >45 s tests `slow`)

- [ ] **Step 1: Rewrite the test to allocate through a fresh anonymous mapping, not numpy's allocator, and collect first**

```python
def test_footprint_reader_reports_this_process_truthfully():
    import gc, mmap
    gc.collect()
    current, peak = phys_footprint_gb()
    assert 0.0 < current < machine_ram_gb()
    assert peak >= current * 0.99
    size = 512 * 1024 * 1024
    region = mmap.mmap(-1, size)            # new mapping: pages cannot be pre-charged
    try:
        region.write(b"\x01" * size)         # touch every page
        grown, grown_peak = phys_footprint_gb()
        assert grown - current > 0.4, "the reader must see our own allocation"
        assert grown_peak >= grown * 0.99
    finally:
        region.close()                      # a failing assert must not leak 512 MB into the next test
```

- [ ] **Step 2: Add `pyproject.toml`**

```toml
[tool.pytest.ini_options]
markers = [
  "slow: recomputes serving-grid blocks from committed evidence (tens of seconds)",
  "gpu: needs a Metal device",
]
```

- [ ] **Step 3: Mark `test_k_stays_four_and_the_member_now_carries_its_class_alone` and `test_the_shipped_verifier_flags_no_correct_kernel_after_the_repair` with `@pytest.mark.slow`; mark every test that builds a Metal kernel or dispatches through `MetalRunner` with `@pytest.mark.gpu` (the `needs_metal` tests in test_serving_survival.py, the kernel fixtures in test_pack_wide_qmv.py / test_pack_kv_attention.py / test_pack_moe_dispatch.py / test_pack_certificates.py / test_metal_runner.py, and V6's new test), so `pytest -m "not gpu"` is what a lane runs while a measurement holds the machine - the one-measurement-at-a-time rule becomes enforceable against the suite. Run `pytest -m "not slow"` and quote its time in AGENTS.md (H1)**
- [ ] **Step 4: Run the full suite twice in different orders (`-p no:randomly` is not installed; use `pytest tests/test_serving_survival.py::test_footprint_reader_reports_this_process_truthfully` after `pytest tests/test_serving_survival.py -k buffer_pool`) - both green; commit**

```bash
git add tests/test_serving_survival.py pyproject.toml tests/test_quant_contract_members.py
git commit -m "The footprint test allocates a fresh mapping; the suite has slow and gpu markers"
```

### Task I4: Progress lines lead with phys_footprint; RSS stops being the headline

> **Regeneration obligation (added 2026-08-16).** This task edits `bench/calibrate_quant_serving.py`, which is one of the modules `bench/derive_repaired_member_column.py` hashes into ADR 0016's committed column. `tests/test_repaired_member_artifact.py` will go red the moment this lands and stay red until the column is regenerated: `.venv/bin/python -u bench/derive_repaired_member_column.py`, ~70 min, CPU only, budget 30 GB, coordinator-run under the lock. Batch it: land I4 and I5 back to back and regenerate ONCE after both, not after each. The hash is whole-file on purpose - it can only over-fire, and over-firing costs a run and says so.

**Files:**
- Modify: `bench/calibrate_quant_serving.py` (every `rss_line()` call site listed in the exploration: 1307, 1319, 1349, 1432, 1466, 1987, 2228 on the contract branch - re-locate after merge) -> `footprint_line()`; the `print(f"peak footprint: {rss_line()}")` mislabel becomes `phys_footprint_gb()`'s peak
- Test: `tests/test_serving_survival.py`

- [ ] **Step 1: Failing tests. `footprint_line()` already exists and already leads with "footprint" (serving:683), so the change to pin is that the seven RSS-only call sites are gone and the final summary reports the phys peak:**

```python
import inspect
import calibrate_quant_serving as harness

def test_no_progress_line_prints_rss_alone():
    src = inspect.getsource(harness)
    # the only remaining caller of rss_line() is footprint_line() itself; the JSON field uses rss_gb()
    assert src.count("rss_line()") == 1, "every progress line goes through footprint_line()"

def test_the_final_summary_reports_the_footprint_peak_not_ru_maxrss():
    line = harness.summary_line({"footprint_peak_gb": 22.4, "rss_peak_gb": 2.4})
    assert line.startswith("peak footprint: 22.4") and "rss 2.4" in line
```

- [ ] **Step 2: Extract `summary_line(meta) -> str` from the print at the old line 2228 (`peak footprint: {rss_line()}` - a mislabel: it prints ru_maxrss), make it read `phys_footprint_gb()`'s peak first and RSS second; replace the seven `rss_line()` call sites with `footprint_line()`; keep `rss_gb()` for the JSON's `rss_gb` field only; commit**

```bash
git add bench/calibrate_quant_serving.py tests/test_serving_survival.py
git commit -m "Every progress line prints the number Jetsam kills on; RSS is a JSON footnote"
```

### Task I5: Close the orphan hole (child dies with its parent)

> **Regeneration obligation (added 2026-08-16).** This task edits `bench/calibrate_quant_serving.py`, which is one of the modules `bench/derive_repaired_member_column.py` hashes into ADR 0016's committed column. `tests/test_repaired_member_artifact.py` will go red the moment this lands and stay red until the column is regenerated: `.venv/bin/python -u bench/derive_repaired_member_column.py`, ~70 min, CPU only, budget 30 GB, coordinator-run under the lock. Batch it: land I4 and I5 back to back and regenerate ONCE after both, not after each. The hash is whole-file on purpose - it can only over-fire, and over-firing costs a run and says so.

**Files:**
- Modify: `bench/calibrate_quant_serving.py` `spawn_measurement` and `child_main` (locate with `grep -n "def spawn_measurement\|def child_main"`)
- Test: `tests/test_serving_survival.py`

- [ ] **Step 1: Failing test: run `child_main` with a task whose record loop checks `os.getppid()`; monkeypatch `os.getppid` to return 1 after the first record; assert the child exits with `EXIT_CHILD_DEATH`-style code and writes no further records**
- [ ] **Step 2: Implement: parent starts the child with `start_new_session=True`; in the child's per-record loop, `if os.getppid() != launching_ppid: exit(EXIT_CHILD_DEATH)`; parent's `finally` sends `SIGTERM` to the child's process group if it is still alive**
- [ ] **Step 3: Suite; commit; remove the TODOS entry "A parent killed mid-child orphans that child"**

```bash
git add bench/calibrate_quant_serving.py tests/test_serving_survival.py TODOS.md
git commit -m "A measurement child dies with its parent: no orphan keeps the GPU behind a freed lock"
```

---

# Workstream H - Hygiene: make the record match the code

Executor: one worker on branch `hygiene/records` off main after both merges.

### Task H1: AGENTS.md - fix every stale fact and document what exists

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Apply this exact list**
  - L22 catalogue "(45 entries)" -> the live count from `python -c "from kernelverify.mutation.catalogue import CATALOGUE; print(len(CATALOGUE))"` (65 today).
  - "Empty packages (battery, detectors, runners, ...)" -> "Empty packages: `corpus`, `detectors`" and remove the rest.
  - The 100% claim line: point at ADR 0008 AND the committed rerun log from H2.
  - The `calibrate_quant_serving.py` Running line: one rule - "must reproduce ADR 0013's records; interpretation per ADR 0014/0016".
  - Test count line: the fast-subset time and the full count, both quoted from a run.
  - Layout: add one line each for `kernelverify/pack/` (three hand-written Metal kernels: wide_qmv live, kv_attention demoted, moe_dispatch unused on the target model), `kernelverify/runners/`, `kernelverify/extraction/`, `kernelverify/report/`, `kernelverify/battery/`, `kernelverify/schemas/native_ops.py`, `bench/emit_pack_certificates.py`, `bench/pack_wide_qmv.py`, `bench/pack_kv_attention.py`, `bench/pack_moe_dispatch.py`, `bench/serve_sub4bit.py`, `bench/detached_run.py` (post-I2 role).
  - Running: add `bench/emit_pack_certificates.py` and the detached invocation from I2.
  - Working rules: replace the unwritten "coordinator runs every measurement" with "every long run executes through `bench/start_binding_run.sh <harness>`; interactive sessions are the contention (ADR 0010)".
  - Mistakes: add "A harness's printed K is its own reading" (already there), and "RSS is not the number" pointing at I4.
- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "AGENTS.md matches the tree: counts, empty packages, layout of the pack half, one rule per command"
```

### Task H2: The 100% claim rests on a committed record

**Files:**
- Create: `bench/results/score_oracles-2026-08-15.txt` (copy of `bench/.cache/score_oracles_rerun_20260815.log`)
- Modify: `docs/adr/0008-native-operators.md` (one amendment paragraph: 65 synthesised / 58 viable / 7 equivalent, 100% at B=16 and 32 exact, dated, pointing at the file)

- [ ] **Step 1: Copy, amend, commit**

```bash
git add bench/results/score_oracles-2026-08-15.txt docs/adr/0008-native-operators.md
git commit -m "The battery's 100% claim points at a committed rerun, not a cache log"
```

### Task H3: The record cannot drift silently again (ADR numbering, AGENTS.md facts, the 100% record)

**Files:**
- Test: `tests/test_docs.py` (create)

Ruling 6A: the audit's eight stale AGENTS.md facts drifted because nothing checked them; these are the guards.

- [ ] **Step 1: Write the tests**

```python
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = (ROOT / "AGENTS.md").read_text()

def test_adr_numbering_is_contiguous():
    nums = sorted(int(p.name[:4]) for p in (ROOT / "docs/adr").glob("0*.md"))
    assert nums == list(range(1, len(nums) + 1)), nums

def test_agents_md_catalogue_count_matches_the_tree():
    from kernelverify.mutation.catalogue import CATALOGUE
    m = re.search(r"fault catalogue \((\d+) entries\)", AGENTS)
    assert m and int(m.group(1)) == len(CATALOGUE), (m and m.group(1), len(CATALOGUE))

def test_every_path_agents_md_layout_names_exists():
    layout = AGENTS.split("## Layout", 1)[1].split("\n## ", 1)[0]
    for path in re.findall(r"`((?:kernelverify|bench|docs|vendor)/[^`]+?)`", layout):
        if path.endswith("/") or path.endswith(".py") or path.endswith(".md") or path.endswith(".mm") or path.endswith(".sh"):
            assert (ROOT / path.rstrip("/")).exists(), f"AGENTS.md Layout names {path}, which does not exist"

def test_agents_md_empty_packages_list_is_true():
    m = re.search(r"Empty packages: `([^`]+)`(?:, `([^`]+)`)*", AGENTS)
    claimed = set(re.findall(r"`([a-z_]+)`", m.group(0))) if m else set()
    actual = {p.name for p in (ROOT / "kernelverify").iterdir()
              if p.is_dir() and not p.name.startswith("__")
              and (p / "__init__.py").exists()                                   # a package, not a data dir
              and all(f.name == "__init__.py" and f.stat().st_size == 0 for f in p.glob("*.py"))}
    assert claimed == actual, (claimed, actual)

def test_the_battery_100_percent_claim_has_a_committed_record():
    rec = ROOT / "bench/results/score_oracles-2026-08-15.txt"
    assert rec.exists()
    assert "B=16: none, every viable fault caught in every run" in rec.read_text()
```

- [ ] **Step 2: Run (after H1 and H2 land, and after the contract merge for the ADR test); commit**

```bash
git add tests/test_docs.py
git commit -m "The record guards itself: ADR numbering, AGENTS.md counts and paths, and the 100% claim's evidence are tested"
```

### Task H4: Certificates re-emitted

Not a code task: this is measurement M3 (GPU, minutes) and is scheduled in Workstream M after V4 and the merges. Listed here so the hygiene lane's checklist is complete.

### Task H6: TODOS entry - a held-out draw with teeth (ruling OV-9)

**Files:**
- Modify: `TODOS.md`

- [ ] **Step 1: Add this entry, verbatim in the file's format**

```markdown
## The held-out draw differs from calibration by seed only

- What: pre-register and run a held-out draw for the device-arithmetic K derivation that varies SHAPE (the six Qwen3-4B serving shapes, not the three calibration shapes) and WEIGHT DISTRIBUTION (a draw the calibration never saw), so G1 on the held-out draw can actually disagree with calibration.
- Why: the 2026-08-16 amendment (ruling 1A) made G1 falsifiable in principle - K is set from calibration seeds 0/1 and tested on seeds 100/101 - but both draws use the same three shapes and the same two weight distributions, so the test is honest in the record and weak in practice; the outside voice of the plan review named it.
- Pros: the serving records already exist as a natural distinct draw, so half the design is done; a real held-out disagreement is the first evidence the K rule could ever produce against itself.
- Cons: a new measurement design needs its own pre-registration and ADR, and a serving-shape draw costs the ~45-minute cold grid rather than the 7-minute device grid.
- Context: `bench/calibrate_quant_device.py` docstring amendment of 2026-08-16 and ADR 0017; `choose_k` takes per-width `(cal, indep)` record lists, so a different held-out source is a change to what `indep` holds, not to the rule.
- Depends on / blocked by: ADR 0017 (the M1 re-run under the amendment) landing first, so the two changes are never mixed in one reading.
```

- [ ] **Step 2: Commit**

```bash
git add TODOS.md
git commit -m "Queue a held-out draw that varies shape and distribution, so G1 can bite"
```

### Task H5: Prune the fleet to the lanes that exist

**Files:** none in the repo; `git worktree` state only.

The infra audit found `/Users/vlad/kv-certs` (branch `lane/certificates`, aff91fe) has no venv and cannot run its own tests, `/Users/vlad/kv-phase0` (61ac463) is fully merged, and two scratch worktrees under `/private/tmp` are prunable.
Removing worktrees discards nothing that is committed, but it is Vlad's call per the standing rule on discarding trees.

- [ ] **Step 1: Report to Vlad the exact list with each worktree's branch, whether it is merged into main (`git branch --merged main`), and whether its tree is dirty; ask before any `git worktree remove`**
- [ ] **Step 2: On his word: `git worktree prune`; `git worktree remove /Users/vlad/kv-phase0` (merged); `kv-certs` only if `lane/certificates` is confirmed superseded by main's `bench/emit_pack_certificates.py`**

---

# Workstream M - Measurements (coordinator or detached runner; one at a time)

- **M2 FIRST** (ruling OV-6), the moment EXPANDED I1 lands - every step of it, because the decision surface's untested branches are exactly the ones a losing kernel triggers: `bench/serve_sub4bit.py --ab` in one quiet window, coordinator-run under the manual discipline (I2 not yet landed), ~30 min, `python -u`, stdout and stderr to `bench/.cache/serve_ab_<date>.log` (durable, gitignored), real exit code echoed. Fills the findings doc's sections 9-10. Its outcomes are pre-registered in Sequencing above: the B = 1 arms pin the batch-1 E2E baseline every spike must beat (this is the plan's only speed number, and it is a baseline, not a claim - this plan makes nothing faster; it decides how, and makes the safety net honest first); B = 5-9 decides only whether `wide_qmv` stays for the multi-stream regime or is retired.
- **M1** after V5 lands: `bench/calibrate_quant_device.py` (7 min). Reading -> ADR 0017 "G1 out of sample": what the held-out draw says per width, whether any INDEPENDENT MISS fires (exit 1), both duplicate readings. Copy records to `bench/results/quant_device_adequacy.json` (overwrites, sha in the ADR).
- **M3** after V4 and the merges: `bench/emit_pack_certificates.py` (minutes, GPU); commit the 26 regenerated certificates and MANIFEST - or fewer, if M2 retired the wide_qmv family.

Not scheduled (deferred, below): the 4-bit pricing run and the R x M cliff sweep - both tune a kernel that loses for the target user; they wait for R2 and for M2's verdict.

# Merge queue (each: `code-review` skill on the merge diff, then Vlad)

1. `contract-b1-ruling` -> main (5 commits, 716 green ON THAT BRANCH - main's count differs; carries the false-positive fix and ADR 0016; heals the split ADR record). TWO known conflict sites (verified with `git merge-tree`): `kernelverify/battery/core.py::_oracle_member_labels` (main derives moe labels; the branch adds `QUANT_ENSEMBLE_VERSION` - keep both) and `AGENTS.md` (both sides amended it; resolve by keeping every line from both, since H1 rewrites the file right after and H3's tests then parse it). Main also moved under `bench/calibrate_quant_device.py`, `bench/calibrate_quant_serving.py` and `bench/emit_pack_certificates.py` since the fork: every line number this plan quotes for those files is a pre-merge LOCATOR - re-locate by symbol after the merge (ruling OV2-pkg).
2. `metal-runner` -> main (`a52e59c`, 739 green).
3. Then `verif/integrity`, `infra/measurement-discipline`, `hygiene/records`, `research/faster-honest-decode` as they finish, smallest first.

# Verification (end-to-end, after everything lands)

- `pytest` full: green in `/Users/vlad/kernelverify/.venv`; `pytest -m "not slow"` under 30 s.
- `bench/calibrate_quant_device.py` prints `device-grid K` and `INDEPENDENT MISS: none` (or names the miss); JSON has `k_grid` and `k_shipped`.
- `python -c "import calibrate_quant_serving as h, kernelverify.schemas.native_ops as n; assert h.K_SHIP is n.K_QUANT"` from `bench/`.
- `bench/serve_sub4bit.py --ab` refuses with exit 4 while another harness holds `/tmp/kernelverify.measurement.lock`.
- `bench/start_binding_run.sh bench/serve_sub4bit.py -- --ab` arms, waits for strong idle, runs, writes `detached_status-serve_sub4bit.json`.
- Every committed certificate's `tolerance_model.ensemble_version == "quant-ensemble-v2"`.
- `docs/adr/` numbered 0001..0017 without gaps; AGENTS.md's counts match `python -c` outputs.
- `docs/research/2026-08-16-faster-honest-decode-kernels-literature.md` has sections 4-7 filled and a pre-registered three-outcome go/no-go for three spikes.
- The findings doc's section 9 carries the B = 1 stock arms' tokens/s from M2: the batch-1 baseline the next plan's spikes are measured against. (This plan asserts no speedup anywhere; that is by design.)

# Deferred (recorded with numbers, not in this plan)

- Kernel spikes and the optimiser: opened by R2 as the next plan.
- Shipped floor = calibrated floor (device members not in the shipped tolerance; measured gap 1.464x on 30/768 device records): contract renegotiation, own ADR.
- Int-domain second member (factored-groups leave-one-out 7.561 vs next 1.692): membership decision, own ADR.
- kv_attention and moe K calibration (V2 labels them; calibrating is a pre-registered harness each).
- 4-bit pricing run; R x M cliff sweep; Track 2 tiled 12+ kernel: on hold pending R2.
- Mac mini: not justified as throughput (queue ~2.5 h, quiet hours ~14/night; a different chip is a new column under SCHEMA.md); revisit only as multi-chip coverage, and `provenance_tier: rental-run` is the cheaper first test of that.

---

# Plan review outputs (2026-08-16, plan-eng-review)

## NOT in scope (considered, deferred, one line each)

- Kernel spikes and the optimiser itself: opened by R2's synthesis as the next plan, under its own pre-registration.
- Shipped floor = calibrated floor (device members not in the shipped tolerance; measured gap 1.464x on 30/768 device records): contract renegotiation, own ADR.
- Int-domain second member (factored-groups leave-one-out 7.561 vs next 1.692): membership decision, own ADR.
- kv_attention and moe K calibration: V2 labels the borrow honestly now; calibrating is a pre-registered harness each.
- A held-out draw that varies shape and distribution (ruling OV-9): H6 queues it; runs after ADR 0017.
- 4-bit pricing run, R x M cliff sweep, Track 2 tiled 12+ kernel: on hold pending R2 and M2's verdict.
- Mac mini: not justified as throughput (queue ~2.5 h vs ~14 quiet hours a night; a different chip is a new SCHEMA.md column); revisit only as multi-chip coverage.
- Prose-substring certificate tests (test_certificate.py): weak but not wrong; not worth a task this block.
- Worktree pruning (H5): Vlad's call per the discard rule; listed, not executed.

## What already exists (reused, not rebuilt)

| Sub-problem | Existing code | Plan reuses? |
|---|---|---|
| Machine lock, budget, memory gate, exit vocabulary | `bench/machine_state.MeasurementLock`, `bench/memory_guard.py` | Yes - I1 copies `price_qmv_boundary.py`'s exact ordering; I2 adds two constants to the same table |
| Detached, session-surviving execution | `bench/detached_run.py` + `start_binding_run.sh` (launchd, caffeinate, strong-idle wait, retry) | Yes - I2 generalises the harness and outcome protocol; the wait and retry loop are untouched |
| Correctness gate, pricing protocol, certificate emitter | `bench/pack_wide_qmv.py`, `bench/price_qmv_boundary.py`, `bench/emit_pack_certificates.py` | Yes - R2's spikes and section 7 sit on them; nothing new is built for measurement |
| Ensemble versioning | `QUANT_ENSEMBLE_VERSION` (contract branch), `MOE_MEMBERS` (main) | Yes - V3 extends the pattern to kv/moe instead of a new mechanism |
| K-grid cover | inline `next(k for k in K_GRID ...)` at device:433 | Extracted once as `cover()` (V5), not duplicated |
| Recompute-from-evidence tests | `tests/test_quant_contract_members.py::_recompute` | Yes - the slow tests are marked, not rewritten |
| End-to-end A/B | `bench/serve_sub4bit.py --ab` (four arms, pre-registered, never run) | Yes - M2 runs it; nothing new is written |

## Failure modes (one per new codepath; test / handling / visibility)

| Codepath | Realistic failure | Test | Handled | Visible |
|---|---|---|---|---|
| V5 choose_k across widths | one width's held-out demand exceeds K | yes | exit 1, cell named | yes |
| V5 distinct dedupe | two members differ only past 17 rounded digits | partial | treated as duplicates | printed both readings |
| I1 lock in serve_sub4bit | foreground harness holds the lock at 3am | yes | EXIT_LOCK_HELD -> runner waits | status file |
| I1 budget in ab() | footprint crosses 24 GB mid-arm | yes | EXIT_BUDGET_REFUSAL, no write | yes |
| I2 exit-code protocol | argparse typo | yes | crashed in one attempt (OV-1) | status file |
| I2 measure_baselines lock | co-fire with a locked harness | yes | EXIT_LOCK_HELD | yes |
| I3 mmap footprint test | machine under memory pressure at test time | n/a | assertion fails loudly | CI red |
| I5 child getppid | parent dies between records | yes | child exits EXIT_CHILD_DEATH | log |
| I5 parent SIGTERM to group | child ignores SIGTERM | **no test** | best-effort | **silent** - noted; the getppid check is the backstop, so not critical |
| H3 docs tests | AGENTS.md edited without updating a count | yes | test fails | CI red |
| M2 A/B | machine goes non-idle mid-run | existing gate | check_idle_after returns 1 | yes |

No critical gaps: the one silent path (parent SIGTERM) has the child's own getppid check as a second line.

## Worktree parallelization

| Step | Modules touched | Depends on |
|---|---|---|
| merges (contract, runner) | main | - |
| I1 | bench/serve_sub4bit.py, bench/memory_guard.py, tests/ | runner merge |
| M2 | (measurement) | I1 |
| R1-R2 | docs/research/, TODOS.md | - |
| V1(device), V2, V3, V4, V5, V6 | bench/calibrate_quant_device.py, bench/calibrate_quant_bits.py, kernelverify/schemas/, kernelverify/battery/, bench/emit_pack_certificates.py, tests/ | contract merge |
| V1(serving), I4, I5 | bench/calibrate_quant_serving.py, tests/ | infra branch, in that order (OV-5) |
| I2, I3 | bench/detached_run.py, start_binding_run.sh, measure_baselines.py, spike_mlx_e2e.py, pyproject.toml, tests/ | I1 (memory_guard constants) |
| H1, H2, H3, H6 | AGENTS.md, TODOS.md, docs/adr/, bench/results/, tests/test_docs.py | both merges; H3 after H1/H2 |
| M1 | (measurement) | V5 |
| M3 | bench/.certificates/ | V4, merges, M2's verdict |

Lanes: `Lane A (infra): V1(serving line + test) -> I1 -> I2 -> I3 -> I4 -> I5 (sequential, shared bench/ harness files)`; `Lane M (measurement, GPU, one at a time): M2 the moment expanded I1 lands in full - in PARALLEL with I2..I5, which depend only on I1's constants, not on M2's result; then M1 after V5; then M3`; `Lane B: R1 -> R2 (independent)`; `Lane C: V1(device) -> V2 -> V3 -> V4 -> V5 -> V6 (sequential, shared schemas/ and device harness)`; `Lane D: H1 -> H2 -> H3 -> H6 (sequential, docs)`.
Execution: after the two merges, launch A, B, C, D in parallel worktrees; M2 starts as soon as A's I1 is on main. Critical path: the two merges (both need Vlad) -> I1 -> M2; everything but Lane B waits on the merges. **Conflict flags:** Lane C and Lane D both touch `TODOS.md` (V2, H6): trivial, append-only; merge D last. Only Lane A touches `bench/memory_guard.py`.

## Implementation Tasks
Synthesized from this review's findings. Each task derives from a specific finding above. Run with Claude Code; checkbox as you ship.

- [ ] **T1 (P1, human: ~half day / CC: ~20 min)** - calibrate_quant_device - K from calibration only, held-out miss stops the run, `cover()`, duplicates collapsed, `main(argv)`, one K across widths
  - Surfaced by: Architecture issue 1 (1A) + outside voice OV-A (ii)
  - Files: bench/calibrate_quant_device.py, bench/calibrate_quant_bits.py, tests
  - Verify: the five V5 tests + the regression key test; then M1
- [ ] **T2 (P1, human: ~half day / CC: ~20 min)** - detached runner - harnesses own the lock; runner never locks; measure_baselines and spike_mlx_e2e get MeasurementLock; `--print-plist` + test
  - Surfaced by: Architecture issue 2 (2A) + test gap 5A
  - Files: bench/detached_run.py, bench/start_binding_run.sh, bench/measure_baselines.py, bench/spike_mlx_e2e.py, tests/test_detached_run.py
  - Verify: tests/test_detached_run.py; a dry `start_binding_run.sh --print-plist`
- [ ] **T3 (P1, human: ~1 h / CC: ~10 min)** - serve_sub4bit + memory_guard - EXIT_PRECONDITION and EXIT_NOT_IDLE; runner maps 9/4/5 wait, 8/3 give up, 2 crashed; tests parametrized over --ab/--mde
  - Surfaced by: Code-quality issue 3 (3A), outside voice OV-1, test gap 7A
  - Files: bench/serve_sub4bit.py, bench/memory_guard.py, tests/test_serve_sub4bit.py, docs/research findings doc T8 line
  - Verify: the parametrized I1 tests; judge_exit_code tests
- [ ] **T4 (P2, human: ~1 h / CC: ~10 min)** - tests/test_docs.py - ADR contiguity, AGENTS.md counts/paths/empty-packages, the 100% record
  - Surfaced by: test gap 6A
  - Files: tests/test_docs.py, AGENTS.md (H1 formats counts parseably)
  - Verify: pytest tests/test_docs.py after H1/H2
- [ ] **T5 (P2, human: ~30 min / CC: ~5 min)** - V6 - kv cache dequant checked against MLX quantize+dequantize on the raw cache
  - Surfaced by: outside voice OV-A (iii)
  - Files: tests/test_native_ops.py
  - Verify: the parametrized bits test
- [ ] **T6 (P2, human: ~1 h / CC: ~10 min)** - I1 first, M2 immediately after, outcome branch pre-registered
  - Surfaced by: outside voice OV-6
  - Files: plan Sequencing; findings doc sections 9-10 after the run
  - Verify: the run's own gates
- [ ] **T7 (P3, human: ~10 min / CC: ~2 min)** - R1 rubric axis = bytes/weight at M=1; R compressed to two tasks
  - Surfaced by: outside voice OV-7, OV-8
  - Files: docs/research/2026-08-16-... (R1 skeleton)
  - Verify: section 3 of the doc reads as amended
- [ ] **T8 (P3, human: ~5 min / CC: ~2 min)** - TODOS: held-out draw with teeth
  - Surfaced by: outside voice OV-9
  - Files: TODOS.md
  - Verify: entry present in the file's format
- [ ] **T9 (P3, human: ~1 min / CC: ~1 min)** - V1 test keeps the identity assertion; hedge deleted
  - Surfaced by: Code-quality issue 4 (4A)
  - Files: plan text (done)
  - Verify: `assert h.K_SHIP is K_QUANT` fails before, passes after
_No new tasks from Performance review._

# CEO review outputs (2026-08-16, plan-ceo-review, HOLD SCOPE)

## Dream state delta

```
CURRENT                                  AFTER THIS PLAN                             12-MONTH IDEAL
verifier strong but pointed inward;      verifier's headline check falsifiable;      generator proposes Metal kernels for a
one hand-written kernel that loses for   ensembles versioned; certificates true;     MacBook user's model; every candidate gated;
the target user; generator absent;       E2E batch-1 baseline measured; research     certified sub-4-bit kernels faster than stock
one laptop, coordinator babysits runs;   has picked the technique class and named    4-bit at batch 1, shipped into MLX; the
AGENTS.md drifts weekly                  the generator's shape; runs go overnight;   verifier catches its first externally-authored
                                         AGENTS.md guarded by tests                  broken kernel
```
Delta: the plan clears the ground and picks the direction; the generator and the first honest kernel are the next plan, opened by R2. Nothing here moves away from the ideal; the risk is only that R2 finds the batch-1 headroom smaller than E3 suggests, which is a finding, not a failure.

## Error and rescue registry

```
CODEPATH                              | WHAT CAN GO WRONG                     | EXCEPTION / SIGNAL          | RESCUED | ACTION                                  | USER SEES
serve_sub4bit.main (I1)               | pins mismatch                         | RuntimeError from verify_pins| Y      | return EXIT_PRECONDITION (8)            | "REFUSED: <reason>", runner gives up
                                      | machine not idle                      | NotIdle                     | Y       | return EXIT_NOT_IDLE (9)                | runner waits
                                      | pinned zone disagrees with table      | PreconditionFailed          | Y       | return EXIT_PRECONDITION (8)            | runner gives up
                                      | lock held                             | acquire() -> False          | Y       | return EXIT_LOCK_HELD (4)               | runner waits
                                      | footprint over budget mid-arm         | BudgetExceeded              | Y       | return EXIT_BUDGET_REFUSAL (3), no write | runner gives up
                                      | machine memory too low for a cell     | LowMemoryRefusal            | Y       | return EXIT_LOW_MEMORY (5)              | runner waits
                                      | model directory missing               | uncaught (FileNotFoundError)| N       | exit 1 + traceback                      | runner: "crashed" + stderr_tail (CEO 1A)
detached_run.main (I2)                | harness argv typo                     | argparse exit 2             | Y       | "crashed" in one attempt (OV-1)         | status file
                                      | strong-idle deadline (12 h)           | wait_for_strong_idle False  | Y       | "gave-up", return 3                     | status file
                                      | harness deliberate return 1           | exit 1, no traceback        | Y       | "stopped"                               | status file
calibrate_quant_device.main (V5)      | held-out demand > K                   | INDEPENDENT MISS            | Y       | print cell, return 1                    | stdout + JSON
                                      | every width fails G0                  | usable == []                | Y       | "no usable width", return 1 (CEO 3A)    | stdout
                                      | demand above the K grid               | cover() ValueError          | N (by design) | KILL branch, traceback            | runner: crashed
cover() (V5)                          | demand > max(K_GRID)                  | ValueError                  | N (by design) | the KILL branch, never clamped     | as above
_derive routed_windows (already main) | recording bytes changed               | ValueError at import        | N (by design) | import fails loudly                | traceback
tests/test_docs.py (H3)               | AGENTS.md count drifts                | AssertionError              | n/a     | CI red                                  | test output
```
No catch-all handlers are introduced anywhere; the two unrescued paths are deliberate KILL/refuse-at-import semantics the repo already uses.

## Failure modes registry

```
CODEPATH                         | FAILURE MODE                          | RESCUED? | TEST? | USER SEES?            | LOGGED?
choose_k across widths           | held-out demand > K on one width      | Y        | Y     | INDEPENDENT MISS line | JSON + stdout
choose_k                         | no usable width                       | Y        | Y     | named verdict         | stdout
distinct dedupe                  | members equal only past 17 digits     | Y        | part  | both readings printed | JSON
serve_sub4bit lock               | foreground harness holds the lock     | Y        | Y     | runner waits          | status file
serve_sub4bit budget             | footprint crosses 24 GB               | Y        | Y     | refusal, no write     | stdout
detached runner exit map         | argparse typo                         | Y        | Y     | crashed in 1 attempt  | status + stderr_tail
detached runner                  | crash exits 1                         | Y        | Y     | crashed (traceback)   | stderr_tail
start_binding_run.sh             | & in an argument                      | Y        | Y     | valid plist           | -
measure_baselines lock           | co-fire with a locked harness         | Y        | Y     | EXIT_LOCK_HELD        | stdout
mmap footprint test              | machine under memory pressure         | n/a      | n/a   | CI red                | -
child getppid                    | parent dies between records           | Y        | Y     | child exits           | log
parent SIGTERM to group          | child ignores SIGTERM                 | best-effort | N   | silent (backstop: getppid) | -
docs tests                       | AGENTS.md edited without counts       | Y        | Y     | CI red                | -
M2 A/B                           | machine non-idle mid-run              | Y (existing) | existing | check_idle_after 1 | log
```
CRITICAL GAPS: 0 (the one silent path has the getppid backstop and is noted).

## Diagrams

System architecture (new pieces in [brackets]):
```
 launchd one-shot ──► detached_run.py [--harness/--protocol/status-per-stem, no lock] ──► harness process
                                                                                            │  MeasurementLock (owner)
        harnesses: calibrate_quant_serving | price_qmv_boundary | serve_sub4bit [I1] | measure_baselines [I2] | spike_mlx_e2e [I2]
                                   └── memory_guard: BudgetGuard, require_available_memory, EXIT_* [+8 PRECONDITION, +9 NOT_IDLE]
 calibrate_quant_device [V5: choose_k(per_bits) -> k_grid, misses; G1 on indep] ──► JSON {k_grid, k_shipped, *_distinct, independent_miss}
 native_ops.K_QUANT (ships) ◄── calibrate_quant_serving.K_SHIP [V1: import, not literal]
 quant_contract.QUANT_ENSEMBLE_VERSION + [V3: KV/MOE_ENSEMBLE_VERSION] ──► battery/core fingerprint ──► verdict cache
 emit_pack_certificates [V4: ensemble_version, ensemble_source_sha256, K_native for moe] ──► bench/.certificates (M3 re-emits)
 docs/research/…literature.md [R1, R2] ──► next plan (spikes, membership blocks for NEW classes, the generator)
 tests/test_docs.py [H3] ──► AGENTS.md, docs/adr numbering, bench/results record
```

Detached runner state machine (I2):
```
 [armed] ──strong idle (5x30s)──► [measuring] ──exit 0──► [done]
    │ deadline 12h                     │ exit 9/4/5 (transient) ──► [waiting] ──► back to armed, attempt NOT consumed
    ▼                                  │ exit 3/8 (permanent)   ──► [gave-up]  (return 3)
 [gave-up]                             │ exit 1, no traceback   ──► [stopped]  (return 0? no: 1 = measured-and-stopped, reported as such)
                                       │ exit 1 + Traceback, or any other code (incl. argparse 2) ──► [crashed] (return 4, stderr_tail kept)
 impossible: [waiting] consuming an attempt (guarded by judge_exit_code's second tuple element)
```

Deployment sequence and rollback:
```
 merge contract ──► merge runner ──► Lane A first commit (V1 serving) ──► I1 ──► M2 (GPU) ‖ I2..I5 ──► V/H lanes ──► M1 ──► M3
 rollback: each lane is short-lived and merges through the gate; `git revert -m 1 <merge sha>` undoes a lane; ADR/TODOS entries stay as record
```

## Stale diagram audit

Files this plan touches carry no ASCII diagrams today except `bench/calibrate_quant_device.py`'s docstring, which V5 ADDS one (before/after data flow) as part of the amendment - so no existing diagram can go stale; the new one is owned by V5.

## Implementation tasks added by the CEO pass

- [ ] **T10 (P1, human: ~1 h / CC: ~10 min)** - detached_run - stderr tail in the status file; exit 1 + Traceback = crashed
  - Surfaced by: CEO Section 2 (1A)
  - Files: bench/detached_run.py, tests/test_detached_run.py
  - Verify: test_a_crashing_harness_leaves_its_traceback_in_the_status_file
- [ ] **T11 (P2, human: ~20 min / CC: ~3 min)** - start_binding_run.sh - XML-escape plist values
  - Surfaced by: CEO Section 3 (2A)
  - Files: bench/start_binding_run.sh, tests/test_detached_run.py
  - Verify: the `a&b<c>` case in the --print-plist test
- [ ] **T12 (P2, human: ~10 min / CC: ~2 min)** - calibrate_quant_device - `choose_k({})` named verdict
  - Surfaced by: CEO Section 4 (3A)
  - Files: bench/calibrate_quant_device.py, tests/test_quant_device_members.py
  - Verify: test_choose_k_with_no_usable_width_is_a_named_verdict_not_a_crash
- [ ] **T13 (P1, human: ~half day / CC: ~30 min)** - research - three-outcome go/no-go (GO / NO-GO / NEW CLASS -> membership block first)
  - Surfaced by: CEO outside voice OV2-6
  - Files: docs/research/2026-08-16-…literature.md section 5
  - Verify: section 5 names the outcome each technique is eligible for before any spike
- [ ] **T14 (P1, human: ~1 h / CC: ~15 min)** - plan bookkeeping - both merge conflict sites named; V1 split with the test on the infra branch; lanes fixed; M2's B=1 baseline; H3 test tightened
  - Surfaced by: CEO outside voice OV2-pkg
  - Files: plan text (done); Merge queue section
  - Verify: `git merge-tree main contract-b1-ruling` names exactly AGENTS.md and battery/core.py
_No new tasks from Sections 5-11._

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | clean | mode: HOLD_SCOPE, 0 critical gaps; approach A confirmed; 5 findings folded (exit-1 disambiguation, plist escaping, empty-width verdict, NEW-CLASS go/no-go, merge/lane bookkeeping) |
| Codex Review | `/codex review` | Independent 2nd opinion | 2 | issues_found (claude subagent; codex not installed) | pass 1: 9 findings, all ruled (OV-A, OV-1, OV-5..OV-9); pass 2: 10 findings, 8 folded, 2 accepted as notes (critical path is the two human merge gates; the plan asserts no speedup by design) |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | clean | 16 issues, 0 critical gaps; plus a superpowers fresh-context reviewer pass (3 impossible tests fixed, exit table completed, V3 merge site flagged, 0 placeholders) |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | - | - |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | - | - |

- **CROSS-MODEL:** all three passes agreed on the reframe (dense 4-bit GEMV is at the roofline; headroom is sub-4-bit bytes/weight and non-kernel time), on falsifiable G1, on harnesses owning the lock, and on testing the premise first; the only sustained disagreement (shrink research, spike now) was rejected in favour of Vlad's research-first ruling and the plan compressed instead.
- **VERDICT:** CEO + ENG CLEARED - ready to implement.

NO UNRESOLVED DECISIONS
