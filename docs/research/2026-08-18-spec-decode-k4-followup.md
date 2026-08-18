# End-to-end speculative decode at K = 4: the pre-registered follow-up

Status: pre-registered 2026-08-18; measured and read 2026-08-18.
Sections 1 through 6 fix the primary cell and the outcomes before any measurement exists.
Sections 7 and 8 carry the binding run and the verdict read off it, against the outcomes fixed above.

## 1. Why this exists, and what it is not allowed to do

The parent measurement in `docs/research/2026-08-18-spec-decode-e2e.md` read NO-GO at its pre-registered primary cell K = 6 on both drafts, and its section 10 recorded, labelled exploratory and selection-biased, that the composed number was positive at K = 2 and K = 4 with the 0.6B draft.
It said the two cells are not the same kind of cell: at K = 2 nothing routes and the whole gain is speculation, while at K = 4 the kernel contributed a positive share of a positive composed number.
It named K = 4 with the 0.6B draft as the only cell in that grid where speculation pays AND the kernel helps, and said the honest way to find out whether that is real is to pre-register it and measure it again rather than read it off the grid.

This document is that pre-registration.
It was written after the parent grid was seen, and nothing here removes that: the cell was chosen by looking.
What changes is that the K = 4 number this document quotes will come from a run the choice did not see, which is the honest form available once the grid exists.
The parent's K = 4 numbers are recorded in section 6 for the replication check and for no other purpose, and this document may not quote them as a result.

## 2. What is reused unchanged

The harness is `bench/serve_spec_decode.py` and the rules are `bench/spec_decode_rules.py`, exactly as merged in `528aeae`.
The five arms, the interleaving, `K_GRID = (2, 4, 6, 8, 10)`, `ROUNDS = 5`, `GEN_TOKENS = 128`, `PROMPT_T = 64`, greedy sampling, both drafts, the 24.0 GB budget, the exact per-site dispatch counts, the two-way token identity, the per-cell arm 2 probe, and the per-outcome noise floors are all the parent's sections 2 through 5 and their amendments, unchanged.
The full grid runs again rather than the one cell, because the harness runs the grid and because every other cell then becomes a second binding sample of the parent at no cost, which section 6 uses.

## 3. The one change

The parent hard-coded the primary cell as `PRIMARY_K = 6`.
This follow-up gives `decide_o3` and `decide_o4` a `primary_k` keyword that defaults to that constant, and gives the harness a `--primary-k` flag that defaults to 6 and passes through to both.
The flag is a registered input to the run, named in this document before the code exists, and it is the whole of the change.
No arm, no threshold, no rule and no other parameter moves, and no verdict string and no floor comparison enters the harness.

Registered invocation: `bench/start_binding_run.sh bench/serve_spec_decode.py -- --primary-k 4`.

## 4. The primary cell

The primary cell is the 0.6B draft, `qwen3-0.6b-4bit-g64`, at K = 4, so M = 5.
The 1.7B draft's K = 4 cell is reported and is secondary; it is not the claim.
Every other cell is reported for the replication check in section 6 and is quoted for nothing else.

## 5. Pre-registered outcomes

### O3', the product number at K = 4

Arm 1 against arm 0 at the primary cell, with the parent's O3 arithmetic: `D(1, 0)`, floor `max(S_1, S_0)`, `speculation_pct = D(2, 0)`, `kernel_pct = D(1, 2)`, and `composed_attribution` from `bench/attribution.py`.

**GO** requires all three of the following.
`D(1, 0) > F_O3'`, so the composed product beats plain decode beyond the floor.
`D(1, 2) > F_O2` at the same cell, so the kernel's own contribution is a WIN by the parent's O2 rule read on this run's data.
The O2 cell is a decider on this run's data, `ceiling_pct > F_O2`, so the second condition was capable of failing.

Anything else is **NO-GO**, and a NO-GO here closes the K = 4 question on this stack; there is no third cell to move to.

The attribution label is reported and does NOT decide GO, and the reason is registered so it cannot become an argument after the fact.
`composed_attribution` reads `artifact-alone` whenever the base comparison, speculation alone against plain decode, clears the floor by itself, whatever the kernel adds on top; the parent's exploratory K = 4 numbers already read that way, with speculation alone at +5.96% against floors under 0.4%.
That label means "the artifact already had a win before the kernel", which is true and is not the question.
The question is whether the kernel adds a real increment to a real win, and that is exactly the second GO condition, `D(1, 2) > F_O2`, which the label does not measure.

### O4', the memory-capacity claim at K = 4

Arm 1 against arm 3 at the primary cell, with the parent's O4 arithmetic and vocabulary.
Reported, not decisive.

### The kernel question at K = 4, restated

O2 at K = 4 on this run's data is read exactly as the parent reads it, WIN, LOSS, NULL, or `not-a-decider`, and section 7 reports it.
The parent's O2 at K = 4 was WIN at +4.02% against a 0.14% floor with a 4.17% ceiling; if this run's O2 at K = 4 is not a WIN, then GO is impossible by construction and that fact is a finding about run-to-run stability, reported before any O3' reading.

## 6. The replication check, and what the parent's numbers may be used for

Because the whole grid runs again, this run is also a second binding sample of every parent cell.
The parent's binding values at the primary cells are recorded here so that agreement or disagreement is a registered reading rather than an observation.

| quantity | parent, binding run of 2026-08-18 |
|---|---|
| O2 at 0.6B K = 6, `D(1, 2)` | +12.56%, floor 2.96%, ceiling 11.51% |
| O3 at 0.6B K = 6, `D(1, 0)` | -6.11%, floor 2.96% |
| O2 at 0.6B K = 4, `D(1, 2)` | +4.02%, floor 0.14%, ceiling 4.17% |
| exploratory at 0.6B K = 4, `D(1, 0)` | +10.22%, arms 66.78 / 73.60 / 70.76 tokens per second for 0 / 1 / 2 |
| exploratory at 0.6B K = 4, `D(2, 0)` | +5.96% |

The replication check: at 0.6B K = 6, this run's `D(1, 2)` and `D(1, 0)` must each land within the larger of the two runs' floors of the parent's value.
If either does not, the two binding runs disagree at the parent's own primary cell, and that is reported as a finding BEFORE any K = 4 reading and section 7 says which reading it casts doubt on.

The parent's exploratory K = 4 values in the last two rows are here so a reader can see whether this run's K = 4 landed near them.
Landing near them is not the criterion for GO and is not a result; the criterion is section 5, read on this run's data alone.

## 7. Measurements

Measured by `bench/serve_spec_decode.py --primary-k 4` through the detached runner on 2026-08-18, 17:19:27 to 17:33:53 UTC, 14 minutes, first attempt of three, harness exit 0.
Five consecutive clean idle samples were held before the start and the closing sample was clean, so the run passed `check_idle_after` and is binding.
MLX 0.32.0, mlx-lm 0.31.3, Apple M3 Pro, Mac15,7, Darwin 26.5.2 build 25F84, the same four pinned models, budget 24.0 GB with no flag.
Every `RESULT:` line for O3 and O4 carries `primary_k: 4`, which is the registered flag reaching the verdicts.

### Interception accounting and divergence

Every cell's arm 1 routed exactly what its observed passes predicted and arm 4 routed nothing, cell for cell identical to the parent run: 54180 and 51660 at K = 4, 46620 and 42840 at K = 6, 44100 and 40320 at K = 8, zero at K = 2, and 1260 at the 1.7B K = 10 cell where one short in-window pass per round routes.
No round in the run recorded a hard fallback, no `kernel-diverged` label and no `mlx-m-dependent` label fired anywhere, and arm 4 matched arm 2 to the token in every cell.
All five rounds were eligible for every outcome in all ten cells.
The divergence report is therefore "none", and it is present because section 4 of the parent requires it to be present even when it is.

### The replication check, read first as section 6 requires

| quantity, 0.6B K = 6 | parent | this run | difference | band | inside |
|---|---|---|---|---|---|
| O2, `D(1, 2)` | +12.56% | +12.65% | 0.09 pts | 2.96 pts | yes |
| O3, `D(1, 0)` | -6.11% | -5.62% | 0.49 pts | 4.18 pts | yes |

Both land well inside the larger of the two runs' floors, so the two binding runs agree at the parent's own primary cell and nothing casts doubt on the K = 4 reading below.

The agreement is much stronger than the check demanded, and what follows describes it rather than testing it: section 6 registered one band, the floor band applied above, and no tighter bound was registered before this run.
The six O2 point estimates moved by at most 0.09 percentage points between the two runs: at K = 4, +4.02% against +3.98% on the 0.6B draft and +2.59% against +2.61% on the 1.7B; at K = 6, +12.56% against +12.65% and +9.20% against +9.18%; at K = 8, +10.59% against +10.64% and +7.58% against +7.60%.
The primary cell's own composed number moved 0.05 points, +10.22% against +10.17%, with speculation at +5.96% against +5.95% and the kernel term at +4.02% against +3.98%.

### The primary cell

| quantity, 0.6B K = 4 | value |
|---|---|
| arm 0, plain stock | 66.77 tokens per second |
| arm 1, ours | 73.56 |
| arm 2, stock 3-bit speculative | 70.74 |
| arm 3, stock 4-bit speculative | 76.46 |
| composed, `D(1, 0)` | +10.17% |
| speculation alone, `D(2, 0)` | +5.95% |
| kernel on top, `D(1, 2)` | +3.98% |
| `F_O3'` and `F_O2` | 5.32% |
| ceiling at this cell | 4.16% |
| attribution, reported only | `artifact-alone` |

### O2 at the primary cell, reported before any O3' reading

Section 5 requires this cell's O2 verdict here, and requires a non-WIN to be reported as a finding about run-to-run stability before O3' is read at all.

O2 at 0.6B K = 4 on this run's data reads **`not-a-decider`**.
`D(1, 2)` is +3.98% against a 5.32% floor, and the cell's 4.16% ceiling does not clear that floor either, so `decider` is false and the WIN/LOSS/NULL branch is never reached.
The parent read the same cell `WIN`, +4.02% against a 0.14% floor with a 4.17% ceiling.

By section 5 this makes GO impossible by construction, and it is a finding about run-to-run stability rather than about the kernel: the point estimate moved 0.05 percentage points between the runs while the floor moved from 0.14% to 5.32%, which is a fact about how this run sampled the cell and not about what the kernel did in it.
The other five routed O2 cells read `WIN` again on this run.

### Why the floor is 5.32% and the parent's was 0.14%

The point estimates replicated and the floor did not, so the floor is where this run differs, and the cause is one sample.
Arm 1's five rounds at this cell were 69.71, 73.44, 73.62, 73.56 and 73.62 tokens per second.
Rounds two through five span 0.24%; round one sits 5.3% below their median, and `spread_pct` is `(max - min) / median`, which has no resistance to a single outlier at all.
The median barely moved, 73.56 against the parent's 73.60, which is why every point estimate above replicated while the floor rose thirty-eightfold.

That cold first round is sporadic rather than systematic, and it is present in both runs.
Comparing each arm-cell's first sample against the median of its remaining four, the parent had four of fifty arm-cells more than 1% cold and this run had three of fifty, and they land on different cells each time: the parent's worst were the 1.7B K = 10 arm 3 at -2.99% and the 0.6B K = 6 arm 1 at -2.60%, while this run's worst were the 0.6B K = 4 arm 1 at -5.27% and the 0.6B K = 6 arm 4 at -3.73%.
Nothing about arm 1, about K = 4, or about the probe that runs before the rounds predicts where it lands.

The consequence is structural and is the honest reading of this run.
A cold first round of two to five percent inflates whichever cell's floor it lands in to its own size.
The K = 6 cell's ceiling is around 11%, so it survives one and stays a decider, which is exactly what happened in the parent where the cold round landed there.
The K = 4 cell's ceiling is 4.16%, which is the same order as the artefact itself, so a single cold round is enough to make that cell unreadable.
The K = 4 cell is therefore under-powered against a failure mode that occurs in roughly one arm-cell in fifteen, and this run is the case where it occurred there.

### The secondary cells

The 1.7B draft's K = 4 cell, which is secondary by section 4, read O2 `WIN` at +2.61% against a 0.13% floor with a 3.21% ceiling and `decider` true, and its O3' composed number was -20.55%.
The whole-grid readings were O1 `WIN` at K = 2 for the 0.6B draft and `INCONCLUSIVE` for the 1.7B, and O2 `WIN` at K = 6 and K = 8 on both drafts with the K = 2 and K = 10 null cells reading `NULL` and, at 1.7B K = 10, `NULL-uncontrolled` again.
None of these is the claim and none is quoted as one.

## 8. Verdicts

### O3', the product number at the registered primary cell

**NO-GO.**

| registered GO condition, section 5 | result |
|---|---|
| `D(1, 0) > F_O3'`: composed beats plain decode beyond the floor | met, +10.17% against 5.32% |
| `D(1, 2) > F_O2`: the kernel's own term is a WIN by the O2 rule | NOT met, +3.98% against 5.32%, so `not-a-decider` rather than `WIN` |
| that O2 cell is a decider, so the second condition could have failed | NOT met, ceiling 4.16% against floor 5.32% |

Two of the three registered conditions failed, so the outcome is NO-GO, and by section 5 that closes the K = 4 question on this stack.

The reason for the NO-GO is recorded exactly, because it is not the reason a reader would assume.
The composed effect did not fail to appear; it reproduced to within 0.05 percentage points of the parent's exploratory value, and so did the kernel's contribution.
What failed is that this run could not tell the kernel's 3.98% from its own noise, because one cold first round put the floor at 5.32% and the cell's ceiling is 4.16%.
A cell whose registered ceiling is smaller than a single cold round can produce is a cell this design cannot reliably read, and that is what the third condition was written to detect.

This is a NO-GO on the design as registered, and it is not a licence to re-read this data with a different floor, a dropped round, or a different criterion.
The number that would come out of any of those is the number the choice was made to produce, which is what sections 1 and 6 exist to prevent.

### O4' at the registered primary cell

Reported and not decisive, as registered: arm 1 at 73.56 against stock 4-bit speculative at 76.46, ratio 0.9620, composed -3.80% and base -7.48% against a 5.32% floor, attribution `inconclusive`.
On the 1.7B draft the same cell read ratio 1.0603 with attribution `artifact-alone`.

### What this run establishes

It establishes that the kernel's O2 point estimates are reproducible to a degree the parent could not show alone: all six moved by at most 0.09 percentage points across two independent binding runs, at every routed width and on both drafts.
The verdicts did not all travel with them, and that gap is what this document is about: five of the six read `WIN` again, while the primary cell read `not-a-decider` on an estimate that had itself moved only 0.05 points.
It establishes that on this run the composed product at K = 4 with the 0.6B draft is +10.17% against plain decode, clearing its floor, and that speculation alone accounts for +5.95 of that.
Section 1 forbids this document from quoting the parent's exploratory K = 4 values as a result, so the claim is about this run's number and the earlier one is not part of it.
It does not establish that the kernel's contribution at that cell is distinguishable from noise, because on this run it was not.

It leaves one thing that is a defect in this experiment rather than a fact about the kernel: `spread_pct` is `(max - min) / median` over five rounds, which a single cold sample dominates, and the warm-up before the rounds does not reliably prevent one.
Fixing that is a different experiment with its own pre-registration, not a third reading of this cell, and it is queued in `TODOS.md` as such.
