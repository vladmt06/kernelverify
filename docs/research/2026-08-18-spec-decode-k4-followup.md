# End-to-end speculative decode at K = 4: the pre-registered follow-up

Status: pre-registered 2026-08-18, before any measurement.
Sections 1 through 6 fix the primary cell and the outcomes.
Sections 7 and 8 remain placeholders until the binding run has completed.

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

This section is filled after the registered run.

## 8. Verdicts

This section is filled after the registered run.
