# ADR 0021: a check that does not fit is scoped in writing, and a gate reads a quantity rather than a pressure

Date: 2026-08-22
Status: Accepted

## Context

The exploratory run was armed on 2026-08-21 and died twice with a Metal out-of-memory, four minutes into each of two granted windows.
Reading the failure took three attempts and the first two were wrong: the machine being busy, then the number of timing marks.
Both were plausible and neither survived measurement, which is the reason this entry exists rather than a comment in the code.

Three measurement methods changed in the repair, and this repository's rule is that every method change carries the measurement that forced it and an ADR entry recording it.

## Decision 1: a check that cannot run at a width is scoped in writing, and its absence is typed

The structural pass installs timing marks and verifies two things, that every seam fired as often as the arrangement demands and that the marked pass computes what an unmarked pass computes.
It takes no timing. It ran at both registered widths.

Measured 2026-08-22 at the pinned 4B, batch 4, each figure in its own fresh process because one process measuring several widths carries the allocator's cache from one into the next:

| Width | Plain backward | Marked structural pass |
|---|---|---|
| short, (4, 65) | 3.730 GiB | 9.090 GiB |
| long, (4, 1057) | 24.364 GiB | refused by the device at 38.000 GiB |

38.000 GiB is 40.802 GB against a machine of 38.655 GB, so the long-width pass asks for more than the machine holds and no quieter window makes it run.

The cost is not the number of marks, and the evidence for that is narrower than it first looked.
Marking one region costs what marking four costs, so splitting the installation across passes saves nothing.
But a strictly positive per-mark model reproduces all five readings exactly, so the table cannot say a mark is free, and Amendment 18 clause 75 withdraws that claim while keeping the decision it supported.

Amendment 17 clause 67 scopes the pass to the short width and clause 68 requires the uncovered width to record a TYPED ABSENCE.
The rule worth keeping is the second half: an absent check and a passed check must never read alike, so a width that was deliberately not checked says so in the record, and a width whose entry went missing is INCOMPLETE.
`taken` is the discriminator of a two-member union and is validated as one, because testing it for truthiness accepted three records the clauses forbid, including a `taken` of `"not_run"` that says the pass did not run and reads as true.

## Decision 2: a refusal gate reads a quantity of memory, not a distance to the killer

The machine-wide availability gate read `kern.memorystatus_level`, chosen deliberately as "the exact meter the killer reads".
That sysctl is the memorystatus subsystem's PRESSURE percentage, and pressure sits near 100 until the machine is already in trouble.

On this 38.655 GB machine it read 92 while a marked backward exhausted the GPU, so the gate reported 35.56 GB available and waved the run through.
For a 26 GB budget to have been refused the meter would have had to fall under 62, which is inside the range where Jetsam is killing processes rather than a range a run can be warned out of.

It now sums the `vm_stat` counters a new allocation can take without forcing compression or swap.
The generalisable mistake is the one to keep in view: reading the authority that acts is not the same as reading the quantity it acts on, and a gate that can only fire after the damage is a gate in name.

## Decision 3: when a cross-check dies with its evidence, what replaces it is named and its strength is stated

`ceiling_sweep._profile_inputs` read clause 8's per-shape call tally from every width and required the widths to AGREE.
Clause 67 leaves one width with a pass, so that check has nothing to compare and the reader refused every legitimate recording until it was repaired.

What replaces the agreement is not a weaker measurement but a different kind of evidence, and Amendment 18 clause 71 states which half is which rather than presenting both as proof.
The tallied shape LABEL is a proof: it is derived from `layer.weight.shape` and `layer.bits`, and a quantity a construction never receives cannot move it.
The per-shape COUNT is an argument from the pinned arrangement: a quantized layer is invoked once per layer per direction in a step, and the token count enters as the size of its operand rather than as more invocations.

The rule is that a removed check is replaced in writing, with the replacement's strength stated, because the failure mode is not a missing check but a missing check that reads as present.
