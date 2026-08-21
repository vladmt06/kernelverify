# ADR 0020: a timed region builds no operand, and a stage that cannot run has never been measured

Date: 2026-08-21
Status: Accepted

## Context

Amendment 13 changed three measurement methods in one range of commits: both benches began interleaving their arms inside every round, the plan validator began refusing a seed mlx-lm will not honour, and an exploratory pass was registered that reads a margin before any calibration exists.
This repository's working rule is that every method change carries a measurement behind it and an ADR entry recording what forced it.
No ADR was written, and an outside audit of the amendment found the omission.

Two further method changes landed the next day under the same rule, and this entry covers all of them, because they are one finding in two places: a stage nothing has ever run has never been measured, and a stage whose clock starts before its operands exist measures the operands.

## Decision 1: a timed callable receives its operands already materialised

Candidate L's bench builds a cotangent for the extra matmul that stands in for the hidden-gradient path a streamed kernel must still produce.
It built that cotangent inside the timed callable, at the KEPT vocabulary, which is the dial the bench fits its slope against.
So the fitted floor slope charged candidate L for generating a tensor whose size moves with the very quantity the slope measures.

Measured at the pinned 4B dimensions, at the registered seed-7 supervised counts, five rounds behind three warm-ups, one construction against the other in the same session:

| Width | Cotangent built inside the timed call | Built with the other operands | Bias |
|---|---|---|---|
| short, 91 supervised rows | 78.039 ms | 70.640 ms | +10.5% |
| long, 3699 supervised rows | 2273.903 ms | 1915.364 ms | +18.7% |

That slope is the credited denominator `F` of candidate L's saving, so the bias sat inside the credit and not beside it.
The cotangent depends only on the supervised row count and the kept vocabulary, both fixed for the life of an arm, so it is materialised where the weight and the labels already were.

The general rule this fixes is the one worth keeping: a timed region may contain no allocation whose size depends on the dial, because a fit cannot tell that allocation apart from the operation it is fitting.

## Decision 2: the parent-to-child boundary is crossed by a test, not by inspection

The profile spawns a child per cell, and the child rechecks the inputs its parent hashed.
The parent has always accepted two plan shapes: one dataset under `data`, which the end-to-end harness pins, and two bands under `bands`, which the profile pins because its two registered widths come from two bands of one corpus.
The child's recheck read `plan["data"]` unconditionally.

Every profile child therefore raised `KeyError` before it loaded a model, and the parent mapped the traceback to child death.
The profile has never run end to end.
This survived because no test crossed that boundary with a two-band plan: one spawn test returns early without a device, and the other replaces the check it would have exercised.

Two tests now drive a real two-band plan from `preflight_inputs` into `_check_child_inputs`, and one of them changes a band underneath the child to prove the check still refuses what it exists to refuse.

## Decision 3: a sweep that opened on a busy machine is not made

`run_profile` samples the idle gate and returns `EXIT_NOT_IDLE` rather than measuring.
The ceiling sweep sampled the same gate, stored the answer in its recording and never read it, while its blockers examined only the closing gate.
A sweep whose arms ran against a competing process therefore landed as a usable recording.

The sweep now refuses at its opening gate, which is where the profile refuses, rather than gaining a blocker that would have masked the same fault after paying for it.

## Decision 4: the exploratory reader applies no gate

Amendment 15 clause 64 carries the registered form of this and the reasoning is not repeated here.
What belongs in this entry is the measurement that forced it: on the reducer's own fixtures, an arbitrary 1.0 ms sentinel produced a margin at both widths where 0.1 ms produced none, so a constant nothing registers decided the one number the exploratory pass exists to produce.

## Consequences

Every figure candidate L's bench has produced before today is superseded, including the arm prices Amendment 13 clause 52 corrected and the offsets clause 55 reports.
Those figures were used to price schedules and to diagnose an interleaving fault, and both uses survive: the fault was a spread across repeats rather than a level, and the schedule arithmetic is dominated by the step and not the bench.
No credited saving has been computed from them, because no floor has ever been recorded.

The profile's first real end-to-end run is therefore also the first time this repository will see a profile recording at all, and the exploratory pass is that run.
