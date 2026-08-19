"""The order candidates are judged in, and the rule that order exists to keep.

The sprint pre-registration (docs/research/2026-08-19-metalrunner-sprint1-prereg.md,
section 6) fixes a funnel: compile, source lint, tolerance-free gates,
tolerance gate, and only then pricing. Its rule V1 is the one that matters:

    no candidate is timed before it is verified.

A rule written only in a document is a rule that gets broken by someone
wiring stages in a hurry, so it is enforced here in two places. Building a
funnel whose stages put a timing stage before a verification stage is
refused outright, before anything runs; and at run time the funnel stops at
the first stage a candidate fails, so a timing stage is unreachable for a
candidate that failed anything ahead of it. Together those make a timed
unverified candidate impossible to produce by mis-ordering a list.

Stages that raise are the other thing this file is careful about. A
generated kernel is hostile input and can break a stage in ways its author
did not foresee, so an exception is caught and recorded rather than losing
the whole session. It is recorded as `errored`, never as `failed`: a stage
that raised did not reach a verdict, and letting a bug in our own code count
as a census of bad kernels is exactly the sort of plausible wrong number
this repository exists to refuse.

Nothing here knows what any stage does. That is the point: the gates decide
correctness, the pricing decides speed, and this decides only what is
allowed to run after what.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kernelverify.compiler.store import CandidateStore


@dataclass(frozen=True)
class StageOutcome:
    """One stage's verdict about one candidate."""

    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Stage:
    """A named step, and whether it is one of the steps that earn a timing.

    `verifies` is the whole of V1's machinery. A stage that verifies asks
    whether the candidate is correct; a stage that does not is measuring it,
    and measuring may only happen behind verification.
    """

    name: str
    run: Callable[..., StageOutcome]
    verifies: bool = True


@dataclass(frozen=True)
class FunnelResult:
    candidate: str
    reached: str
    passed: bool
    errored: bool = False
    detail: str = ""


class FunnelOrderError(ValueError):
    """The stage list would let something be timed before it was verified."""


class Funnel:
    def __init__(self, stages: list[Stage]):
        if not stages:
            raise FunnelOrderError("a funnel with no stages verifies nothing")
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise FunnelOrderError(f"stage names must be unique: {names}")

        first_timing = next((i for i, s in enumerate(stages) if not s.verifies),
                            len(stages))
        late = [s.name for s in stages[first_timing:] if s.verifies]
        if late:
            raise FunnelOrderError(
                f"{late} verify but run after {stages[first_timing].name}, "
                f"which does not: that orders a timing before a verification, "
                f"which rule V1 of the pre-registration forbids")
        self.stages = tuple(stages)

    def screen(self, store: CandidateStore, candidate: str,
               **context) -> FunnelResult:
        """Run one candidate down the funnel, recording every step it reached."""
        for stage in self.stages:
            try:
                outcome = stage.run(store.source(candidate), **context)
            except Exception as error:  # a candidate is hostile input
                detail = f"{type(error).__name__}: {error}"
                store.record(candidate, stage=stage.name, passed=False,
                             detail=detail, errored=True)
                return FunnelResult(candidate, stage.name, passed=False,
                                    errored=True, detail=detail)

            store.record(candidate, stage=stage.name, passed=outcome.passed,
                         detail=outcome.detail)
            if not outcome.passed:
                return FunnelResult(candidate, stage.name, passed=False,
                                    detail=outcome.detail)

        return FunnelResult(candidate, self.stages[-1].name, passed=True)
