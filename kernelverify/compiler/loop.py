"""Run proposals through the funnel and say what the session produced.

This is the whole of the generate-gate-keep cycle: something proposes
candidates, the funnel judges each one, the store remembers all of it, and at
the end the session reports what happened. Nothing here decides whether a
kernel is correct or fast; it decides only that every proposal gets recorded
and counted, including the ones nothing looked at.

Two counts are kept apart on purpose.

A re-proposal is not a failure. A knob sweep whose knob does not appear in the
body produces the same source twice, and so does a generator that repeats
itself; the store refuses it by hash before it costs a compile, which is the
cheapest rejection in the loop. Counting those as candidates would inflate the
session's own yield, and counting them as failures would invent faults that
never existed, so they get their own line.

Anything that reached a stage is in the census, which is what outcome O5 of
the sprint pre-registration asks for: how many were generated, how many died
where, and how long it took. A session that keeps nothing still reports it,
because the loop's yield is the thing being built and a zero measures it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kernelverify.compiler.funnel import Funnel
from kernelverify.compiler.store import CandidateStore, Reproposed


@dataclass(frozen=True)
class Proposal:
    """One candidate, with anything the funnel needs that is specific to it.

    `extra` is merged over the session's context, because a knob setting can
    change the launch configuration as well as the source, and the stage that
    compiles it has to be told the one that goes with this body.
    """

    source: str
    origin: str
    prompt: str | None = None
    parent: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Session:
    proposed: int
    reproposed: int
    survived: tuple[str, ...]
    census: dict
    seconds: float

    @property
    def offered(self) -> int:
        """Everything the generator handed over, duplicates included."""
        return self.proposed + self.reproposed

    def summary(self) -> str:
        lines = [f"offered {self.offered}, new {self.proposed}, "
                 f"already seen {self.reproposed}, "
                 f"survived {len(self.survived)}, {self.seconds:.1f}s"]
        for outcome, count in sorted(self.census.items()):
            lines.append(f"  {outcome}: {count}")
        return "\n".join(lines)


def run(proposals, funnel: Funnel, store: CandidateStore, *, context: dict,
        limit: int | None = None, clock=time.monotonic) -> Session:
    """Judge every proposal and return the session's own account of itself."""
    started = clock()
    new, repeated, survived = 0, 0, []

    for proposal in proposals:
        if limit is not None and new >= limit:
            break
        try:
            candidate = store.propose(proposal.source, origin=proposal.origin,
                                      parent=proposal.parent,
                                      prompt=proposal.prompt)
        except Reproposed:
            repeated += 1
            continue

        new += 1
        result = funnel.screen(store, candidate, **{**context, **proposal.extra})
        if result.passed:
            survived.append(candidate)

    return Session(proposed=new, reproposed=repeated,
                   survived=tuple(survived), census=store.census(),
                   seconds=clock() - started)
