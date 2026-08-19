"""Did the kernel write every cell it declared? A screen with no tolerance in it.

The cheapest way a generated kernel is wrong is that part of its output never
gets written: a grid an element short, a bound off by one, an early return, a
tail nobody covered. The runner hides this by design, because it clears every
output before the correctness dispatch, so an untouched cell reads back as a
plain 0 and no comparison can tell it apart from a computed 0.

The mechanism is to run each case twice with complementary fills, 0xA5 and
0x5A, and refuse only cells that came back holding BOTH. For a kernel whose
output does not depend on the buffer's prior contents, the two passes produce
identical bytes, so it can match at most one of the two and refusal is
impossible for it by construction rather than by luck.

Byte patterns rather than a NaN sentinel, for two reasons that are not
preferences. A shipped, verified kernel in this repository writes NAN into
every output cell on its capacity-overflow path, and the next gate in the
funnel exists to run kernels whose correct output IS NaN, so "a NaN left
behind means unwritten" refuses correct kernels. And half the tensor dtypes
are integer carriers where no value is out of range, so no single sentinel
can ever be safe for them.

Where the gate would have to make a rule, it abstains instead
---------------------------------------------------------------
A kernel may legitimately accumulate into its output binding: a reduction
blocked across threadgroups can only be assembled that way on Metal, and the
tolerance contract grants blocking at any width. Such a kernel reads the
fill, so on a cell whose true value the pattern absorbs it hands back the
pattern in both passes and looks unwritten. The first draft of this gate
answered that by forbidding output-reading kernels, which would have refused
something the contract admits.

It abstains instead. A kernel that does not read its output writes identical
bytes in both passes on every cell it touches, so a cell that differs between
the passes without being unwritten is proof that the fill was read. When any
such cell appears, the case is recorded as not screened and no unwritten
claim is made about it. Abstaining costs coverage; refusing would have cost a
promise.

What a clean verdict does and does not mean
-------------------------------------------
It attests that a store reached every declared cell. It does NOT attest that
the kernel produced each cell's value. A kernel that clears its own output
and then applies a wrong bound stores into every cell and passes here while
computing half of them; that fault belongs to the tolerance gate, and this
gate's policy string says so rather than letting a clean verdict be read as
coverage it does not have.

One hole remains and is named rather than hidden: a kernel that reads its
output AND whose every declared cell is absorbed by the pattern shows no
differing cell, so the abstention does not trigger and the cells read as
unwritten. It needs an accumulating kernel whose entire output is absorbed,
which the abstention catches the moment any one cell is not.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

import numpy as np

from kernelverify.runners.result import RunStatus
from kernelverify.runners.spec import PREFILL_PATTERNS, TENSOR_DTYPES

POLICY = (
    "every declared output cell must be reached by a store; screened by two "
    "dispatches per case whose output buffers are prefilled with the "
    "complementary byte patterns 0xA5 and 0x5A, refusing only cells that came "
    "back holding both. This attests that a store reached a cell, NOT that the "
    "kernel produced that cell's value: a kernel that clears its own output and "
    "then applies a wrong bound passes here and is the tolerance gate's to catch."
)

CLEAN = "clean"
UNWRITTEN = "unwritten"
PREFILL_DEPENDENT = "prefill-dependent"
NO_RUN = "no-run"

_UNSIGNED = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


@dataclass(frozen=True)
class CaseVerdict:
    label: str
    verdict: str
    unwritten_cells: int = 0
    first: tuple = ()
    detail: str = ""

    @property
    def screened(self) -> bool:
        """False when the gate declined to judge this case at all."""
        return self.verdict in (CLEAN, UNWRITTEN)

    @property
    def refused(self) -> bool:
        return self.verdict == UNWRITTEN


@dataclass(frozen=True)
class Report:
    policy: str
    cases: tuple[CaseVerdict, ...]

    @property
    def ok(self) -> bool:
        """No case was refused. Abstentions do not refuse; they cover nothing."""
        return not any(case.refused for case in self.cases)

    @property
    def screened(self) -> int:
        return sum(1 for case in self.cases if case.screened)

    @property
    def reason(self) -> str:
        refused = [c for c in self.cases if c.refused]
        if refused:
            return "; ".join(
                f"{c.label}: {c.unwritten_cells} cells never written, first at "
                f"{list(c.first)}" for c in refused)
        skipped = [c for c in self.cases if not c.screened]
        if skipped:
            return "not screened: " + "; ".join(
                f"{c.label} ({c.verdict})" for c in skipped)
        return ""


def _pattern_word(name: str, itemsize: int) -> int:
    byte = PREFILL_PATTERNS[name]
    return int.from_bytes(bytes([byte]) * itemsize, sys.byteorder)


def _judge(a: np.ndarray, b: np.ndarray, dtype: str):
    """Compare one output's bytes from the two passes, through an unsigned view.

    The view is load-bearing: comparing floats would never match a fill that
    happens to be a NaN bit pattern, and the gate would silently pass
    everything it was built to catch.
    """
    itemsize = np.dtype(TENSOR_DTYPES[dtype]).itemsize
    view = _UNSIGNED[itemsize]
    left, right = a.view(view), b.view(view)
    unwritten = (left == _pattern_word("sentinel_a", itemsize)) & \
                (right == _pattern_word("sentinel_b", itemsize))
    read_prefill = (left != right) & ~unwritten
    return unwritten, read_prefill


def screen(runner, spec, cases, *, first_n: int = 4) -> Report:
    """Run every case under both fills and say what each one showed."""
    doubled = []
    for case in cases:
        doubled.append(replace(case, prefill="sentinel_a",
                               label=f"{case.label}|a"))
        doubled.append(replace(case, prefill="sentinel_b",
                               label=f"{case.label}|b"))

    # warmup and repeats are zero: this gate never times anything, and a
    # repeat would run the kernel again over a buffer it has already written.
    results = runner.run(spec, doubled, warmup=0, repeats=0)

    verdicts = []
    for index, case in enumerate(cases):
        pass_a, pass_b = results[2 * index], results[2 * index + 1]
        if pass_a.status is not RunStatus.OK or pass_b.status is not RunStatus.OK:
            verdicts.append(CaseVerdict(
                case.label, NO_RUN,
                detail=(pass_a.detail or pass_b.detail or
                        f"{pass_a.status.value}/{pass_b.status.value}")))
            continue
        verdicts.append(_verdict_for(case, pass_a, pass_b, first_n))
    return Report(policy=POLICY, cases=tuple(verdicts))


def _verdict_for(case, pass_a, pass_b, first_n: int) -> CaseVerdict:
    total, first = 0, ()
    for slot, (shape, dtype) in enumerate(case.output_shapes):
        unwritten, read_prefill = _judge(pass_a.outputs[slot],
                                         pass_b.outputs[slot], dtype)
        if read_prefill.any():
            return CaseVerdict(
                case.label, PREFILL_DEPENDENT,
                detail=(f"output {slot} differs between the two fills, so the "
                        f"kernel reads its own output buffer; the contract "
                        f"grants that, so nothing is claimed about this case"))
        if unwritten.any():
            coordinates = np.argwhere(unwritten)[:first_n]
            total += int(unwritten.sum())
            first = first or tuple(tuple(int(i) for i in c) for c in coordinates)
    if total:
        return CaseVerdict(case.label, UNWRITTEN, unwritten_cells=total,
                           first=first)
    return CaseVerdict(case.label, CLEAN)
