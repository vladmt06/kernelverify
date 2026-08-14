"""Interleaved A/B timing between two live workers, with a noise canary.

`compare(spec_a, spec_b, cases)` answers the optimiser's second question -
which of two working kernels is faster - under the discipline ADR 0006's
measurements force:

- Both specs stay compiled in their own live worker for the whole
  comparison, one process per arm, and every round dispatches the reference
  arm and then the candidate back to back. Interleaving is cheap and it is
  the default for any published comparison; ADR 0006 records that ms-scale
  dispatch alone is not sufficient and neither is interleaving alone.
- `spec_a` is the reference arm, and it doubles as the canary. Any round in
  which the reference arm's own samples spread wider than `spread_limit`
  (max over min, default 1.5, decision D6) is auto-rejected: the machine
  moved during that round, so the round supports no ratio. A comparison
  that rejected any round is not `ok`, and a gate built on it must exit
  non-zero rather than publish ratios the reference arm cannot back.
- Speedups are computed on `gpu_best` per ADR 0006 (interference only ever
  makes a kernel look slower), from the accepted rounds only, and the
  per-round data is kept in the report so a withheld claim is inspectable.

The workers run the same streaming protocol as the batch worker; a hang or
crash in either arm aborts the comparison with the arm's status in
`failure`, because a timing claim about a kernel that did not finish is not
a timing claim.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field

from kernelverify.runners.metal import MetalRunner, _EventStream, _stream
from kernelverify.runners.result import RunResult, RunStatus
from kernelverify.runners.spec import KernelSpec, RunCase, SpecError

DEFAULT_SPREAD_LIMIT = 1.5  # decision D6: the reference arm's max/min per round
DEFAULT_ROUNDS = 5


@dataclass
class RoundSample:
    """One interleaved round for one case: both arms, back to back."""

    round_index: int
    ref_best: float
    cand_best: float
    ref_spread: float
    cand_spread: float
    accepted: bool

    @property
    def speedup(self) -> float:
        """reference time over candidate time; above 1.0 the candidate wins."""
        return self.ref_best / self.cand_best if self.cand_best > 0 else float("nan")


@dataclass
class CaseComparison:
    label: str
    samples: list = field(default_factory=list)

    @property
    def accepted(self) -> list:
        return [s for s in self.samples if s.accepted]

    @property
    def rejected(self) -> int:
        return sum(1 for s in self.samples if not s.accepted)

    @property
    def speedup(self) -> float:
        """Median accepted-round speedup; NaN when no round survived."""
        accepted = self.accepted
        if not accepted:
            return float("nan")
        return statistics.median(s.speedup for s in accepted)


@dataclass
class CompareReport:
    cases: list = field(default_factory=list)
    spread_limit: float = DEFAULT_SPREAD_LIMIT
    failure: str = ""

    @property
    def rejected_rounds(self) -> int:
        return sum(c.rejected for c in self.cases)

    @property
    def ok(self) -> bool:
        """True only when every round of every case stood: no arm failure and
        no canary rejection. Gates exit non-zero on anything else."""
        return (not self.failure and bool(self.cases)
                and self.rejected_rounds == 0)

    def summary(self) -> str:
        if self.failure:
            return f"comparison failed: {self.failure}"
        lines = []
        for case in self.cases:
            verdict = (f"speedup {case.speedup:6.3f}x" if case.accepted
                       else "no accepted rounds")
            rejected = f", {case.rejected} round(s) rejected" if case.rejected else ""
            lines.append(f"  [{case.label}] {verdict} over "
                         f"{len(case.accepted)} round(s){rejected}")
        if self.rejected_rounds:
            lines.append(f"  REJECTED: reference-arm spread exceeded "
                         f"{self.spread_limit}x in {self.rejected_rounds} round(s); "
                         "the machine moved, rerun when quiet")
        return "\n".join(lines)


class _LiveArm:
    """One worker process holding one compiled spec across many dispatches."""

    def __init__(self, runner: MetalRunner, spec: KernelSpec, name: str,
                 warmup: int, repeats: int):
        self.name = name
        self.runner = runner
        self.process = runner._spawn(stream=True)
        self.errors: list[str] = []
        self.reader = _stream(self.process.stderr, self.errors)
        self.events = _EventStream(self.process.stdout)
        self._send({"spec": spec.to_json(), "warmup": warmup,
                    "repeats": repeats, "math_mode": runner.math_mode})
        self._await_compiled()

    def _send(self, payload: dict) -> None:
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def _await_compiled(self) -> None:
        budget = self.runner.startup_timeout
        while True:
            event = self.events.next_event(budget)
            if event is None:
                raise ComparisonAborted(self._died("before compiling"))
            kind = event.get("event")
            if kind == "compiled":
                return
            if kind == "fatal":
                result = RunResult.from_json(event["result"])
                raise ComparisonAborted(
                    f"arm {self.name}: {result.status.value}: {result.detail}")

    def time_case(self, index: int, case: RunCase) -> RunResult:
        self._send({"index": index, "case": case.to_json()})
        while True:
            try:
                event = self.events.next_event(self.runner.case_timeout)
            except TimeoutError:
                raise ComparisonAborted(
                    f"arm {self.name}: timed out on case {case.label!r}") from None
            if event is None:
                raise ComparisonAborted(self._died(f"on case {case.label!r}"))
            if event.get("event") == "fatal":
                result = RunResult.from_json(event["result"])
                raise ComparisonAborted(
                    f"arm {self.name}: {result.status.value}: {result.detail}")
            if event.get("event") == "case":
                result = RunResult.from_json(event["result"])
                if not result.ok:
                    raise ComparisonAborted(
                        f"arm {self.name} failed case {case.label!r}: "
                        f"{result.status.value}: {result.detail}")
                return result

    def _died(self, when: str) -> str:
        tail = "".join(self.errors).strip().splitlines()[-4:]
        return (f"arm {self.name}: the worker died {when}"
                + (f": {' / '.join(tail)}" if tail else ""))

    def close(self) -> None:
        try:
            self._send({"end": True})
        except (BrokenPipeError, ValueError, OSError):
            pass
        self.runner._terminate(self.process)
        self.reader.join(timeout=1.0)


class ComparisonAborted(RuntimeError):
    """An arm could not hold up its half of the comparison."""


def compare(spec_a: KernelSpec, spec_b: KernelSpec, cases, *,
            rounds: int = DEFAULT_ROUNDS,
            warmup: int = 1, repeats: int = 3,
            spread_limit: float = DEFAULT_SPREAD_LIMIT,
            runner: MetalRunner | None = None) -> CompareReport:
    """Interleaved A/B over `cases`; `spec_a` is the reference arm and canary.

    Returns a report whose `ok` is False - and whose caller must exit
    non-zero - when any arm failed or any round's reference-arm spread
    exceeded `spread_limit`. Ratios from the surviving rounds are still
    reported, labelled, so a rejected run is inspectable rather than mute.
    """
    runner = runner or MetalRunner()
    cases = list(cases)
    report = CompareReport(spread_limit=spread_limit)
    for spec, name in ((spec_a, "A"), (spec_b, "B")):
        for case in cases:
            try:
                case.validate_against(spec)
            except SpecError as error:
                report.failure = f"arm {name}, case {case.label!r}: {error}"
                return report

    arms: list[_LiveArm] = []
    try:
        arm_a = _LiveArm(runner, spec_a, "A (reference)", warmup, repeats)
        arms.append(arm_a)
        arm_b = _LiveArm(runner, spec_b, "B (candidate)", warmup, repeats)
        arms.append(arm_b)

        comparisons = [CaseComparison(label=case.label or f"case {i}")
                       for i, case in enumerate(cases)]
        for round_index in range(rounds):
            for index, case in enumerate(cases):
                ref = arm_a.time_case(index, case)
                cand = arm_b.time_case(index, case)
                ref_spread = ref.timing.gpu_spread
                comparisons[index].samples.append(RoundSample(
                    round_index=round_index,
                    ref_best=ref.timing.gpu_best,
                    cand_best=cand.timing.gpu_best,
                    ref_spread=ref_spread,
                    cand_spread=cand.timing.gpu_spread,
                    accepted=ref_spread <= spread_limit,
                ))
        report.cases = comparisons
    except ComparisonAborted as error:
        report.failure = str(error)
    except OSError as error:
        report.failure = f"a worker would not start: {error}"
    finally:
        for arm in arms:
            arm.close()
    return report
