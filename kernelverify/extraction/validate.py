"""Judge an extraction: does the extracted kernel match the surface it left?

The two arms are the extracted source run by our Metal runner and the native
surface run by the live MLX arm. The match criterion is the shipped oracle's
verdict function - the caller passes it in, closed over whatever tolerance
machinery the operator uses - applied per case across the battery, structured
modes included. Never bitwise: the two arms legitimately differ by rounding,
and a bitwise criterion would reject every honest extraction.

The compile-option alarm is the one diagnostic this module owns. A math-mode
divergence between the arms (one compiled fast, one safe) shows up as
reassociation error: failures that concentrate on the cases where summation
order can move the result, while order-insensitive cases keep passing. That
pattern raises `compile_option_alarm`; scattered failures do not, because
they mean the extraction itself is wrong, which no compile option explains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kernelverify.extraction.mlx_arm import LiveResult
from kernelverify.runners.result import RunResult


@dataclass
class CaseOutcome:
    label: str
    #: True/False is the verdict on a judged case; None means an arm never
    #: produced arrays (harness failure), so nothing numerical was judged.
    matched: bool | None
    sensitive: bool
    detail: str = ""


@dataclass
class ExtractionReport:
    outcomes: list = field(default_factory=list)
    compile_option_alarm: bool = False

    @property
    def judged(self) -> list:
        return [o for o in self.outcomes if o.matched is not None]

    @property
    def mismatches(self) -> list:
        return [o for o in self.outcomes if o.matched is False]

    @property
    def broken(self) -> list:
        return [o for o in self.outcomes if o.matched is None]

    @property
    def all_matched(self) -> bool:
        return bool(self.judged) and not self.mismatches and not self.broken

    def summary(self) -> str:
        lines = [
            f"{len(self.judged)} judged, {len(self.mismatches)} mismatched, "
            f"{len(self.broken)} unjudgeable"
            + ("; COMPILE-OPTION ALARM" if self.compile_option_alarm else "")
        ]
        for outcome in self.outcomes:
            if outcome.matched is not True:
                state = "BROKEN" if outcome.matched is None else "MISMATCH"
                lines.append(f"  {state} [{outcome.label}] {outcome.detail}")
        return "\n".join(lines)


def validate_extraction(extracted_results, live_results, sensitive, verdict,
                        labels=None) -> ExtractionReport:
    """One verdict per case, plus the concentration diagnostic.

    `extracted_results` are the runner's `RunResult`s, `live_results` the live
    arm's, aligned by index with `sensitive`, a flag per case marking the
    reassociation-sensitive ones (opposed signs, long accumulations - the
    battery knows which of its modes those are). `verdict(index, extracted
    outputs, live outputs)` returns `(ok, detail)` and is expected to be the
    shipped oracle's verdict function for the operator.
    """
    extracted_results = list(extracted_results)
    live_results = list(live_results)
    sensitive = list(sensitive)
    if not len(extracted_results) == len(live_results) == len(sensitive):
        raise ValueError("extracted, live, and sensitivity lists must align")

    outcomes = []
    for index, (ours, live) in enumerate(zip(extracted_results, live_results)):
        label = (labels[index] if labels is not None else "") or \
            getattr(ours, "label", "") or f"case {index}"
        if isinstance(ours, RunResult) and not ours.ok:
            outcomes.append(CaseOutcome(label, None, sensitive[index],
                                        f"extracted arm: {ours.status.value}: {ours.detail}"))
            continue
        if isinstance(live, LiveResult) and not live.ok:
            outcomes.append(CaseOutcome(label, None, sensitive[index],
                                        f"live arm: {live.error}"))
            continue
        ok, detail = verdict(index, ours.outputs, live.outputs)
        outcomes.append(CaseOutcome(label, bool(ok), sensitive[index], detail))

    mismatched = [o for o in outcomes if o.matched is False]
    insensitive_passed = any(o.matched is True and not o.sensitive for o in outcomes)
    alarm = (bool(mismatched)
             and all(o.sensitive for o in mismatched)
             and insensitive_passed)
    return ExtractionReport(outcomes=outcomes, compile_option_alarm=alarm)
