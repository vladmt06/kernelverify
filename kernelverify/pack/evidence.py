"""Structured evidence from a pack gate's verification run.

A gate used to prove itself by printing; anything downstream would have had
to parse stdout, which the certificate lane's eng review ruled out (D7).
Instead `verify()` now returns a `GateEvidence`, the printed banner renders
FROM it, and the certificate emitter consumes it directly.

The structure mirrors what a certificate needs and nothing more:

- one `SpecializationEvidence` per compile-time specialization of a kernel
  (the unit a certificate is issued for - one translation unit, one hash);
- one `CaseEvidence` per runner-isolated case, carrying the verdict, the
  error and tolerance that produced it, and an advisory fingerprint of the
  output bytes (GPU outputs are not bit-stable, so the fingerprint describes
  the certifying run and is never a re-check criterion);
- gate-level `checks` for the side conditions that make the gate honest but
  are not kernel verdicts: artefact-byte identity with mx.quantize, the
  incumbent arm passing the same oracle.

`SpecializationEvidence.calls` carries the MLX-door `LiveCall` for every
case, aligned with `cases` by index, which is exactly what the extraction
capture and behavioral validation need; the emitter never re-derives inputs.

Nothing in this module records a timing, deliberately: verification may run
under contention (D12.2), so a timing recorded here would be an invitation
to quote it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from kernelverify.extraction.surface import LiveCall

__all__ = [
    "CaseEvidence",
    "GateEvidence",
    "SpecializationEvidence",
    "by_family",
    "output_fingerprint",
    "render_banner",
]


def output_fingerprint(array) -> str:
    """Advisory sha256 of the output bytes from the certifying run."""
    contiguous = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


@dataclass(frozen=True)
class CaseEvidence:
    """One runner-isolated case's verdict, with the numbers behind it.

    `err` and `tol` are None when nothing numerical was judged (a runner
    failure, or a check that is a boolean fact rather than a comparison).
    """

    label: str
    passed: bool
    err: float | None = None
    tol: float | None = None
    detail: str = ""
    output_sha256: str = ""
    aux: dict = field(default_factory=dict)


@dataclass
class SpecializationEvidence:
    """Everything the gate learned about one compile-time specialization.

    `template` holds the raw-door `specialize()` substitutions; each entry of
    `calls` holds the same case as an MLX-door `LiveCall` (template values in
    MLX's spelling), aligned with `cases` by index.
    """

    kernel: str        # kernel family name, e.g. "kv_wide_qmv"
    operator: str      # the NATIVE_OPS operator the cases were judged against
    template: dict     # raw-door substitutions, e.g. {"T": "half", "BITS": 4, ...}
    threadgroup: tuple = (1, 1, 1)
    cases: list = field(default_factory=list)      # CaseEvidence
    calls: list = field(default_factory=list)      # LiveCall, aligned with cases

    @property
    def cases_run(self) -> int:
        return len(self.cases)

    @property
    def cases_passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def ok(self) -> bool:
        return bool(self.cases) and self.cases_passed == self.cases_run

    def add_case(self, case: CaseEvidence, call: LiveCall | None = None) -> None:
        self.cases.append(case)
        if call is not None:
            self.calls.append(call)


@dataclass
class GateEvidence:
    """One gate run: its policy, its specializations, and its side checks."""

    gate: str            # e.g. "pack_wide_qmv"
    policy: str          # the TRUE pack-gate policy, in words a reader can re-run
    seed_protocol: str   # how a re-checker draws the same cases
    specializations: list = field(default_factory=list)   # SpecializationEvidence
    checks: list = field(default_factory=list)            # CaseEvidence

    @property
    def cases_run(self) -> int:
        return sum(s.cases_run for s in self.specializations)

    @property
    def cases_passed(self) -> int:
        return sum(s.cases_passed for s in self.specializations)

    @property
    def checks_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def ok(self) -> bool:
        return (bool(self.specializations)
                and all(s.ok for s in self.specializations)
                and self.checks_passed)

    def specialization(self, kernel: str, operator: str, template: dict,
                       threadgroup: tuple) -> SpecializationEvidence:
        """The evidence bucket for `template`, created on first use."""
        for existing in self.specializations:
            if existing.kernel == kernel and existing.template == template:
                return existing
        created = SpecializationEvidence(kernel=kernel, operator=operator,
                                         template=dict(template),
                                         threadgroup=tuple(threadgroup))
        self.specializations.append(created)
        return created


def by_family(evidences) -> dict:
    """Specializations of every gate, grouped by kernel family name."""
    families: dict = {}
    for evidence in evidences:
        for spec in evidence.specializations:
            families.setdefault(spec.kernel, []).append(spec)
    return families


def _case_line(case: CaseEvidence) -> str:
    if case.err is None or case.tol is None:
        verdict = "pass" if case.passed else f"FAIL: {case.detail or 'no verdict'}"
        return f"{case.label}  {verdict}"
    line = (f"{case.label}  err {case.err:9.3e}  tol {case.tol:9.3e}  "
            f"{'pass' if case.passed else 'FAIL'}")
    return line + (f"  ({case.detail})" if case.detail else "")


def render_banner(evidence: GateEvidence, header: str) -> None:
    """The gate's console banner, rendered from the evidence and nothing else."""
    print(header)
    for check in evidence.checks:
        print(f"  [check] {_case_line(check)}")
    for spec in evidence.specializations:
        joined = " ".join(f"{k}={v}" for k, v in sorted(spec.template.items()))
        print(f"  {spec.kernel} [{joined}] "
              f"{spec.cases_passed}/{spec.cases_run} passed")
        for case in spec.cases:
            print(f"    {_case_line(case)}")
    verdict = "green" if evidence.ok else "NOT GREEN"
    print(f"  gate {evidence.gate}: {evidence.cases_passed}/{evidence.cases_run} "
          f"cases, checks {'pass' if evidence.checks_passed else 'FAIL'} -> {verdict}")
