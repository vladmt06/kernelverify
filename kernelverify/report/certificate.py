"""The certificate: what a kernel's verification actually proves.

A certificate ships beside a kernel and lets anyone re-check the claim. Its
whole value is that it is precise about the difference between two things a
reader will otherwise conflate:

    byte-bound      hashes that identify WHAT was verified. They pin the exact
                    translation unit, catalogue and harness. They say nothing
                    about outputs.
    protocol-bound  the assertions themselves. These say the VERDICTS reproduce
                    when the published protocol is re-run, never that numbers
                    come out identical.

Byte-exactness is not achievable on a GPU: reduction order, compiler version
and power state all move the last bits legitimately. A certificate claiming it
would be false, so the two blocks are separated structurally rather than by a
sentence of prose someone can skim past.

The second thing this module refuses to blur: not every clause of the
admissible-implementation contract is enforced the same way. Some are checked
numerically by the battery; some provably cannot be, and are attested from the
kernel's source instead. C1 at fp16 activations is the measured example -
sequential half-precision accumulation is caught numerically on 192 of 280
cases, while tree-reduced half-precision accumulation is separable on 0 of 280
and is therefore a source-level attestation or nothing. A certificate that
listed both as "verified" would be claiming a check it never ran.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable

SCHEMA_VERSION = 1

#: How a contract clause was established for this kernel.
NUMERIC = "numeric-battery"        # the battery can separate a violation
STRUCTURAL = "source-attestation"  # provably not separable; read from source
MECHANISMS = (NUMERIC, STRUCTURAL)

#: How the verified translation unit was obtained.
EXTRACTED_VALIDATED = "extracted-behaviorally-validated"
EXTRACTED_UNVALIDATED = "extracted-unvalidated"
WRAPPER_ASSEMBLED = "wrapper-assembled"
PROVENANCE = (EXTRACTED_VALIDATED, EXTRACTED_UNVALIDATED, WRAPPER_ASSEMBLED)

PROVENANCE_MEANING = {
    EXTRACTED_VALIDATED:
        "the translation unit was captured from the framework's own generator and "
        "reproduces the framework's outputs under this protocol on the pinned "
        "toolchain; it is not claimed to be byte-identical to what the framework "
        "compiled, because numerically equivalent programs pass that check by "
        "construction and compile options never appear in the captured source",
    EXTRACTED_UNVALIDATED:
        "the translation unit was captured from the framework's own generator, but "
        "the reproduction check has not been run on this toolchain version",
    WRAPPER_ASSEMBLED:
        "we assembled the translation unit ourselves; it attests our wrapper, not "
        "the program the framework ships",
}

#: Things a source hash structurally cannot cover, stated so a reader does not
#: assume the certificate reaches them. Each was measured, not guessed.
NOT_ATTESTED = (
    "host-side contiguity copies the framework may insert before launch",
    "launch configuration beyond the declared spec",
    "compile options, which do not appear in generated source and are recorded "
    "separately as inputs",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class ClauseAttestation:
    """One contract clause, and how it was actually established."""

    clause: str            # "C1", "C4", ...
    statement: str         # what the clause requires, in the contract's words
    mechanism: str         # NUMERIC or STRUCTURAL
    evidence: str          # the measurement or the source fact
    scope: str = ""        # when the mechanism is conditional, say on what

    def __post_init__(self):
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"unknown attestation mechanism {self.mechanism!r}")
        if not self.evidence:
            raise ValueError(f"clause {self.clause} has no evidence; an unevidenced "
                             "attestation is the thing this module exists to prevent")

    def to_json(self) -> dict:
        out = {"clause": self.clause, "statement": self.statement,
               "mechanism": self.mechanism, "evidence": self.evidence}
        if self.scope:
            out["scope"] = self.scope
        return out


@dataclass(frozen=True)
class PerformanceClaim:
    """A speed claim, scoped and sourced, or not made at all.

    Two rules the pack lane established by measurement: a claim is scoped to
    the dispatch range it was measured over, never to the operator, and a claim
    sourced from a non-binding row is not published as an absolute.
    """

    ratio: float
    against: str
    scope: str               # "batch 5-11", "T <= 1024", ...
    binding: bool
    sampling_group: str = ""

    def to_json(self) -> dict:
        return {"ratio": self.ratio, "against": self.against, "scope": self.scope,
                "binding": self.binding, "sampling_group": self.sampling_group,
                "status": "binding" if self.binding
                          else "directional, pending a certified measurement"}


@dataclass(frozen=True)
class Certificate:
    """Everything needed to re-check a kernel's verification claim."""

    kernel_name: str
    operator: str
    translation_unit: str          # the exact source that was verified
    extraction_provenance: str
    compile_options: dict
    clauses: tuple                 # ClauseAttestation
    catalogue_fingerprint: str     # which fault population
    harness_commit: str
    contract_version: str
    tolerance_model: dict          # K values and ensemble membership
    policy: str                    # which test policy selected the cases
    budget: int                    # evaluations per operator
    cases_run: int
    cases_passed: int
    seed_protocol: str             # how a re-checker draws the same cases
    chip_generation: str
    toolchain: dict
    performance: tuple = ()        # PerformanceClaim
    notes: tuple = ()

    def __post_init__(self):
        if self.extraction_provenance not in PROVENANCE:
            raise ValueError(f"unknown provenance {self.extraction_provenance!r}")
        if self.cases_passed > self.cases_run:
            raise ValueError("more cases passed than were run")
        if not self.clauses:
            raise ValueError("a certificate with no clause attestations asserts nothing")

    # -- the two structurally separated blocks -----------------------------
    def byte_bound(self) -> dict:
        """Hashes that identify what was verified. Not a claim about outputs."""
        return {
            "_meaning": "these hashes identify WHAT was verified; they assert "
                        "nothing about output values",
            "translation_unit_sha256": _sha256(self.translation_unit),
            "catalogue_fingerprint": self.catalogue_fingerprint,
            "harness_commit": self.harness_commit,
            "contract_version": self.contract_version,
            "compile_options": self.compile_options,
            "extraction_provenance": self.extraction_provenance,
            "extraction_provenance_meaning": PROVENANCE_MEANING[self.extraction_provenance],
            "not_attested_by_these_hashes": list(NOT_ATTESTED),
        }

    def protocol_bound(self) -> dict:
        """The assertions. Reproducibility of VERDICTS, never of bits."""
        return {
            "_assertion": "re-running the published protocol on a machine matching "
                          "the recorded chip generation and toolchain reproduces "
                          "these verdicts; output values are NOT asserted to be "
                          "identical, which is not achievable on a GPU",
            "operator": self.operator,
            "policy": self.policy,
            "budget_evaluations_per_operator": self.budget,
            "cases_run": self.cases_run,
            "cases_passed": self.cases_passed,
            "all_passed": self.cases_passed == self.cases_run,
            "seed_protocol": self.seed_protocol,
            "tolerance_model": self.tolerance_model,
        }

    def attestation(self) -> dict:
        numeric = [c for c in self.clauses if c.mechanism == NUMERIC]
        structural = [c for c in self.clauses if c.mechanism == STRUCTURAL]
        return {
            "_meaning": "not every clause is established the same way; a clause no "
                        "test can separate is attested from source or not at all",
            "clauses": [c.to_json() for c in self.clauses],
            "counts": {"numeric": len(numeric), "source_attested": len(structural)},
        }

    def validity(self) -> dict:
        return {
            "binding_on": {"chip_generation": self.chip_generation,
                           "toolchain": self.toolchain},
            "elsewhere": "advisory: a mismatched chip generation or toolchain may "
                         "move numerics legitimately, so a failure there is a new "
                         "matrix row to investigate, not a refutation",
        }

    def to_json(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "kernel": self.kernel_name,
            "byte_bound": self.byte_bound(),
            "protocol_bound": self.protocol_bound(),
            "attestation": self.attestation(),
            "validity": self.validity(),
            "performance": [p.to_json() for p in self.performance],
            "notes": list(self.notes),
        }

    def dumps(self) -> str:
        return json.dumps(self.to_json(), indent=2, sort_keys=False)


def audit(certificate: Certificate) -> list[str]:
    """Complaints a reviewer would make. Empty means the certificate is honest.

    This runs before a certificate is published, and it is deliberately picky:
    the failure mode it guards against is a certificate that reads as stronger
    than the verification behind it.
    """
    problems: list[str] = []
    cert = certificate

    if cert.cases_passed != cert.cases_run:
        problems.append(
            f"{cert.cases_run - cert.cases_passed} of {cert.cases_run} cases failed; "
            "a certificate is not issued for a kernel that did not pass")

    if cert.budget and cert.cases_run < cert.budget:
        problems.append(
            f"only {cert.cases_run} cases run against a stated budget of {cert.budget}")

    if cert.extraction_provenance == WRAPPER_ASSEMBLED:
        problems.append(
            "wrapper-assembled: this attests our own wrapper rather than the program "
            "the framework ships, and the kernel description must say so plainly")

    if cert.extraction_provenance == EXTRACTED_UNVALIDATED:
        problems.append(
            "extraction not validated on this toolchain; the reproduction check must "
            "run before the top provenance tier may be claimed")

    for claim in cert.performance:
        if not claim.scope:
            problems.append(
                f"performance claim of {claim.ratio}x has no scope; a speed claim "
                "belongs to the dispatch range it was measured over, not to the operator")
        if claim.binding and not claim.sampling_group:
            problems.append(
                f"performance claim of {claim.ratio}x claims binding without a "
                "sampling group; a ratio needs interleaved rows from one group")

    covered = {c.clause for c in cert.clauses}
    if "C1" not in covered:
        problems.append(
            "C1 (working-precision floor) is unattested; it is the clause that keeps "
            "a precision fault a fault, so its absence voids the certificate")

    return problems


def render_markdown(certificate: Certificate) -> str:
    """A human-readable summary that leads with the limits, not the claim."""
    cert = certificate
    lines = [f"# Certificate: {cert.kernel_name}", "",
             f"Operator `{cert.operator}`, verified at {cert.cases_passed}/"
             f"{cert.cases_run} cases under the `{cert.policy}` policy.", ""]

    lines += ["## What this asserts", "",
              "Re-running the published protocol on a matching machine reproduces "
              "these verdicts.", "",
              "## What this does not assert", "",
              "That output values are identical. GPU arithmetic is not "
              "bit-reproducible, and a certificate claiming otherwise would be false.",
              ""]

    lines += ["## Contract clauses and how each was established", "",
              "| Clause | Requirement | Established by | Evidence |",
              "|---|---|---|---|"]
    for clause in cert.clauses:
        mechanism = ("battery measurement" if clause.mechanism == NUMERIC
                     else "reading the kernel source")
        scope = f" ({clause.scope})" if clause.scope else ""
        lines.append(f"| {clause.clause} | {clause.statement} "
                     f"| {mechanism}{scope} | {clause.evidence} |")
    lines.append("")

    if cert.performance:
        lines += ["## Speed", "", "| Ratio | Against | Valid for | Status |",
                  "|---|---|---|---|"]
        for claim in cert.performance:
            status = ("binding" if claim.binding
                      else "directional, pending a certified measurement")
            lines.append(f"| {claim.ratio:.2f}x | {claim.against} | {claim.scope} "
                         f"| {status} |")
        lines.append("")

    lines += ["## Validity", "",
              f"Binding on {cert.chip_generation} with the recorded toolchain "
              f"({', '.join(f'{k} {v}' for k, v in sorted(cert.toolchain.items()))}); "
              "advisory elsewhere.", ""]

    problems = audit(cert)
    if problems:
        lines += ["## Audit complaints", ""]
        lines += [f"- {p}" for p in problems]
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def contract_clauses(*, fp16_activations: bool) -> tuple:
    """The standard clause set for a quantized kernel on this project's contract.

    `fp16_activations` changes C1's mechanism, and that is the measured point:
    at fp32 activations the battery separates every half-precision accumulator;
    at fp16 activations it separates the sequential class only, so the
    tree-reduced class must be attested from source.
    """
    c1_numeric = ClauseAttestation(
        clause="C1",
        statement="every intermediate at binary32 or wider",
        mechanism=NUMERIC,
        evidence="half-precision accumulation separated on the battery at these "
                 "activation dtypes (280/280 cases at fp32 activations)",
    )
    c1_split = ClauseAttestation(
        clause="C1",
        statement="every intermediate at binary32 or wider",
        mechanism=STRUCTURAL,
        scope="fp16 activations, tree-reduced accumulators",
        evidence="tree-reduced half-precision accumulation is separable on 0/280 "
                 "fp16-activation cases, so it is read from the kernel source; the "
                 "sequential class remains numerically separated (192/280)",
    )
    return (
        c1_split if fp16_activations else c1_numeric,
        ClauseAttestation("C4", "exponential sums shifted by their own maximum",
                          NUMERIC, "unshifted sums overflow and are caught by the "
                                   "battery's structured input modes"),
        ClauseAttestation("C5", "the specified formula, not a defensible relative",
                          NUMERIC, "checked against the operator's fp64 reference"),
        ClauseAttestation("C6", "specification constants fixed",
                          NUMERIC, "epsilon and scale-exponent faults are catalogue "
                                   "entries and are caught"),
    )


def emit(certificates: Iterable[Certificate], directory) -> list:
    """Write certificates as JSON beside the kernels they describe."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for cert in certificates:
        problems = audit(cert)
        if problems:
            raise ValueError(
                f"refusing to emit a certificate for {cert.kernel_name}: "
                + "; ".join(problems))
        path = directory / f"{cert.kernel_name}.certificate.json"
        path.write_text(cert.dumps() + "\n")
        written.append(path)
    return written
