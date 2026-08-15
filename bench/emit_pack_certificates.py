"""Emit verification certificates for the pack kernels, one per specialization.

The pipeline, ruled by the certificate-lane eng review:

1. Run the three pack gates' `verify()` and take their structured evidence
   objects; nothing here parses stdout (D7).
2. For every specialization that has actual gate evidence - kv_attention at
   4/8 bit only, since 2/3 bit have unit tests but no gate coverage; the
   exercised wide-qmv (BITS, M) pairs; the moe routing/dispatch
   specializations - capture the MLX door's generated translation unit in a
   fresh process and behaviorally validate it: the extracted source runs
   under the crash-isolated Metal runner, the surface runs natively in the
   live MLX arm, and the two must agree within the gate's own tolerance on
   the gate's own cases (D2).
3. Build one `Certificate` per specialization, one translation-unit hash
   each (D3). The protocol block names the REAL pack-gate policy with its
   TRUE case budget - never the 16-eval mutation-scored battery, which has
   not run against these operators (D10). Each certificate carries its
   validity domain (D9): kv_attention's T <= 1024 capacity ceiling is a
   CORRECTNESS bound and is stated as one. Advisory per-case output
   fingerprints and error-vs-tolerance margins are labeled not-reproducible
   (D8). No performance claims are made.
4. Refusal is per kernel family with audit-all-then-write-all inside each
   family (D10): a validation failure refuses that kernel's certificates
   with a named reason, and the other kernels still emit.
5. A markdown manifest indexes every certificate and every refusal.

Machine rule (D12.2): verification and extraction may run under contention,
so this script needs no idle gate, and it records no timings anywhere - a
timing measured under contention would only exist to be misquoted.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelverify.extraction import (  # noqa: E402
    ExtractionError,
    MLXKernelSurface,
    capture_specialization,
    extracted_spec,
    metadata_inputs,
    run_live,
    validate_extraction,
)
from kernelverify.pack import kv_attention, moe_dispatch, wide_qmv  # noqa: E402
from kernelverify.pack.evidence import (  # noqa: E402
    GateEvidence,
    SpecializationEvidence,
    by_family,
)
from kernelverify.report.certificate import (  # noqa: E402
    EXTRACTED_VALIDATED,
    NUMERIC,
    STRUCTURAL,
    Certificate,
    ClauseAttestation,
    EmitReport,
    emit,
)
from kernelverify.runners import LaunchSpec, MetalRunner, RunCase  # noqa: E402
from kernelverify.schemas.native_ops import K_QUANT  # noqa: E402
from kernelverify.schemas.quant_contract import ENSEMBLE  # noqa: E402

CERT_DIR = Path(__file__).resolve().parent / ".certificates"

#: Same-program-two-doors agreement bound for the routing gate weights: both
#: arms compute the same fp32 arithmetic, so anything beyond rounding noise
#: means the extraction is wrong. The top-2 indices must match exactly.
ROUTING_GATE_TOL = 1e-5


def _surface(family: str) -> MLXKernelSurface:
    """The MLX door of `family`, built from the pack's own door source."""
    doors = {
        wide_qmv.KERNEL_NAME: (wide_qmv.mlx_door_source, wide_qmv.INPUT_NAMES,
                               wide_qmv.OUTPUT_NAMES),
        kv_attention.KERNEL_NAME: (kv_attention.mlx_door_source,
                                   kv_attention.INPUT_NAMES,
                                   kv_attention.OUTPUT_NAMES),
        moe_dispatch.ROUTING_NAME: (moe_dispatch.routing_door_source,
                                    moe_dispatch.ROUTING_INPUTS,
                                    moe_dispatch.ROUTING_OUTPUTS),
        moe_dispatch.DISPATCH_NAME: (moe_dispatch.dispatch_door_source,
                                     moe_dispatch.DISPATCH_INPUTS,
                                     moe_dispatch.DISPATCH_OUTPUTS),
    }
    source, inputs, outputs = doors[family]
    return MLXKernelSurface(name=family, source=source(),
                            input_names=tuple(inputs), output_names=tuple(outputs))


def spec_name(family: str, spec: SpecializationEvidence) -> str:
    keys = [k for k in ("BITS", "DH", "M", "R", "E") if k in spec.template]
    return f"{family}-" + "-".join(f"{k}{spec.template[k]}" for k in keys)


def validation_calls(family: str, spec: SpecializationEvidence) -> list:
    """The behavioral-validation subset: the gate's own cases.

    wide_qmv validates on its first matrix shape only (the calls are
    shape-major, so the first two are the 2560x2560 pair); shipping every
    4096x4096 artefact through two subprocess pipes per specialization buys
    no additional behavioral signal for its cost. kv and moe validate on
    every gate case.
    """
    if family == wide_qmv.KERNEL_NAME:
        return list(spec.calls[:2])
    return list(spec.calls)


def _make_verdict(family: str, spec: SpecializationEvidence, calls):
    """The per-case agreement criterion between the extracted and live arms."""
    if family == moe_dispatch.ROUTING_NAME:
        def verdict(index, ours, live):
            idx_ok = bool(np.array_equal(ours[0].astype(np.int64),
                                         live[0].astype(np.int64)))
            gate_err = float(np.max(np.abs(
                ours[1].astype(np.float64) - live[1].astype(np.float64))))
            ok = idx_ok and gate_err <= ROUTING_GATE_TOL
            return ok, f"idx exact {idx_ok}, gate err {gate_err:.3e}"
        return verdict

    tol_by_label = {c.label: c.tol for c in spec.cases}

    def verdict(index, ours, live):
        tolerance = tol_by_label[calls[index].label]
        err = float(np.max(np.abs(
            ours[0].astype(np.float64) - live[0].astype(np.float64))))
        return err <= tolerance, f"err {err:.3e} tol {tolerance:.3e}"

    return verdict


def validate_spec(family: str, spec: SpecializationEvidence, runner: MetalRunner):
    """Capture the specialization's translation unit and prove it behaves.

    Returns `(record, compile_options, report, n_validated)`; raises
    `ExtractionError` when the capture itself fails.
    """
    surface = _surface(family)
    calls = validation_calls(family, spec)
    if family == kv_attention.KERNEL_NAME:
        for call in calls:  # the raw-door caller's capacity obligation
            kv_attention.require_capacity(int(call.inputs["k_wq"].shape[1]))

    record = capture_specialization(surface, calls[0])
    kspec = extracted_spec(record, surface,
                           LaunchSpec(grid=("gx", "gy", "gz"),
                                      threadgroup=spec.threadgroup))
    cases = []
    for call in calls:
        extra_inputs, extra_params = metadata_inputs(call, surface)
        cases.append(RunCase(
            inputs={**call.inputs, **extra_inputs},
            params={**extra_params, "gx": call.grid[0], "gy": call.grid[1],
                    "gz": call.grid[2]},
            output_shapes=call.output_shapes,
            label=call.label))
    batch = runner.run(kspec, cases)
    live = run_live(surface, calls)

    # The pack gate's cases carry none of the battery's reassociation-
    # structured modes, so the compile-option alarm is structurally silent
    # here; sensitivity is all-False and recorded as such.
    report = validate_extraction(batch.results, live, [False] * len(calls),
                                 _make_verdict(family, spec, calls),
                                 labels=[c.label for c in calls])
    return record, dict(batch.compile_options), report, len(calls)


# ---------------------------------------------------------------------------
# Certificate assembly
# ---------------------------------------------------------------------------
def clauses_for(family: str) -> tuple:
    """The contract clauses a pack certificate may honestly attest.

    C1 is structural at fp16 activations because the tree-reduced class is
    numerically inseparable (0/280); C5 is the numeric check the gate itself
    is; C4 exists only for the kernels that compute a softmax, and is
    source-read because the gate's inputs never provoke overflow.
    """
    c1 = ClauseAttestation(
        clause="C1",
        statement="every intermediate at binary32 or wider",
        mechanism=STRUCTURAL,
        scope="fp16 activations, tree-reduced accumulators",
        evidence="every accumulator in the kernel source is float (fp32); "
                 "tree-reduced half-precision accumulation is separable on "
                 "0/280 fp16-activation cases, so source attestation is the "
                 "only honest mechanism at this dtype",
    )
    c5 = ClauseAttestation(
        clause="C5",
        statement="the specified formula, not a defensible relative",
        mechanism=NUMERIC,
        evidence="judged against the operator's fp64 shipped reference within "
                 "the shipped tolerance on every case in the protocol block",
    )
    clauses = [c1, c5]
    if family in (kv_attention.KERNEL_NAME, moe_dispatch.ROUTING_NAME):
        clauses.append(ClauseAttestation(
            clause="C4",
            statement="exponential sums shifted by their own maximum",
            mechanism=STRUCTURAL,
            evidence="the kernel subtracts the reduced maximum before exp in "
                     "source; the pack gate's inputs do not provoke overflow, "
                     "so this clause is source-read pending the battery "
                     "extension to pack operators",
        ))
    return tuple(clauses)


def contract_version_for(family: str) -> str:
    if family == wide_qmv.KERNEL_NAME:
        return (f"MLX-affine quant contract, device-joined ensemble, "
                f"K_QUANT={K_QUANT:g} (ADR 0009/0012)")
    if family == kv_attention.KERNEL_NAME:
        return "kv_attention contract (frozen at 8ec7eca; NATIVE_OPS['kv_attention'])"
    return ("moe_dispatch contract (NATIVE_OPS['moe_dispatch'], Qwen3-class "
            "routing over dense experts)")


def tolerance_model_for(family: str) -> dict:
    if family == wide_qmv.KERNEL_NAME:
        return {"anchor": "r_contract (fp64 dequantized reference)",
                "floor": "quant-contract ensemble",
                "K_quant": K_QUANT,
                "ensemble_members": sorted(ENSEMBLE),
                "form": "max(base_tol(dtype), K_QUANT * floor)"}
    if family == moe_dispatch.ROUTING_NAME:
        return {"criterion": "top-2 indices bit-exact against the contract's "
                             "tie-to-lower-index rule; no numeric tolerance",
                "gate_weights": "renormalized top-2 probabilities"}
    operator = ("kv_attention" if family == kv_attention.KERNEL_NAME
                else "moe_dispatch")
    return {"anchor": f"NATIVE_OPS['{operator}'] fp64 reference",
            "floor": "operator ensemble floor",
            "K_quant": K_QUANT,
            "form": "max(base_tol(dtype), K_QUANT * floor)"}


# The quant tolerance's own validity domain (ADR 0014): what the K_QUANT
# floor attests is the ADMISSIBLE class of contract C1-C6, and MLX's stock
# kernels are not everywhere inside it. Every certificate whose tolerance
# model divides by that floor carries this wording, so a reader cannot take
# the attestation as evidence about cells where no admissible device
# held-out exists.
TOLERANCE_VALIDITY = (
    "admissible class only (contract C1-C6): MLX's own batch-1 and batch-16 "
    "quantized-matmul kernels at fp16 activations are OUT of contract "
    "(ADR 0014: half-precision intermediates in affine_qmv/affine_qmm_t), "
    "so those cells carry no admissible device held-out evidence and the "
    "quant-tolerance attestation there is adequate-by-exclusion")


def domain_for(family: str, spec: SpecializationEvidence) -> dict:
    template = spec.template
    if family == wide_qmv.KERNEL_NAME:
        return {
            "tolerance_validity": TOLERANCE_VALIDITY,
            "BITS": template["BITS"], "M": template["M"], "R": template["R"],
            "d_in": "any multiple of 64 (gate evidence at 2560 and 4096)",
            "d_out": "any positive count (out-of-range rows clamp-read, "
                     "stores guarded; gate evidence at 2560 and 4096)",
            "dispatch_boundary": "should_dispatch: M >= 5; below it MLX "
                                 "already reads the weights once and 2-bit "
                                 "measures a loss - a routing boundary, not "
                                 "a correctness bound",
        }
    if family == kv_attention.KERNEL_NAME:
        return {
            "tolerance_validity": TOLERANCE_VALIDITY,
            "BITS": template["BITS"], "DH": template["DH"],
            "T": "1..1024 inclusive - a CORRECTNESS bound, not "
                 "profitability: the T+1 softmax buffer is compile-time "
                 "threadgroup memory (TCAP=1024) and both doors reject "
                 "t > 1024 loudly",
            "batch": "any B >= 1; one threadgroup per (batch row, head), "
                     "rows independent",
            "cache_semantics": "K/V caches (H, T, DH) as MLX-affine group-64 "
                               "artefacts bit-identical to mx.quantize; the "
                               "step's own new_k/new_v enter at full "
                               "precision inside the T+1 softmax",
        }
    if family == moe_dispatch.ROUTING_NAME:
        return {
            "E": template["E"], "top_k": 2,
            "tie_rule": "ties go to the LOWER index, compared on "
                        "probabilities, not logits",
            "d_model": "any multiple of 4 (half4 activation loads; gate "
                       "evidence at 512 and 256)",
        }
    return {
        "tolerance_validity": TOLERANCE_VALIDITY,
        "R": template["R"], "BITS": 4,
        "experts": "MLX-affine 4-bit group-64 expert artefacts",
        "d_model": "any multiple of 64 (gate evidence at 512 and 256)",
        "routing_inputs": "consumes (idx, gate) produced by kv_moe_routing; "
                          "meaningful only alongside a certified routing "
                          "kernel",
    }


def advisory_for(spec: SpecializationEvidence) -> tuple:
    entries = []
    for case in spec.cases:
        entry = {"label": case.label, "output_sha256": case.output_sha256,
                 "err": case.err, "tol": case.tol}
        if case.err is not None and case.tol:
            entry["margin_err_over_tol"] = round(case.err / case.tol, 6)
        if case.aux:
            entry.update(case.aux)
        entries.append(entry)
    return tuple(entries)


def build_certificate(family: str, spec: SpecializationEvidence,
                      gate: GateEvidence, record, compile_options: dict,
                      n_validated: int, harness_commit: str,
                      chip_generation: str) -> Certificate:
    toolchain = {"mlx": record.mlx_version,
                 "python": platform.python_version(),
                 "numpy": np.__version__,
                 "macos": platform.mac_ver()[0]}
    notes = (
        f"evidence source: the {gate.gate} pack gate, not the 16-eval "
        "mutation-scored battery; no fault catalogue has run against this "
        "operator yet (TODOS.md: pack-operator battery extension is the "
        "upgrade path)",
        f"extraction behaviorally validated on {n_validated} of "
        f"{spec.cases_run} gate cases (see validation_calls in the emitter)",
        "no timings recorded: certification may run under contention "
        "(D12.2), so no performance claim is made or implied",
    )
    return Certificate(
        kernel_name=spec_name(family, spec),
        kernel_family=family,
        operator=spec.operator,
        translation_unit=record.source,
        extraction_provenance=EXTRACTED_VALIDATED,
        compile_options=compile_options,
        clauses=clauses_for(family),
        catalogue_fingerprint="none: no mutation-scored fault catalogue "
                              "exists for this operator yet",
        harness_commit=harness_commit,
        contract_version=contract_version_for(family),
        tolerance_model=tolerance_model_for(family),
        policy=gate.policy,
        budget=spec.cases_run,
        cases_run=spec.cases_run,
        cases_passed=spec.cases_passed,
        seed_protocol=gate.seed_protocol,
        chip_generation=chip_generation,
        toolchain=toolchain,
        domain=domain_for(family, spec),
        advisory_cases=advisory_for(spec),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Per-family certification (audit-all-then-write-all inside each family)
# ---------------------------------------------------------------------------
def certify_family(family: str, specs: list, gate: GateEvidence,
                   runner: MetalRunner, harness_commit: str,
                   chip_generation: str) -> tuple[list, list]:
    """All of one kernel's certificates, or none of them with named reasons."""
    reasons = []
    for spec in specs:
        if not spec.ok:
            failed = [c.label for c in spec.cases if not c.passed]
            reasons.append(f"{spec_name(family, spec)}: gate case(s) failed: "
                           f"{failed}")
    failed_checks = [c.label for c in gate.checks if not c.passed]
    if failed_checks:
        reasons.append(f"gate check(s) failed: {failed_checks}")
    if reasons:
        return [], reasons

    validated = []
    for spec in specs:
        name = spec_name(family, spec)
        try:
            record, compile_options, report, n = validate_spec(family, spec, runner)
        except ExtractionError as error:
            reasons.append(f"{name}: extraction capture failed: {error}")
            continue
        if not report.all_matched:
            reasons.append(f"{name}: extraction validation failed: "
                           f"{report.summary()}")
            continue
        validated.append((spec, record, compile_options, n))
    if reasons:
        return [], reasons

    return [build_certificate(family, spec, gate, record, compile_options, n,
                              harness_commit, chip_generation)
            for spec, record, compile_options, n in validated], []


def harness_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=str(Path(__file__).resolve().parents[1]))
    return result.stdout.strip() or "unknown"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def _compact_domain(domain: dict) -> str:
    parts = []
    for key, value in domain.items():
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}: {str(value).split(' - ')[0].split(' (')[0]}")
    return "; ".join(parts)


def write_manifest(certificates: list, refused: dict, directory: Path,
                   commit: str) -> Path:
    lines = [
        "# Pack certificates manifest",
        "",
        f"Generated by `bench/emit_pack_certificates.py` at harness commit {commit}.",
        "One certificate per kernel specialization; each carries its own translation-unit hash.",
        "The provenance tier `extracted-behaviorally-validated` is claimed only where the extraction validation actually passed.",
        "kv_attention 2/3-bit widths have unit-test coverage only and no gate evidence, so they carry no certificates.",
        "No certificate makes a performance claim.",
        "",
        "| Kernel | Specialization | Provenance | TU sha256 (prefix) | Validity domain |",
        "|---|---|---|---|---|",
    ]
    for cert in certificates:
        tu_prefix = cert.byte_bound()["translation_unit_sha256"][:12]
        spec_desc = cert.kernel_name[len(cert.kernel_family) + 1:] or "-"
        lines.append(f"| {cert.kernel_family} | {spec_desc} "
                     f"| {cert.extraction_provenance} | `{tu_prefix}` "
                     f"| {_compact_domain(cert.domain)} |")
    if refused:
        lines += ["", "## Refused", ""]
        for family, reasons in sorted(refused.items()):
            lines.append(f"- `{family}`:")
            for reason in reasons:
                lines.append(f"  - {reason}")
    path = directory / "MANIFEST.md"
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def gate_evidence(runner: MetalRunner) -> list:
    """Run the three pack gates and return their evidence objects."""
    import pack_kv_attention
    import pack_moe_dispatch
    import pack_wide_qmv

    return [pack_wide_qmv.verify(runner),
            pack_moe_dispatch.verify(runner),
            pack_kv_attention.verify(runner)]


def certify(evidences: list, runner: MetalRunner, directory) -> EmitReport:
    """Evidence to written certificates and manifest, refusing per kernel."""
    directory = Path(directory)
    commit = harness_commit()
    device = runner.probe()
    chip = device.name if device is not None else "unknown"

    gate_of = {}
    for evidence in evidences:
        for spec in evidence.specializations:
            gate_of[spec.kernel] = evidence

    certificates, refusals = [], {}
    for family, specs in by_family(evidences).items():
        certs, reasons = certify_family(family, specs, gate_of[family],
                                        runner, commit, chip)
        if reasons:
            refusals[family] = reasons
        else:
            certificates.extend(certs)

    report = emit(certificates, directory)
    for family, reasons in refusals.items():
        report.refused.setdefault(family, []).extend(reasons)
    write_manifest([c for c in certificates
                    if any(p.name == f"{c.kernel_name}.certificate.json"
                           for p in report.written)],
                   report.refused, directory, commit)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(CERT_DIR),
                        help="directory for certificates and the manifest")
    args = parser.parse_args(argv)

    runner = MetalRunner()
    print("pack certificate emission: gates first, then extraction, then emit")
    evidences = gate_evidence(runner)
    report = certify(evidences, runner, args.out)

    print(f"\nwritten: {len(report.written)} certificate(s) -> {args.out}")
    for path in report.written:
        print(f"  {path.name}")
    if report.refused:
        print("refused:")
        for family, reasons in sorted(report.refused.items()):
            for reason in reasons:
                print(f"  {family}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
