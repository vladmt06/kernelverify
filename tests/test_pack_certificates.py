"""Certificate emission from real pack-gate evidence.

The evidence object is the contract between a gate and the emitter (D7:
never stdout parsing), so these tests drive the real reduced gate, the real
extraction capture and live arm, and the real emitter, end to end. Refusal
tests then break exactly one seam at a time: a capture failure and a
behavioral-validation failure must each refuse that kernel's certificates
with a named reason while other kernels still emit (D10).

The gate is reduced (one 256x256 shape, one tile width, one bit width) so
the suite pays seconds, not the full gate's minutes; nothing about the path
is stubbed in the happy-path test.
"""

import importlib.util
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("pack certificates need the Metal GPU", allow_module_level=True)

import emit_pack_certificates as emitter  # noqa: E402
import pack_wide_qmv  # noqa: E402

from kernelverify.extraction import ExtractionError  # noqa: E402
from kernelverify.extraction.validate import CaseOutcome, ExtractionReport  # noqa: E402
from kernelverify.pack import kv_attention, moe_dispatch  # noqa: E402
from kernelverify.pack.evidence import (  # noqa: E402
    CaseEvidence,
    GateEvidence,
    SpecializationEvidence,
)
from kernelverify.runners import MetalRunner  # noqa: E402

RUNNER = MetalRunner()
if RUNNER.probe() is None:
    pytest.skip("no Metal device", allow_module_level=True)


@pytest.fixture(scope="module")
def qmv_evidence():
    """Real gate evidence from a reduced wide-qmv gate run."""
    patch = pytest.MonkeyPatch()
    patch.setattr(pack_wide_qmv, "SHAPES", [(256, 256)])
    patch.setattr(pack_wide_qmv, "VERIFY_M", [5])
    patch.setattr(pack_wide_qmv, "SUPPORTED_BITS", (4,))
    try:
        yield pack_wide_qmv.verify(RUNNER)
    finally:
        patch.undo()


# ---------------------------------------------------------------------------
# The evidence object itself
# ---------------------------------------------------------------------------
def test_gate_evidence_carries_cases_errors_tolerances_and_labels(qmv_evidence):
    assert qmv_evidence.ok
    [spec] = qmv_evidence.specializations
    assert spec.kernel == "kv_wide_qmv"
    assert spec.template == {"T": "half", "BITS": 4, "M": 5, "R": 4}
    assert spec.cases_run == 2 and spec.cases_passed == 2
    assert len(spec.calls) == len(spec.cases), "calls must align with cases"
    for case in spec.cases:
        assert "4-bit M=5" in case.label
        assert case.err is not None and case.tol is not None
        assert case.err <= case.tol
        assert len(case.output_sha256) == 64
    [check] = qmv_evidence.checks
    assert check.passed and "mx.quantize" in check.label
    assert "NOT the 16-eval" in qmv_evidence.policy


# ---------------------------------------------------------------------------
# Happy path: evidence -> extraction -> certificates -> manifest, all real
# ---------------------------------------------------------------------------
def test_emission_happy_path_from_real_gate_evidence(qmv_evidence, tmp_path):
    report = emitter.certify([qmv_evidence], RUNNER, tmp_path)
    assert report.ok, report.refused

    # Certified inventory matches the evidence, one certificate per
    # specialization (D3).
    n_specs = len(qmv_evidence.specializations)
    assert len(report.written) == n_specs == 1

    [path] = report.written
    text = path.read_text()
    doc = json.loads(text)

    # The protocol block names the REAL pack-gate policy with its TRUE
    # budget, never the 16-eval battery (D10).
    protocol = doc["protocol_bound"]
    assert "pack-gate" in protocol["policy"]
    assert "NOT the 16-eval" in protocol["policy"]
    assert protocol["budget_evaluations_per_operator"] == 2
    assert protocol["cases_run"] == 2 and protocol["all_passed"]

    # Top provenance tier only because behavioral validation actually
    # passed (D2), with one translation-unit hash (D3).
    byte_bound = doc["byte_bound"]
    assert byte_bound["extraction_provenance"] == "extracted-behaviorally-validated"
    assert len(byte_bound["translation_unit_sha256"]) == 64
    assert byte_bound["compile_options"] == {"math_mode": "safe"}

    # The validity domain travels with the certificate (D9).
    domain = doc["validity"]["domain"]
    assert domain["BITS"] == 4 and domain["M"] == 5
    assert "should_dispatch" in domain["dispatch_boundary"]

    # The quant tolerance's own validity domain (ADR 0014): admissible
    # class only, and the excluded MLX cells are named as evidence-free.
    assert "admissible class only" in domain["tolerance_validity"]
    assert "ADR 0014" in domain["tolerance_validity"]
    assert "adequate-by-exclusion" in domain["tolerance_validity"]

    # Advisory block: fingerprints and margins, labeled not reproducible (D8).
    assert "NOT reproducible" in doc["advisory"]["_meaning"]
    assert len(doc["advisory"]["cases"]) == 2
    for entry in doc["advisory"]["cases"]:
        assert len(entry["output_sha256"]) == 64
        assert 0 <= entry["margin_err_over_tol"] <= 1

    # No performance claims and no timings anywhere (D12.2).
    assert doc["performance"] == []
    assert '"timing"' not in text and "gpu_seconds" not in text

    # The manifest indexes the certificate (D5).
    manifest = (tmp_path / "MANIFEST.md").read_text()
    assert "| kv_wide_qmv | BITS4-M5-R4 |" in manifest
    assert byte_bound["translation_unit_sha256"][:12] in manifest
    assert "extracted-behaviorally-validated" in manifest


# ---------------------------------------------------------------------------
# Refusal is per kernel, with named reasons
# ---------------------------------------------------------------------------
def _synthetic_routing_evidence() -> GateEvidence:
    evidence = GateEvidence(gate="pack_moe_dispatch", policy="pack-gate stub",
                            seed_protocol="stub")
    spec = SpecializationEvidence(kernel=moe_dispatch.ROUTING_NAME,
                                  operator="moe_dispatch",
                                  template={"E": 4}, threadgroup=(32, 1, 1))
    spec.cases.append(CaseEvidence(label="routing stub", passed=True))
    evidence.specializations.append(spec)
    return evidence


def test_one_kernels_capture_failure_leaves_the_other_kernel_emitting(
        qmv_evidence, tmp_path, monkeypatch):
    real_validate = emitter.validate_spec

    def selective(family, spec, runner):
        if family == moe_dispatch.ROUTING_NAME:
            raise ExtractionError("forced capture failure")
        return real_validate(family, spec, runner)

    monkeypatch.setattr(emitter, "validate_spec", selective)
    report = emitter.certify([qmv_evidence, _synthetic_routing_evidence()],
                             RUNNER, tmp_path)

    assert list(report.refused) == [moe_dispatch.ROUTING_NAME]
    [reason] = report.refused[moe_dispatch.ROUTING_NAME]
    assert "extraction capture failed" in reason and "forced" in reason
    assert [p.name for p in report.written] == \
        ["kv_wide_qmv-BITS4-M5-R4.certificate.json"]

    manifest = (tmp_path / "MANIFEST.md").read_text()
    assert "## Refused" in manifest and "forced capture failure" in manifest


def test_a_behavioral_validation_failure_refuses_with_a_named_reason(
        qmv_evidence, tmp_path, monkeypatch):
    failing = ExtractionReport(outcomes=[
        CaseOutcome("4-bit M=5 unit 256x256", False, False, "forced mismatch")])

    def fake_validate(family, spec, runner):
        return None, {"math_mode": "safe"}, failing, 2

    monkeypatch.setattr(emitter, "validate_spec", fake_validate)
    report = emitter.certify([qmv_evidence], RUNNER, tmp_path)

    assert not report.ok
    [reason] = report.refused["kv_wide_qmv"]
    assert "extraction validation failed" in reason
    assert "forced mismatch" in reason
    assert not list(tmp_path.glob("*.certificate.json")), (
        "a kernel whose validation failed must emit nothing")


def test_a_failed_gate_case_refuses_before_any_extraction_runs(tmp_path,
                                                               monkeypatch):
    def exploding_validate(family, spec, runner):  # must never be reached
        raise AssertionError("extraction ran for a kernel whose gate is red")

    monkeypatch.setattr(emitter, "validate_spec", exploding_validate)
    evidence = _synthetic_routing_evidence()
    evidence.specializations[0].cases.append(
        CaseEvidence(label="routing red", passed=False, detail="forced"))
    report = emitter.certify([evidence], RUNNER, tmp_path)
    [reason] = report.refused[moe_dispatch.ROUTING_NAME]
    assert "gate case(s) failed" in reason and "routing red" in reason


# ---------------------------------------------------------------------------
# The kv doors reject t over capacity, loudly (D9)
# ---------------------------------------------------------------------------
def test_require_capacity_accepts_the_bound_and_rejects_beyond_it():
    kv_attention.require_capacity(kv_attention.TCAP)
    with pytest.raises(ValueError, match="capacity bound TCAP=1024"):
        kv_attention.require_capacity(kv_attention.TCAP + 1)


def test_the_mlx_door_rejects_a_cache_beyond_capacity_before_dispatch():
    kernel = kv_attention.build(mx)
    t_over = kv_attention.TCAP + 1
    inputs = [mx.zeros((1, 1, 64), dtype=mx.float16),          # q
              mx.zeros((1, t_over, 8), dtype=mx.uint32),       # k_wq: T over cap
              mx.zeros((1, t_over, 1), dtype=mx.float16),      # k_scales
              mx.zeros((1, t_over, 1), dtype=mx.float16),      # k_biases
              mx.zeros((1, t_over, 8), dtype=mx.uint32),       # v_wq
              mx.zeros((1, t_over, 1), dtype=mx.float16),      # v_scales
              mx.zeros((1, t_over, 1), dtype=mx.float16),      # v_biases
              mx.zeros((1, 1, 64), dtype=mx.float16),          # new_k
              mx.zeros((1, 1, 64), dtype=mx.float16)]          # new_v
    grid, threadgroup = kv_attention.launch_config(1, 1)
    # The merged door takes the logical length explicitly; the rejection
    # keys on it rather than on the padded buffer's physical extent.
    with pytest.raises(ValueError, match="exceeds TCAP"):
        kernel(inputs=inputs, t_cached=t_over, output_shapes=[(1, 1, 64)],
               output_dtypes=[mx.float16], grid=grid, threadgroup=threadgroup,
               template=[("T", mx.float16), ("BITS", 4), ("DH", 64)])
