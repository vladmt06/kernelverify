"""The pack's wide-tile quantized matvec: correct at every tile width it claims.

The kernel's whole reason to exist is holding M input vectors in one weight
pass, so the properties worth pinning are the ones that break when the tile
widens: agreement with the contract across M, the R switch at the register
wall, a d_out that does not divide R, and the artefact layout it assumes.

Skipped wholesale when MLX/Metal is unavailable so the suite stays green on
non-Mac CI, matching tests/test_metal_runner.py.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal unavailable", allow_module_level=True)

from kernelverify.pack.verify import (
    judge,
    qmv_inputs,
    reference_and_tolerance,
    verify_output,
)
from kernelverify.pack.wide_qmv import (
    MAX_PROFITABLE_M,
    MIN_PROFITABLE_M,
    SUPPORTED_BITS,
    build,
    launch_config,
    pack_codes,
    rows_per_simdgroup,
    should_dispatch,
)
from kernelverify.schemas.quant_contract import QuantContract, canonical_quantize


@pytest.fixture(scope="module")
def kernel():
    return build(mx)


def _artefact(d_out, d_in, seed=7, bits=4):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.02).astype(np.float16)
    return w, canonical_quantize(w, QuantContract(bits=bits, group_size=64))


def _run(kernel, x, art, d_out, bits=4):
    m = x.shape[0]
    grid, threadgroup, r = launch_config(d_out, m)
    out = kernel(
        inputs=[mx.array(x), mx.array(pack_codes(art.q, bits)),
                mx.array(art.scales), mx.array(art.biases)],
        output_shapes=[(m, d_out)], output_dtypes=[mx.float16],
        grid=grid, threadgroup=threadgroup,
        template=[("T", mx.float16), ("M", m), ("R", r), ("BITS", bits)])[0]
    mx.eval(out)
    return np.array(out)


@pytest.mark.parametrize("m", [1, 2, 4, 5, 6, 7, 8, 10, 11])
def test_agrees_with_contract_at_every_tile_width(kernel, m):
    """One weight pass must give the same answer as the contract at any M."""
    d_out, d_in = 256, 512
    w, art = _artefact(d_out, d_in)
    x = np.random.default_rng(m).standard_normal((m, d_in)).astype(np.float16)
    v = verify_output("quantized_matmul", qmv_inputs(x, w, 4),
                      _run(kernel, x, art, d_out))
    assert v.ok, v


def test_rows_per_simdgroup_drops_at_the_register_wall():
    """R = 8 and M*R past ~40 spilled when measured; the switch is at M = 10."""
    assert [rows_per_simdgroup(m) for m in (1, 8, 10)] == [4, 4, 4]
    assert rows_per_simdgroup(11) == 2


def test_routes_to_mlx_outside_the_measured_win_zone():
    """The win zone is two-sided. Below M = 5 MLX already reads the weights
    once and this kernel measured a loss at 2 bits (0.81-0.89x at M = 1, 2).
    At M >= 12 MLX stops tiling by fives and switches kernels entirely, so the
    extra-pass defect this kernel fixes no longer exists there. The bounds are
    pinned as literals because the values ARE the ruled decision (D3.1: 5..11
    default until priced at the E2E shapes); repricing moves them on purpose,
    through this test."""
    assert (MIN_PROFITABLE_M, MAX_PROFITABLE_M) == (5, 11)
    assert not should_dispatch(4)    # below: MLX already reads weights once
    assert should_dispatch(5)        # first tile width with a second pass to win
    assert should_dispatch(11)       # last width MLX routes to qmv_wide
    assert not should_dispatch(12)   # MLX switches kernels here
    # The whole batch range the block serves (B = 1..16, ruling D1).
    assert [m for m in range(1, 17) if should_dispatch(m)] == list(range(5, 12))


def test_handles_d_out_not_divisible_by_r(kernel):
    """Out-of-range rows read clamped and must never be stored."""
    d_out, d_in = 254, 512  # 254 % 4 == 2
    w, art = _artefact(d_out, d_in)
    x = np.random.default_rng(3).standard_normal((6, d_in)).astype(np.float16)
    got = _run(kernel, x, art, d_out)
    assert got.shape == (6, d_out)
    v = verify_output("quantized_matmul", qmv_inputs(x, w, 4), got)
    assert v.ok, v


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_artefact_layout_matches_mlx(kernel, bits):
    """The kernel reads MLX's own packing, so the two artefacts must be equal."""
    w, art = _artefact(128, 256, bits=bits)
    wq, scales, biases = mx.quantize(mx.array(w), group_size=64, bits=bits)
    mx.eval(wq, scales, biases)
    assert np.array_equal(pack_codes(art.q, bits), np.array(wq))
    assert np.array_equal(art.scales, np.array(scales))
    assert np.array_equal(art.biases, np.array(biases))


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
@pytest.mark.parametrize("m", [1, 6, 8])
def test_agrees_with_contract_at_every_bit_width(kernel, bits, m):
    """Sub-4-bit codes straddle words; the block striding must still be exact."""
    d_out, d_in = 256, 512
    w, art = _artefact(d_out, d_in, seed=bits, bits=bits)
    x = np.random.default_rng(m).standard_normal((m, d_in)).astype(np.float16)
    got = _run(kernel, x, art, d_out, bits=bits)
    v = verify_output("quantized_matmul", qmv_inputs(x, w, bits), got)
    assert v.ok, v


def test_gate_covers_the_e2e_dispatch_shapes_at_3_bit():
    """The gate must price coverage where the serving path dispatches
    (ruling D1): all six distinct Qwen3-4B decode shapes, 3-bit only (D4 cut
    2-bit), at exactly the M values should_dispatch routes to this kernel."""
    import pack_wide_qmv as gate

    assert {(s.d_out, s.d_in) for s in gate.E2E_SHAPES} == {
        (4096, 2560), (1024, 2560), (2560, 4096),
        (9728, 2560), (2560, 9728), (151936, 2560),
    }
    assert gate.E2E_BITS == 3
    assert gate.e2e_verify_m() == [5, 6, 7, 8, 9, 10, 11]


# ---------------------------------------------------------------------------
# D2 evidence retention: fingerprints always, full arrays only for failures.
# The boundary-pricing probe was killed holding every case's input arrays in
# GateEvidence for a whole 30-minute run; the ruling keeps the audit trail
# (which exact bytes ran, tamper-evident) as sha256 per input and keeps the
# arrays themselves only where a failure needs reproducing.
# ---------------------------------------------------------------------------
INPUT_NAMES = {"x", "w_q", "scales", "biases"}


@pytest.fixture()
def reduced_gate(monkeypatch):
    """The real gate, one 256x256 shape at one tile width, so each test pays
    seconds; nothing on the verify path is stubbed."""
    import pack_wide_qmv

    monkeypatch.setattr(pack_wide_qmv, "SHAPES", [(256, 256)])
    monkeypatch.setattr(pack_wide_qmv, "VERIFY_M", [5])
    monkeypatch.setattr(pack_wide_qmv, "SUPPORTED_BITS", (4,))
    monkeypatch.setattr(pack_wide_qmv, "E2E_SHAPES", ())
    return pack_wide_qmv


def test_passing_evidence_keeps_fingerprints_not_arrays(reduced_gate):
    from kernelverify.runners import MetalRunner

    evidence = reduced_gate.verify(MetalRunner())
    assert evidence.ok
    [spec] = evidence.specializations
    assert len(spec.calls) == len(spec.cases) == 2
    for case, call in zip(spec.cases, spec.calls):
        assert set(case.input_sha256) == INPUT_NAMES
        assert all(len(h) == 64 for h in case.input_sha256.values())
        assert call.inputs == {}, "a judged passing case must not retain arrays"


def test_failing_evidence_keeps_the_arrays(reduced_gate, monkeypatch):
    """One case passes, one fails: only the failing case's LiveCall keeps its
    input arrays, and both keep their fingerprints."""
    from types import SimpleNamespace

    from kernelverify.runners import MetalRunner

    real = reduced_gate.judge
    verdicts = iter([True, False])  # unit passes, corpus fails

    def selective(out, ref, tol):
        v = real(out, ref, tol)
        if next(verdicts, True):
            return v
        return SimpleNamespace(ok=False, err=v.err, tol=v.tol)

    monkeypatch.setattr(reduced_gate, "judge", selective)
    evidence = reduced_gate.verify(MetalRunner())
    assert not evidence.ok
    [spec] = evidence.specializations
    passing, failing = spec.cases
    assert passing.passed and not failing.passed
    passing_call, failing_call = spec.calls
    assert passing_call.inputs == {}
    assert set(failing_call.inputs) == INPUT_NAMES, (
        "a failing case must keep the exact bytes that failed")
    assert set(failing.input_sha256) == INPUT_NAMES


def test_the_extraction_pipeline_may_retain_inputs(reduced_gate):
    """The certificate emitter re-dispatches the gate's own calls through the
    extraction capture, so its evidence keeps the arrays on request."""
    from kernelverify.runners import MetalRunner

    evidence = reduced_gate.verify(MetalRunner(), retain_inputs=True)
    assert evidence.ok
    [spec] = evidence.specializations
    for case, call in zip(spec.cases, spec.calls):
        assert set(call.inputs) == INPUT_NAMES
        assert set(case.input_sha256) == INPUT_NAMES


def test_gate_evidence_carries_e2e_cases_with_projection_names(monkeypatch):
    """A reduced gate run over one E2E shape must produce specialization
    evidence labelled with the projection name, so a certificate reader can
    see WHICH dispatch site a case priced."""
    import pack_wide_qmv as gate

    from kernelverify.pack.dispatch_shapes import DispatchShape
    from kernelverify.runners import MetalRunner

    monkeypatch.setattr(gate, "SHAPES", [])
    monkeypatch.setattr(gate, "E2E_SHAPES", (DispatchShape("q_proj", 256, 256),))
    evidence = gate.verify(MetalRunner(), e2e_m=[5])
    assert evidence.ok
    templates = [s.template for s in evidence.specializations]
    assert templates == [{"T": "half", "BITS": 3, "M": 5, "R": 4}]
    labels = [c.label for c in evidence.specializations[0].cases]
    assert labels and all("q_proj" in label for label in labels)


def test_matches_mlx_quantized_matmul_within_contract(kernel):
    """Both are admissible implementations, so they agree within the floor."""
    d_out, d_in, m = 256, 512, 8
    w, art = _artefact(d_out, d_in)
    x = np.random.default_rng(5).standard_normal((m, d_in)).astype(np.float16)
    ref, tol = reference_and_tolerance("quantized_matmul", qmv_inputs(x, w, 4))
    wq, scales, biases = mx.quantize(mx.array(w), group_size=64, bits=4)
    theirs = np.array(mx.quantized_matmul(mx.array(x), wq, scales, biases,
                                          transpose=True, group_size=64, bits=4))
    ours = _run(kernel, x, art, d_out)
    assert judge(theirs, ref, tol).ok, judge(theirs, ref, tol)
    assert judge(ours, ref, tol).ok, judge(ours, ref, tol)
