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


Q_PROJ = (4096, 2560)
LM_HEAD = (151936, 2560)


def test_routes_per_shape_over_the_served_batch_range():
    """Routing is per-shape and per-width, read off the priced table, not off
    one pair of bounds. Five shapes route M = 5..9; lm_head routes one width
    further because at 151936 rows MLX's second weight pass still costs more
    than our single pass at M = 10. The batch range is the block's own
    B = 1..16 (ruling D1)."""
    assert ([m for m in range(1, 17) if should_dispatch(m, 3, *Q_PROJ)]
            == [5, 6, 7, 8, 9])
    assert ([m for m in range(1, 17) if should_dispatch(m, 3, *LM_HEAD)]
            == [5, 6, 7, 8, 9, 10])


def test_the_cells_the_old_uniform_window_got_wrong():
    """The 5..11 default routed two widths that measure LOSS at q_proj, and
    lm_head M=4 measures WIN but stays unrouted under ruling D2."""
    assert not should_dispatch(10, 3, *Q_PROJ)   # LOSS, routed by the default
    assert not should_dispatch(11, 3, *Q_PROJ)   # LOSS, routed by the default
    assert not should_dispatch(4, 3, *LM_HEAD)   # WIN, refused by D2
    assert should_dispatch(10, 3, *LM_HEAD)      # WIN only at this shape


def test_anything_unpriced_routes_nowhere():
    """4-bit routes nowhere until its own pricing run lands (ruling D1), and
    an unrecorded shape has no evidence, so it gets no routing rather than a
    default."""
    assert not should_dispatch(7, 4, *Q_PROJ)
    assert not should_dispatch(7, 2, *Q_PROJ)
    assert not should_dispatch(7, 3, 4096, 2624)


def test_should_dispatch_demands_the_whole_key():
    """A caller written against the old one-argument boundary must fail loudly
    rather than route on a shape and width nobody priced."""
    with pytest.raises(TypeError):
        should_dispatch(7)


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
    2-bit), at exactly the M values should_dispatch routes to EACH of them."""
    import pack_wide_qmv as gate

    assert {(s.d_out, s.d_in) for s in gate.E2E_SHAPES} == {
        (4096, 2560), (1024, 2560), (2560, 4096),
        (9728, 2560), (2560, 9728), (151936, 2560),
    }
    assert gate.E2E_BITS == 3
    assert gate.e2e_verify_m(*LM_HEAD) == [5, 6, 7, 8, 9, 10]
    assert all(gate.e2e_verify_m(s.d_out, s.d_in) == [5, 6, 7, 8, 9]
               for s in gate.E2E_SHAPES if (s.d_out, s.d_in) != LM_HEAD)


def test_gate_coverage_is_exactly_the_routed_cells_per_shape():
    """One list applied to every shape would leave lm_head's widest routed
    cell unverified or verify five cells the pack never dispatches, so the
    coverage is per-shape and must equal the table cell for cell."""
    import pack_wide_qmv as gate

    from kernelverify.pack.routed_windows import window_for

    for s in gate.E2E_SHAPES:
        assert (set(gate.e2e_verify_m(s.d_out, s.d_in))
                == window_for(gate.E2E_BITS, s.d_out, s.d_in))


def test_no_routed_shape_is_missing_from_the_gate():
    """The other direction, and the one that can go wrong silently: the gate's
    shapes come from the MODEL CONFIG and the routing table comes from the
    RECORDING, so a recording that prices a shape outside Qwen3-4B's decode set
    would route it with zero gate coverage and every per-shape check above
    would still pass, because they only walk the shapes the gate already has."""
    import pack_wide_qmv as gate

    from kernelverify.pack.routed_windows import ROUTED_WINDOWS

    gated = {(s.d_out, s.d_in) for s in gate.E2E_SHAPES}
    routed = {(d_out, d_in) for (bits, d_out, d_in) in ROUTED_WINDOWS
              if bits == gate.E2E_BITS}
    assert routed <= gated, f"routed but never verified: {sorted(routed - gated)}"


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


# ---------------------------------------------------------------------------
# The guard-callback seam. The pricing probe's memory checks run between
# verification tile-widths and between timing rounds, and both loops live in
# THIS shared module (the probe's docstring forbids a diverged sampler copy,
# the ADR 0004 two-halves mistake), so the checks enter through a callback:
# `guard(cell)` may refuse by raising, and None means no checks - the
# microbenchmark gate is unchanged.
# ---------------------------------------------------------------------------
def test_verify_calls_the_guard_between_tile_widths(reduced_gate, monkeypatch):
    from kernelverify.runners import MetalRunner

    monkeypatch.setattr(reduced_gate, "VERIFY_M", [5, 6])
    seen = []
    evidence = reduced_gate.verify(MetalRunner(), guard=seen.append)
    assert evidence.ok
    assert sum("M=5" in cell for cell in seen) == 1
    assert sum("M=6" in cell for cell in seen) == 1


def test_a_refusing_guard_stops_verification(reduced_gate):
    from kernelverify.runners import MetalRunner

    class Refused(RuntimeError):
        pass

    def guard(cell):
        raise Refused(cell)

    with pytest.raises(Refused):
        reduced_gate.verify(MetalRunner(), guard=guard)


def test_interleaved_samples_calls_the_guard_between_rounds():
    import pack_wide_qmv

    a = mx.array([1.0])
    b = mx.array([2.0])
    seen = []
    ours, theirs = pack_wide_qmv.interleaved_samples(
        lambda i: a + i, lambda i: b + i, rounds=3, guard=seen.append)
    assert len(ours) == len(theirs) == 3
    assert len(seen) == 3


def test_a_refusing_guard_stops_the_rounds():
    import pack_wide_qmv

    class Refused(RuntimeError):
        pass

    def guard(cell):
        raise Refused(cell)

    with pytest.raises(Refused):
        pack_wide_qmv.interleaved_samples(
            lambda i: mx.array([1.0]), lambda i: mx.array([2.0]),
            rounds=3, guard=guard)


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
