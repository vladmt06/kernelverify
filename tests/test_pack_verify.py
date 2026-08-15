"""The shared pack gate, pinned without Metal.

kernelverify/pack/verify.py is pure numpy, so these run on non-Mac CI. What
is worth pinning is the gate's discrimination, in both directions: every
member of the legitimate ensemble must pass it, and every artefact fault
and shipping routing violation must fail it. The formula itself is NOT
pinned here - it lives in NATIVE_OPS and must stay free to evolve; that is
the whole point of routing the pack through it.
"""

import numpy as np
import pytest

from kernelverify.pack.verify import (
    judge,
    moe_inputs,
    qmv_inputs,
    reference_and_tolerance,
    verify_output,
)
from kernelverify.reference.native_kernels import moe_dispatch
from kernelverify.schemas.quant_contract import (
    ENSEMBLE,
    FAULTS,
    QuantContract,
    canonical_quantize,
    r_contract,
)


def _qmv_case(bits=4, d_out=64, d_in=256, m=4, seed=11):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.02).astype(np.float16)
    x = (rng.standard_normal((m, d_in)).astype(np.float32) * 0.5).astype(np.float16)
    return x, w, canonical_quantize(w, QuantContract(bits=bits, group_size=64))


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("member", sorted(ENSEMBLE))
def test_every_legitimate_member_passes(bits, member):
    x, w, art = _qmv_case(bits=bits)
    got = ENSEMBLE[member](x, art)
    v = verify_output("quantized_matmul", qmv_inputs(x, w, bits), got)
    assert v.ok, v


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_every_artefact_fault_fails(fault):
    """A kernel computing the WRONG artefact's answer exactly must still
    fail: r_contract of the faulted artefact is that kernel at fp64."""
    x, w, art = _qmv_case(bits=4)
    got = r_contract(x, FAULTS[fault](art))
    v = verify_output("quantized_matmul", qmv_inputs(x, w, 4), got)
    assert not v.ok, f"{fault} escaped: {v}"


def test_judge_accepts_the_spike_output_shape():
    """The dequant-GEMV spike judges a flat (d_out,) output against the
    (1, d_out) reference; the broadcast must stay legal so the runner
    lane's migration (W4) can rely on it."""
    x, w, art = _qmv_case(m=1)
    ref, tol = reference_and_tolerance("quantized_matmul", qmv_inputs(x, w, 4))
    flat = ENSEMBLE["dequant-pairwise"](x, art)[0]
    assert flat.shape == ref.shape[1:]
    assert judge(flat, ref, tol).ok


def _moe_case(n_tokens=4, d_model=128, n_experts=4, d_ffn=32, seed=3):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n_tokens, d_model)).astype(np.float32) * 0.5).astype(np.float16)
    router = (rng.standard_normal((n_experts, d_model)).astype(np.float32) * 0.05).astype(np.float16)
    experts = (rng.standard_normal((n_experts, d_ffn, d_model)).astype(np.float32)
               * 0.02).astype(np.float16)
    arts = [canonical_quantize(e, QuantContract(bits=4, group_size=64)) for e in experts]
    return x, router, arts


def test_the_shipped_moe_reference_kernel_passes():
    x, router, arts = _moe_case()
    inputs = moe_inputs(x, router, arts)
    v = verify_output("moe_dispatch", inputs, moe_dispatch(inputs))
    assert v.ok, v


@pytest.mark.parametrize("seam", [{"renormalize": False}, {"expert_offset": 1}])
def test_shipping_routing_violations_fail(seam):
    x, router, arts = _moe_case()
    inputs = moe_inputs(x, router, arts)
    v = verify_output("moe_dispatch", inputs, moe_dispatch(inputs, **seam))
    assert not v.ok, f"{seam} escaped: {v}"


# ---------------------------------------------------------------------------
# The serial ensemble members' transient must stay bounded. The 2026-08-15
# boundary-pricing instrumentation measured the unchunked members' full
# (m, d_out, d_in) product plus its running-sum copy ratcheting macOS
# malloc's dirty pages monotonically through the gate's tile-width sweep
# (22 GB by 9728x2560 M=16; ~25 GB per member at the lm_head shape) - the
# same allocation class that Jetsam-killed the probe. Row chunking is
# bit-identical at every chunk size because the elementwise product and the
# accumulation run along d_in and never cross a row boundary, and no BLAS
# is involved - the serving harness's eval_serial_chunked proved exactly
# this equivalence, so the tolerance floor, and with it every recorded
# verdict and ADR table, is unchanged.
# ---------------------------------------------------------------------------
def _direct_serial(x, art, reverse):
    """The unchunked member formula, inline, as the known-good spec."""
    from kernelverify.schemas.quant_contract import dequantize

    w = dequantize(art, np.float32)
    prod = x.astype(np.float32)[:, None, :] * w[None, :, :]
    if reverse:
        prod = prod[:, :, ::-1]
    return np.add.accumulate(prod, axis=2)[:, :, -1].astype(x.dtype)


@pytest.mark.parametrize("reverse,name", [(False, "dequant-serial"),
                                          (True, "dequant-reversed")])
@pytest.mark.parametrize("m", [1, 5])
def test_serial_members_bit_identical_even_at_one_row_chunks(
        monkeypatch, reverse, name, m):
    import kernelverify.schemas.quant_contract as qc

    monkeypatch.setattr(qc, "SERIAL_CHUNK_BYTES", 1)  # force chunk_rows = 1
    x, w, art = _qmv_case(bits=3, d_out=96, d_in=256, m=m)
    assert np.array_equal(ENSEMBLE[name](x, art), _direct_serial(x, art, reverse))


def test_serial_members_bound_their_transient():
    """In a fresh process (so no earlier test's dirty arena can absorb the
    allocation), one serial-member call at (8, 2048x4096) must not grow the
    footprint by the full product's ~0.5 GB."""
    import subprocess
    import sys
    from pathlib import Path

    bench = Path(__file__).resolve().parents[1] / "bench"
    script = (
        "import sys\n"
        f"sys.path[:0] = [{str(bench)!r}]\n"
        "import numpy as np\n"
        "from memory_guard import phys_footprint_gb\n"
        "from kernelverify.schemas.quant_contract import (ENSEMBLE,\n"
        "    QuantContract, canonical_quantize)\n"
        "rng = np.random.default_rng(7)\n"
        "w = (rng.standard_normal((2048, 4096)).astype(np.float32)\n"
        "     * 0.02).astype(np.float16)\n"
        "art = canonical_quantize(w, QuantContract(bits=3, group_size=64))\n"
        "x = rng.standard_normal((8, 4096)).astype(np.float16)\n"
        "before = phys_footprint_gb()[0]\n"
        "ENSEMBLE['dequant-serial'](x, art)\n"
        "print(phys_footprint_gb()[0] - before)\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    grown_gb = float(out.stdout.strip())
    assert grown_gb < 0.3, (
        f"serial member grew the footprint by {grown_gb:.2f} GB; its "
        f"transient must stay bounded by SERIAL_CHUNK_BYTES")


def test_the_artefact_is_derived_once_per_weight_matrix(monkeypatch):
    """reference and tolerance each derive the artefact from the RAW weights
    (the anchoring property), and the gate judges up to 32 cases per
    (shape, bits) against the SAME matrix - without reuse that is 64
    canonical quantizations of a 389M-element matrix per lm_head group, the
    per-case transient level the 2026-08-15 instrumentation measured at
    14-22 GB. The memo is keyed by weight content, so the anchoring property
    survives: the artefact still comes from exactly the bytes passed in."""
    import kernelverify.schemas.native_ops as native_ops

    calls = []
    real = native_ops.canonical_quantize

    def counting(w, contract):
        calls.append(w.shape)
        return real(w, contract)

    monkeypatch.setattr(native_ops, "canonical_quantize", counting)
    x, w, art = _qmv_case(bits=3)
    first = reference_and_tolerance("quantized_matmul", qmv_inputs(x, w, 3))
    x2 = x + np.float16(0.25)
    second = reference_and_tolerance("quantized_matmul", qmv_inputs(x2, w, 3))
    assert len(calls) == 1, (
        f"the same weight matrix was quantized {len(calls)} times")

    # A different matrix must miss the memo and re-derive.
    w2 = np.ascontiguousarray(w[:, ::-1])
    reference_and_tolerance("quantized_matmul", qmv_inputs(x, w2, 3))
    assert len(calls) == 2

    # And the memo must never change what is computed.
    del calls[:]
    monkeypatch.undo()
    fresh_ref, fresh_tol = reference_and_tolerance(
        "quantized_matmul", qmv_inputs(x, w, 3))
    assert np.array_equal(first[0], fresh_ref) and first[1] == fresh_tol
