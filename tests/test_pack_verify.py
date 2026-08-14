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
