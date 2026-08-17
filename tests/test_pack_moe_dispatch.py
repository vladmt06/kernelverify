"""The pack's fused MoE decode: the routing contract, exactly.

The dispatch half is arithmetic and fails loudly; the routing half is where a
kernel silently disagrees with the contract, so most of these pin routing.
The contract is the one in kernelverify/reference/native_kernels.moe_dispatch:
softmax over ALL experts, top-2 by probability, ties to the LOWER index,
renormalize the top-2 weights.

Skipped wholesale when MLX/Metal is unavailable, matching test_metal_runner.py.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal unavailable", allow_module_level=True)

from pack_moe_dispatch import make_case, quantize_experts

from kernelverify.pack.moe_dispatch import (
    build_dispatch,
    build_routing,
    dispatch_launch,
    routing_launch,
)
from kernelverify.pack.verify import moe_inputs, verify_output
from kernelverify.reference.native_kernels import _topk_by_prob

# Whole-module: every test here compiles and dispatches the moe kernel,
# so the gpu marker is module-level.
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def kernels():
    return build_routing(mx), build_dispatch(mx)


def _route(kernels, x, router, n_experts):
    routing, _ = kernels
    grid, threadgroup = routing_launch(x.shape[0])
    idx, gate = routing(inputs=[mx.array(x), mx.array(router)],
                        output_shapes=[(x.shape[0], 2)] * 2,
                        output_dtypes=[mx.uint32, mx.float32],
                        grid=grid, threadgroup=threadgroup,
                        template=[("E", n_experts)])
    mx.eval(idx, gate)
    return idx, gate


def _contract_routing(x, router):
    logits = x.astype(np.float32) @ router.astype(np.float32).T
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    return _topk_by_prob(probs, 2, tie_high=False), probs


def test_routing_matches_the_contract(kernels):
    x, router, _ = make_case(8, 512, 16, 256, seed=2)
    idx, _ = _route(kernels, x, router, 16)
    want, _ = _contract_routing(x, router)
    assert np.array_equal(np.array(idx).astype(int), want)


def test_ties_go_to_the_lower_index(kernels):
    """Constant rows tie every logit, which is the only input that shows it."""
    x, router, _ = make_case(4, 256, 8, 128, seed=2, mode="constant_rows")
    idx, _ = _route(kernels, x, router, 8)
    got = np.array(idx).astype(int)
    assert np.array_equal(got, np.tile([0, 1], (4, 1)))


def test_gates_are_renormalized(kernels):
    x, router, _ = make_case(8, 512, 16, 256, seed=2)
    _, gate = _route(kernels, x, router, 16)
    sums = np.array(gate).sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-6)
    want_idx, probs = _contract_routing(x, router)
    want = np.take_along_axis(probs, want_idx, axis=-1)
    want = want / want.sum(-1, keepdims=True)
    assert np.max(np.abs(np.array(gate) - want)) < 1e-5


@pytest.mark.parametrize("n_tokens", [1, 4, 8])
def test_dispatch_agrees_with_the_shipped_reference(kernels, n_tokens):
    _, dispatch = kernels
    d_model, n_experts, d_ffn = 512, 16, 256
    x, router, experts = make_case(n_tokens, d_model, n_experts, d_ffn, seed=n_tokens)
    packed, scales, biases, arts = quantize_experts(experts)

    idx, gate = _route(kernels, x, router, n_experts)
    grid, threadgroup, r = dispatch_launch(d_ffn, n_tokens)
    out = dispatch(inputs=[mx.array(x), idx, gate, mx.array(packed),
                           mx.array(scales), mx.array(biases)],
                   output_shapes=[(n_tokens, d_ffn)], output_dtypes=[mx.float16],
                   grid=grid, threadgroup=threadgroup,
                   template=[("T", mx.float16), ("R", r)])[0]
    mx.eval(out)
    v = verify_output("moe_dispatch", moe_inputs(x, router, arts), np.array(out))
    assert v.ok, v


def test_handles_d_ffn_not_divisible_by_r(kernels):
    _, dispatch = kernels
    d_model, n_experts, d_ffn = 256, 8, 130   # 130 % 4 == 2
    x, router, experts = make_case(4, d_model, n_experts, d_ffn, seed=6)
    packed, scales, biases, arts = quantize_experts(experts)
    idx, gate = _route(kernels, x, router, n_experts)
    grid, threadgroup, r = dispatch_launch(d_ffn, 4)
    out = dispatch(inputs=[mx.array(x), idx, gate, mx.array(packed),
                           mx.array(scales), mx.array(biases)],
                   output_shapes=[(4, d_ffn)], output_dtypes=[mx.float16],
                   grid=grid, threadgroup=threadgroup,
                   template=[("T", mx.float16), ("R", r)])[0]
    mx.eval(out)
    assert np.array(out).shape == (4, d_ffn)
    v = verify_output("moe_dispatch", moe_inputs(x, router, arts), np.array(out))
    assert v.ok, v
