"""Native operators: ground-truth honesty and seam behaviour.

The rule from the eng-review amendments: every native reference is
cross-checked against an independent implementation. quantized_matmul checks
against MLX's own on-device quantized_matmul; moe_dispatch against a
hand-rolled MLX implementation of the same routing contract. Seam tests pin
that every catalogue fault expresses somewhere and the predicted equivalent
stays equivalent.
"""

import numpy as np
import pytest

from conftest import requires_metal
from kernelverify.battery.core import case_space, make_mode_inputs
from kernelverify.mutation.catalogue import CATALOGUE, KERNEL_TO_OP
from kernelverify.reference.native_kernels import moe_dispatch, quantized_matmul
from kernelverify.schemas.native_ops import NATIVE_OPS

RNG = np.random.default_rng(3)


def qmm_inputs(b=2, d_in=256, d_out=32, bits=4, dtype=np.float16):
    x = RNG.standard_normal((b, d_in)).astype(dtype)
    w = (RNG.standard_normal((d_out, d_in)).astype(np.float32) * 0.05).astype(dtype)
    return {"x": x, "w": w, "bits": np.array([bits], dtype=np.int32)}


def moe_inputs(b=4, d=64, e=8, d_out=32, dtype=np.float32):
    return {
        "x": RNG.standard_normal((b, d)).astype(dtype),
        "router": (RNG.standard_normal((e, d)) * 0.05).astype(dtype),
        "experts": (RNG.standard_normal((e, d_out, d)) * 0.05).astype(dtype),
    }


def test_qmm_kernel_agrees_with_contract_reference():
    for bits in (2, 3, 4, 8):
        inputs = qmm_inputs(bits=bits)
        ref = NATIVE_OPS["quantized_matmul"].reference(inputs)
        out = quantized_matmul(inputs).astype(np.float64)
        assert np.max(np.abs(out - ref)) < 5e-2


@pytest.mark.gpu
def test_qmm_reference_cross_checked_against_mlx_device():
    mx = pytest.importorskip("mlx.core")
    inputs = qmm_inputs(bits=4)
    ref = NATIVE_OPS["quantized_matmul"].reference(inputs)
    w_q, s, b = mx.quantize(mx.array(inputs["w"]), group_size=64, bits=4)
    dev = mx.quantized_matmul(mx.array(inputs["x"]), w_q, s, b, transpose=True,
                              group_size=64, bits=4)
    mx.eval(dev)
    assert np.max(np.abs(np.array(dev).astype(np.float64) - ref)) < 5e-2


@requires_metal
def test_moe_reference_cross_checked_against_mlx():
    mx = pytest.importorskip("mlx.core")
    inputs = moe_inputs()
    ref = NATIVE_OPS["moe_dispatch"].reference(inputs)
    x = mx.array(inputs["x"]); router = mx.array(inputs["router"])
    logits = x @ router.T
    probs = mx.softmax(logits, axis=-1)
    order = mx.argsort(-probs, axis=-1)[:, :2]
    weights = mx.take_along_axis(probs, order, axis=-1)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    out = np.zeros_like(np.asarray(ref), dtype=np.float32)
    experts = inputs["experts"]
    for row in range(inputs["x"].shape[0]):
        for slot in range(2):
            idx = int(order[row, slot])
            out[row] += float(weights[row, slot]) * (
                inputs["x"][row].astype(np.float32) @ experts[idx].astype(np.float32).T)
    assert np.max(np.abs(out.astype(np.float64) - ref)) < 1e-4


def test_every_native_seam_expresses_somewhere():
    """Viability, the way the harness defines it: each non-equivalent fault is
    DETECTED by the shipped oracle on at least one sampled battery case."""
    from bench.measure_escape import corpus_oracle_passes

    natives = [m for m in CATALOGUE if KERNEL_TO_OP[m.kernel] in NATIVE_OPS
               and m.params != {"softmax_topk_order": True}
               and m.params != {"tie_high": True}]
    assert len(natives) == 18
    for m in natives:
        op = NATIVE_OPS[KERNEL_TO_OP[m.kernel]]
        faulty = m.build()
        caught = False
        for case in case_space(op.meta)[::11]:
            inputs = make_mode_inputs(op.meta, case.dim_map, case.dtype,
                                      case.seed, case.distribution)
            if op.augment:
                inputs = op.augment(case, inputs)
            ref = op.reference(inputs)
            tol = op.tolerance(case, inputs, ref)
            if not corpus_oracle_passes(faulty(inputs), ref, tol):
                caught = True
                break
        assert caught, f"{m.name} undetected on every sampled case"


def kv_inputs(b=2, h=2, t=64, dh=64, bits=4, dtype=np.float16):
    return {
        "q": (RNG.standard_normal((b, h, dh)).astype(np.float32) * 0.2).astype(dtype),
        "k_cache": (RNG.standard_normal((h, t, dh)).astype(np.float32) * 3).astype(dtype),
        "v_cache": (RNG.standard_normal((h, t, dh)).astype(np.float32) * 3).astype(dtype),
        "new_k": (RNG.standard_normal((b, h, dh)).astype(np.float32) * 0.2).astype(dtype),
        "new_v": (RNG.standard_normal((b, h, dh)).astype(np.float32) * 3).astype(dtype),
        "bits": np.array([bits], dtype=np.int32),
    }


def test_kv_kernel_agrees_with_reference():
    from kernelverify.reference.native_kernels import kv_attention

    for bits in (4, 8):
        inputs = kv_inputs(bits=bits)
        ref = NATIVE_OPS["kv_attention"].reference(inputs)
        out = kv_attention(inputs).astype(np.float64)
        assert np.max(np.abs(out - ref)) < 5e-3


@pytest.mark.gpu
def test_kv_reference_cross_checked_against_mlx():
    mx = pytest.importorskip("mlx.core")
    from kernelverify.reference.native_kernels import _cache_dequant

    inputs = kv_inputs()
    ref = NATIVE_OPS["kv_attention"].reference(inputs)
    bits = int(inputs["bits"][0])
    kd = _cache_dequant(inputs["k_cache"], bits)
    vd = _cache_dequant(inputs["v_cache"], bits)
    # Full K/V per batch row: cached entries plus the row's own new entry.
    b, h, dh = inputs["q"].shape
    outs = []
    for row in range(b):
        k_full = np.concatenate([kd, inputs["new_k"][row].astype(np.float32)[:, None, :]], axis=1)
        v_full = np.concatenate([vd, inputs["new_v"][row].astype(np.float32)[:, None, :]], axis=1)
        q_row = mx.array(inputs["q"][row].astype(np.float32))[None, :, None, :]
        o = mx.fast.scaled_dot_product_attention(
            q_row, mx.array(k_full)[None], mx.array(v_full)[None],
            scale=1.0 / np.sqrt(dh))
        mx.eval(o)
        outs.append(np.array(o)[0, :, 0, :])
    assert np.max(np.abs(np.stack(outs).astype(np.float64) - ref)) < 1e-4


def test_kv_members_are_distinct_and_two_classes():
    from kernelverify.schemas.native_ops import KV_MEMBERS, _kv_member

    inputs = kv_inputs(t=512, dh=128, dtype=np.float32)
    outs = {name: _kv_member(inputs, **kw) for name, kw in KV_MEMBERS.items()}
    names = list(outs)
    for i, a in enumerate(names):
        for b_ in names[i + 1:]:
            assert not np.array_equal(outs[a], outs[b_]), f"{a} == {b_}"
    scores_class = [n for n in names if "scores" in n or n.endswith("pairwise-pairwise")]
    combine_class = [n for n in names if "combine" in n or n.endswith("pairwise-pairwise")]
    assert len(scores_class) >= 2 and len(combine_class) >= 2


def test_kv_tolerance_says_out_loud_that_its_k_is_borrowed():
    """ADR 0008 says each operator family ships "its own calibrated K".
    For kv_attention that is not true: K_QUANT = 4.0 arrived from the
    quantized_matmul device calibration and no harness has ever derived a K
    over KV_MEMBERS, so the docstring has to say so where a reader of the
    tolerance will see it."""
    import inspect

    from kernelverify.schemas import native_ops

    doc = inspect.getdoc(native_ops.kv_tolerance) or ""
    assert "borrowed" in doc.lower()
    assert "KV_MEMBERS" in doc, "the docstring names the ensemble nobody calibrated"


def test_predicted_equivalents_measure_equivalent():
    inputs = moe_inputs()
    base = moe_dispatch(inputs).astype(np.float64)
    same = moe_dispatch(inputs, softmax_topk_order=True).astype(np.float64)
    np.testing.assert_allclose(base, same, atol=1e-5)
    # tie_high: exact logit ties never occur under continuous draws, so the
    # tie-break direction is equivalent-in-practice across the whole battery.
    tie = moe_dispatch(inputs, tie_high=True).astype(np.float64)
    np.testing.assert_allclose(base, tie, atol=0)


def test_controls_pass_across_sampled_battery():
    from bench.measure_escape import corpus_oracle_passes
    from kernelverify.reference.kernels import KERNELS

    for op_name, op in NATIVE_OPS.items():
        for case in case_space(op.meta)[::17]:
            inputs = make_mode_inputs(op.meta, case.dim_map, case.dtype,
                                      case.seed, case.distribution)
            if op.augment:
                inputs = op.augment(case, inputs)
            ref = op.reference(inputs)
            tol = op.tolerance(case, inputs, ref)
            assert corpus_oracle_passes(KERNELS[op_name](inputs), ref, tol), \
                f"{op_name} control fails at {case}"


@requires_metal
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_cache_dequant_matches_mlx_quantize_dequantize_bit_for_bit(bits):
    """The kv reference's cache half, checked against code it shares nothing with.

    The attention half is already cross-checked against
    `mx.fast.scaled_dot_product_attention`. The cache half was checked only
    against this project's own numpy kv_attention, which calls the SAME
    `_cache_dequant` - so a shared misreading of the quantized cache passes on
    both sides. `_cache_dequant` takes the RAW cache and quantizes then
    dequantizes internally, so MLX's own quantize-then-dequantize on the same
    raw cache is the independent statement of both steps.
    """
    mx = pytest.importorskip("mlx.core")
    from kernelverify.reference.native_kernels import _cache_dequant

    rng = np.random.default_rng(3)
    cache = (rng.standard_normal((4, 128, 256)) * 0.05).astype(np.float16)
    ours = _cache_dequant(cache, bits)          # fp32, as the contract anchors
    wq, scales, biases = mx.quantize(mx.array(cache.reshape(-1, 256)),
                                     group_size=64, bits=bits)

    # `mx.dequantize` returns the SCALES' dtype, so with the fp16 scales it
    # hands back an fp16 array and a bit-for-bit comparison against our fp32
    # result would only be measuring that truncation. Upcasting the scales and
    # biases - the same values, in a wider container - makes MLX compute the
    # same quantity at the same precision, which is the comparison worth
    # making: same codes, same group layout, same s*q+b, no shared code.
    exact = np.array(mx.dequantize(wq, scales.astype(mx.float32),
                                   biases.astype(mx.float32),
                                   group_size=64, bits=bits)).reshape(cache.shape)
    assert exact.dtype == np.float32
    assert np.array_equal(ours, exact)

    # And at MLX's own stock output precision the two still agree exactly, so
    # the fp32 result above is the fp16 one with nothing else changed.
    stock = np.array(mx.dequantize(wq, scales, biases, group_size=64,
                                   bits=bits)).reshape(cache.shape)
    assert stock.dtype == np.float16
    assert np.array_equal(ours.astype(np.float16), stock)
