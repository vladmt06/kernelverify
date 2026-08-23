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


@requires_metal
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


@requires_metal
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


# ---------------------------------------------------------------------------
# train_attention: causal grouped-query attention, forward and backward
# ---------------------------------------------------------------------------
def attn_inputs(b=1, hkv=2, g=4, t=9, dh=16, dtype=np.float32, seed=11):
    rng = np.random.default_rng(seed)
    return {
        "q": (rng.standard_normal((b, hkv, g, t, dh)) * 0.4).astype(dtype),
        "k": (rng.standard_normal((b, hkv, t, dh)) * 0.9).astype(dtype),
        "v": (rng.standard_normal((b, hkv, t, dh)) * 0.9).astype(dtype),
        "d_out": (rng.standard_normal((b, hkv, g, t, dh)) * 0.7).astype(dtype),
    }


def test_train_attention_control_agrees_with_reference():
    from kernelverify.reference.native_kernels import train_attention

    inputs = attn_inputs()
    ref = NATIVE_OPS["train_attention"].reference(inputs)
    out = train_attention(inputs).astype(np.float64)
    assert np.max(np.abs(out - ref)) < 1e-5


def test_the_causal_mask_is_a_mask_and_not_a_decoration():
    """The fault a training attention kernel must never ship: a position that
    reads the future. Changing a key that position i is not allowed to see
    must leave row i bit-identical."""
    inputs = attn_inputs()
    before = NATIVE_OPS["train_attention"].reference(inputs)
    moved = {name: arr.copy() for name, arr in inputs.items()}
    moved["k"][:, :, -1, :] += 5.0          # the last key, visible only to the last row
    moved["v"][:, :, -1, :] += 5.0
    after = NATIVE_OPS["train_attention"].reference(moved)
    assert np.array_equal(before[..., :-1, :], after[..., :-1, :])
    assert not np.array_equal(before[..., -1, :], after[..., -1, :])


def test_attn_lse_reference_matches_a_direct_log_sum_exp():
    """The row statistic the forward hands its backward, checked against the
    definition rather than against the shifted form that computes it."""
    from kernelverify.schemas.native_ops import _attn_scores, attn_lse_reference

    inputs = attn_inputs()
    scores = _attn_scores(inputs, np.float64)
    with np.errstate(divide="ignore"):
        direct = np.log(np.sum(np.exp(scores), axis=-1))   # safe: fp64, rms ~ 1
    assert np.max(np.abs(attn_lse_reference(inputs) - direct)) < 1e-12


def test_attn_grad_reference_matches_central_differences():
    """The analytic vjp checked against something that knows nothing about it.

    Amendment 19 clause 79 makes this reference the oracle for a rewrite-class
    backward, so it carries the whole gradient claim: there is no stock
    attention backward in MLX to fall back on.
    """
    from kernelverify.schemas.native_ops import attn_grad_reference, attn_reference

    inputs = attn_inputs(b=1, hkv=2, g=2, t=5, dh=4, dtype=np.float64)
    grads = attn_grad_reference(inputs)

    def loss(name, value):
        moved = dict(inputs)
        moved[name] = value
        return float(np.sum(attn_reference(moved) * inputs["d_out"]))

    eps = 1e-6
    for name in ("q", "k", "v"):
        numeric = np.zeros_like(inputs[name])
        walk = np.nditer(inputs[name], flags=["multi_index"])
        while not walk.finished:
            index = walk.multi_index
            plus, minus = inputs[name].copy(), inputs[name].copy()
            plus[index] += eps
            minus[index] -= eps
            numeric[index] = (loss(name, plus) - loss(name, minus)) / (2 * eps)
            walk.iternext()
        analytic = grads["d" + name]
        scale = max(1e-12, float(np.max(np.abs(numeric))))
        assert np.max(np.abs(numeric - analytic)) / scale < 1e-6, name


def test_the_group_sum_in_dk_and_dv_is_not_optional():
    """Four query heads share one key-value head, so that head's gradient is
    the sum of four. A kernel that writes one group member's contribution is
    correct at a group ratio of one and wrong at four, which is why the ratio
    is a battery dimension and why this is pinned here."""
    from kernelverify.schemas.native_ops import attn_grad_reference

    inputs = attn_inputs(g=4)
    full = attn_grad_reference(inputs)
    one_member = attn_grad_reference({**inputs,
                                      "q": inputs["q"][:, :, :1],
                                      "d_out": inputs["d_out"][:, :, :1]})
    for name in ("dk", "dv"):
        assert full[name].shape == inputs["k"].shape
        gap = np.max(np.abs(full[name] - one_member[name]))
        assert gap > 1e-2 * np.max(np.abs(full[name]))


@requires_metal
def test_attn_reference_cross_checked_against_mlx_fused_attention():
    """The fused primitive is the independent implementation: it is Apple's
    code, it takes the grouped heads flat, and it applies the causal mask
    itself from the same string mlx-lm's training path passes."""
    mx = pytest.importorskip("mlx.core")

    inputs = attn_inputs(b=2, hkv=2, g=4, t=17, dh=64)
    b, hkv, g, t, dh = inputs["q"].shape
    ref = NATIVE_OPS["train_attention"].reference(inputs)
    fused = mx.fast.scaled_dot_product_attention(
        mx.array(inputs["q"].reshape(b, hkv * g, t, dh)),
        mx.array(inputs["k"]), mx.array(inputs["v"]),
        scale=1.0 / np.sqrt(dh), mask="causal")
    mx.eval(fused)
    got = np.array(fused).astype(np.float64).reshape(b, hkv, g, t, dh)
    assert np.max(np.abs(got - ref)) < 1e-4


@requires_metal
def test_attn_grad_reference_cross_checked_against_mlx_autograd():
    """MLX's own vjp through a composed attention, which is exactly what stock
    training runs, against the analytic gradients this project ships."""
    mx = pytest.importorskip("mlx.core")
    from kernelverify.schemas.native_ops import attn_grad_reference

    inputs = attn_inputs(b=1, hkv=2, g=4, t=13, dh=32)
    _b, _hkv, _g, t, dh = inputs["q"].shape
    allowed = mx.array(np.tril(np.ones((t, t), dtype=bool)))

    def composed(q, k, v):
        scores = (q @ mx.swapaxes(k[:, :, None], -1, -2)) * (1.0 / np.sqrt(dh))
        scores = mx.where(allowed, scores, mx.array(-float("inf"), scores.dtype))
        return mx.softmax(scores, axis=-1) @ v[:, :, None]

    primals = [mx.array(inputs[name]) for name in ("q", "k", "v")]
    _out, vjps = mx.vjp(composed, primals, [mx.array(inputs["d_out"])])
    mx.eval(vjps)

    analytic = attn_grad_reference(inputs)
    for name, got in zip(("dq", "dk", "dv"), vjps):
        theirs = np.array(got).astype(np.float64)
        assert theirs.shape == analytic[name].shape, name
        scale = max(1e-12, float(np.max(np.abs(analytic[name]))))
        assert np.max(np.abs(theirs - analytic[name])) / scale < 1e-5, name
