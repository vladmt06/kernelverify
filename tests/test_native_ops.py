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


def test_qmm_reference_cross_checked_against_mlx_device():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal unavailable")
    inputs = qmm_inputs(bits=4)
    ref = NATIVE_OPS["quantized_matmul"].reference(inputs)
    w_q, s, b = mx.quantize(mx.array(inputs["w"]), group_size=64, bits=4)
    dev = mx.quantized_matmul(mx.array(inputs["x"]), w_q, s, b, transpose=True,
                              group_size=64, bits=4)
    mx.eval(dev)
    assert np.max(np.abs(np.array(dev).astype(np.float64) - ref)) < 5e-2


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
    assert len(natives) == 11
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
