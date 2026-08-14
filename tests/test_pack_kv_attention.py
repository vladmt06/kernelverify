"""The pack's quantized-KV attention decode, against the shipped contract.

Every verdict here comes from `NATIVE_OPS["kv_attention"]` through the shared
gate; no test states its own tolerance. The clause-shaped tests pin the two
seams a fused decode kernel most plausibly gets wrong: the step's own new
entry (the off-by-one class) and the batch/head indexing over a shared cache.

Skipped wholesale when MLX/Metal is unavailable, matching test_metal_runner.py,
and when the kv_attention contract has not landed on this branch yet (it
freezes on branch kv-attention-op, commit 8ec7eca; merge queued).
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal unavailable", allow_module_level=True)

from kernelverify.schemas.native_ops import NATIVE_OPS

if "kv_attention" not in NATIVE_OPS:
    pytest.skip("kv_attention contract not merged yet (branch kv-attention-op)",
                allow_module_level=True)

from kernelverify.pack.kv_attention import (
    SUPPORTED_BITS,
    build,
    launch_config,
    quantize_cache,
)
from kernelverify.pack.verify import kv_inputs, verify_output

MXD = {"float16": mx.float16, "float32": mx.float32}


@pytest.fixture(scope="module")
def kernel():
    return build(mx)


def _peak_rescale(a, target):
    peak = float(np.max(np.abs(a.astype(np.float32))))
    return (a.astype(np.float32) * (target / peak)).astype(a.dtype)


def _case(b, h, t, dh, dtype, seed):
    """Inputs in the battery's own regime: q and new_k at peak 0.6 (the
    contract's QUERY_SCALE), caches at unit scale."""
    rng = np.random.default_rng(seed)
    q = _peak_rescale(rng.standard_normal((b, h, dh)).astype(dtype), 0.6)
    new_k = _peak_rescale(rng.standard_normal((b, h, dh)).astype(dtype), 0.6)
    k_cache = rng.standard_normal((h, t, dh)).astype(dtype)
    v_cache = rng.standard_normal((h, t, dh)).astype(dtype)
    new_v = rng.standard_normal((b, h, dh)).astype(dtype)
    return q, k_cache, v_cache, new_k, new_v


def _run(kernel, q, k_cache, v_cache, new_k, new_v, bits, dtype):
    b, h, dh = q.shape
    k_wq, k_sc, k_bi = quantize_cache(k_cache, bits)
    v_wq, v_sc, v_bi = quantize_cache(v_cache, bits)
    grid, tg = launch_config(b, h)
    out = kernel(
        inputs=[mx.array(q), mx.array(k_wq), mx.array(k_sc), mx.array(k_bi),
                mx.array(v_wq), mx.array(v_sc), mx.array(v_bi),
                mx.array(new_k), mx.array(new_v)],
        output_shapes=[(b, h, dh)], output_dtypes=[MXD[dtype]],
        grid=grid, threadgroup=tg,
        template=[("T", MXD[dtype]), ("BITS", bits), ("DH", dh)])[0]
    mx.eval(out)
    return np.array(out)


@pytest.mark.parametrize("shape", [(1, 8, 512, 128), (4, 2, 64, 64),
                                   (1, 4, 300, 128)])
@pytest.mark.parametrize("bits", [4, 8])
def test_agrees_with_the_shipped_reference(kernel, shape, bits):
    b, h, t, dh = shape
    q, kc, vc, nk, nv = _case(b, h, t, dh, "float16", seed=t + bits)
    out = _run(kernel, q, kc, vc, nk, nv, bits, "float16")
    v = verify_output("kv_attention", kv_inputs(q, kc, vc, nk, nv, bits), out)
    assert v.ok, v


@pytest.mark.parametrize("bits", [4, 8])
def test_float32_dtype(kernel, bits):
    """Both contract dtypes; at fp32 the base tolerance is ~4e-7 * scale, so
    this is the strict test of the fp32-everywhere arithmetic."""
    q, kc, vc, nk, nv = _case(2, 4, 256, 128, "float32", seed=17 + bits)
    out = _run(kernel, q, kc, vc, nk, nv, bits, "float32")
    v = verify_output("kv_attention", kv_inputs(q, kc, vc, nk, nv, bits), out,
                      dtype="float32")
    assert v.ok, v


@pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
def test_every_supported_width(kernel, bits):
    """2 and 3 bits ride the same 8-code block reader; the oracle quantizes
    at whatever width the case names."""
    q, kc, vc, nk, nv = _case(1, 4, 128, 64, "float16", seed=40 + bits)
    out = _run(kernel, q, kc, vc, nk, nv, bits, "float16")
    v = verify_output("kv_attention", kv_inputs(q, kc, vc, nk, nv, bits), out)
    assert v.ok, v


def test_new_entry_dominates_when_it_should(kernel):
    """The off-by-one clause, behaviourally: a new key aligned with the query
    concentrates the softmax on the new entry, so a kernel that dropped it
    from scores or combine could not track the reference here."""
    b, h, t, dh = 2, 4, 128, 64
    q, kc, vc, _, nv = _case(b, h, t, dh, "float16", seed=5)
    new_k = (q.astype(np.float32) * 60.0).astype(np.float16)  # score >> cached
    out = _run(kernel, q, kc, vc, new_k, nv, 8, "float16")
    v = verify_output("kv_attention", kv_inputs(q, kc, vc, new_k, nv, 8), out)
    assert v.ok, v
    # And the output really is pinned to new_v, not to the cached average.
    assert float(np.max(np.abs(out - nv.astype(np.float64)))) < 0.05


def test_batch_rows_are_independent(kernel):
    """The shared cache is read per (batch row, head); each batch row must
    reproduce its own B=1 answer exactly."""
    b, h, t, dh = 4, 2, 96, 64
    q, kc, vc, nk, nv = _case(b, h, t, dh, "float16", seed=11)
    full = _run(kernel, q, kc, vc, nk, nv, 4, "float16")
    for row in range(b):
        one = _run(kernel, q[row:row + 1], kc, vc,
                   nk[row:row + 1], nv[row:row + 1], 4, "float16")
        assert np.array_equal(full[row:row + 1], one)


def test_single_cached_position(kernel):
    """T=1: the smallest cache, where every simdgroup but one is idle in the
    combine and the strided score loop touches two positions total."""
    q, kc, vc, nk, nv = _case(1, 2, 1, 64, "float16", seed=3)
    out = _run(kernel, q, kc, vc, nk, nv, 4, "float16")
    v = verify_output("kv_attention", kv_inputs(q, kc, vc, nk, nv, 4), out)
    assert v.ok, v
