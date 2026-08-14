"""The pack's quantized-KV attention decode, against the shipped contract.

Every verdict here comes from `NATIVE_OPS["kv_attention"]` through the shared
gate; no test states its own tolerance. The clause-shaped tests pin the two
seams a fused decode kernel most plausibly gets wrong: the step's own new
entry (the off-by-one class) and the batch/head indexing over a shared cache.

Skipped wholesale when MLX/Metal is unavailable, matching test_metal_runner.py.
The kv_attention contract itself is no longer guarded: it landed on main in
e2bb208, so its absence is a broken install, not a branch that has not caught
up, and these tests should fail loudly rather than disappear.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal unavailable", allow_module_level=True)

from kernelverify.schemas.native_ops import NATIVE_OPS

from kernelverify.pack.kv_attention import (
    SUPPORTED_BITS,
    TCAP,
    build,
    kernel_spec,
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
    t = k_cache.shape[1]
    k_wq, k_sc, k_bi = quantize_cache(k_cache, bits)
    v_wq, v_sc, v_bi = quantize_cache(v_cache, bits)
    grid, tg = launch_config(b, h)
    out = kernel(
        inputs=[mx.array(q), mx.array(k_wq), mx.array(k_sc), mx.array(k_bi),
                mx.array(v_wq), mx.array(v_sc), mx.array(v_bi),
                mx.array(new_k), mx.array(new_v)],
        t_cached=t,
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


def test_padded_cache_reads_only_the_logical_prefix(kernel):
    """The mlx-lm integration passes the cache's padded buffer whole with the
    logical length as a scalar (never sliced, ruling D12.1). Rows past the
    logical length are filled with garbage here, so any read past t_cached,
    or any stride confusion between heads, fails the oracle."""
    b, h, t, t_pad, dh, bits = 1, 4, 300, 512, 128, 8
    q, kc, vc, nk, nv = _case(b, h, t, dh, "float16", seed=23)
    rng = np.random.default_rng(99)

    def padded(cache):
        wq, sc, bi = quantize_cache(cache, bits)
        junk = rng.standard_normal((h, t_pad - t, dh)).astype(np.float16)
        jwq, jsc, jbi = quantize_cache(junk, bits)
        return (np.concatenate([wq, jwq], axis=1),
                np.concatenate([sc, jsc], axis=1),
                np.concatenate([bi, jbi], axis=1))

    grid, tg = launch_config(b, h)
    out = kernel(
        inputs=[mx.array(q), *(mx.array(a) for a in padded(kc)),
                *(mx.array(a) for a in padded(vc)),
                mx.array(nk), mx.array(nv)],
        t_cached=t,
        output_shapes=[(b, h, dh)], output_dtypes=[mx.float16],
        grid=grid, threadgroup=tg,
        template=[("T", mx.float16), ("BITS", bits), ("DH", dh)])[0]
    mx.eval(out)
    v = verify_output("kv_attention", kv_inputs(q, kc, vc, nk, nv, bits),
                      np.array(out))
    assert v.ok, v


def test_gqa_matches_the_tiled_cache_reference(kernel):
    """Query heads over fewer cache heads (Qwen3's 4:1 shape, scaled down).
    The shipped reference has no GQA form, so the oracle case tiles each
    cache head and new_k/new_v row R times: quantization is per-row and
    deterministic, so the tiled-cache answer IS the GQA answer, and the
    kernel reading the untiled cache must reproduce it."""
    b, h, hkv, t, dh, bits = 1, 8, 2, 128, 64, 8
    r = h // hkv
    q, kc_full, vc_full, nk_full, nv_full = _case(b, h, t, dh, "float16", seed=31)
    kc, vc = kc_full[:hkv], vc_full[:hkv]          # the untiled hkv-head caches
    nk, nv = nk_full[:, :hkv], nv_full[:, :hkv]
    k_wq, k_sc, k_bi = quantize_cache(kc, bits)
    v_wq, v_sc, v_bi = quantize_cache(vc, bits)
    grid, tg = launch_config(b, h)
    out = kernel(
        inputs=[mx.array(q), mx.array(k_wq), mx.array(k_sc), mx.array(k_bi),
                mx.array(v_wq), mx.array(v_sc), mx.array(v_bi),
                mx.array(nk), mx.array(nv)],
        t_cached=t,
        output_shapes=[(b, h, dh)], output_dtypes=[mx.float16],
        grid=grid, threadgroup=tg,
        template=[("T", mx.float16), ("BITS", bits), ("DH", dh)])[0]
    mx.eval(out)
    tiled = lambda a, axis: np.repeat(a, r, axis=axis)
    v = verify_output(
        "kv_attention",
        kv_inputs(q, tiled(kc, 0), tiled(vc, 0), tiled(nk, 1), tiled(nv, 1), bits),
        np.array(out))
    assert v.ok, v


def test_mlx_door_rejects_over_capacity(kernel):
    """t > TCAP must raise before dispatch: the softmax buffer is sized at
    compile time, and the overrun it prevents is silent memory corruption."""
    q, kc, vc, nk, nv = _case(1, 1, TCAP + 1, 64, "float16", seed=7)
    with pytest.raises(ValueError, match="TCAP"):
        _run(kernel, q, kc, vc, nk, nv, 8, "float16")


def test_raw_door_poisons_over_capacity_output():
    """The raw door's t_cached is a runtime scalar binding, so the rejection
    lives in the kernel body: an over-capacity launch must come back all-NaN,
    never as plausible numbers computed over a corrupted buffer."""
    from kernelverify.runners import MetalRunner, RunCase, specialize

    b, h, t, dh, bits = 1, 1, TCAP + 1, 64, 8
    q, kc, vc, nk, nv = _case(b, h, t, dh, "float16", seed=7)
    k_wq, k_sc, k_bi = quantize_cache(kc, bits)
    v_wq, v_sc, v_bi = quantize_cache(vc, bits)
    spec = specialize(kernel_spec(), {"T": "half", "BITS": bits, "DH": dh})
    case = RunCase(
        inputs={"q": q, "k_wq": k_wq, "k_scales": k_sc, "k_biases": k_bi,
                "v_wq": v_wq, "v_scales": v_sc, "v_biases": v_bi,
                "new_k": nk, "new_v": nv},
        params={"b_rows": b, "n_heads": h, "t_cached": t,
                "n_kv_heads": h, "t_stride": t},
        output_shapes=[((b, h, dh), "float16")],
        label=f"T={t} over capacity")
    [[result]] = MetalRunner().run_candidate([(spec, [case])])
    assert result.ok, result
    assert np.all(np.isnan(result.outputs[0]))
