"""Empirically pin the MLX-affine quantization contract, bit for bit.

Phase 0 calibrates K against "the intended quantized weights (the contract)".
The pack's kernels must compute against the same contract. This reconstructs
mx.quantize in numpy from the MSL source (mlx v0.32.0
mlx/backend/metal/kernels/quantized.h, affine_quantize) and checks the
reconstruction is exact across bits, group sizes, dtypes and edge cases.

Nothing here ships; it is the measurement behind the contract spec.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

EPS = np.float32(1e-7)
BITS = [2, 3, 4, 5, 6, 8]
GROUPS = [32, 64, 128]


def round_half_away(x):
    """Metal's round(): half away from zero, not numpy's half-to-even rint."""
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def affine_params(g: np.ndarray, bits: int):
    """scale and bias per group, exactly as affine_quantize derives them.

    Everything is float32 regardless of the weight dtype, which is what the
    kernel does: it loads T into float, reduces in float, and only casts on
    the store.
    """
    n_bins = np.float32(2 ** bits - 1)
    w_min = g.min(axis=-1).astype(np.float32)
    # The kernel seeds w_max at 0, not -inf, so an all-negative group's upper
    # end is 0. w_min is seeded at +max, so it is a true minimum. Asymmetric.
    w_max = np.maximum(g.max(axis=-1).astype(np.float32), np.float32(0.0))

    scale = np.maximum((w_max - w_min) / n_bins, EPS).astype(np.float32)
    side = np.abs(w_min) > np.abs(w_max)
    scale = np.where(side, scale, -scale).astype(np.float32)
    edge = np.where(side, w_min, w_max).astype(np.float32)

    q0 = round_half_away(edge / scale).astype(np.float32)
    at_zero = q0 == 0.0
    scale = np.where(at_zero, scale, edge / q0).astype(np.float32)
    bias = np.where(at_zero, np.float32(0.0), edge).astype(np.float32)
    return scale, bias


def np_quantize_affine(w: np.ndarray, group_size: int, bits: int):
    """Bit-exact numpy port of mx.quantize(mode="affine")."""
    store_dtype = w.dtype
    rows, cols = w.shape
    g = w.reshape(rows, cols // group_size, group_size).astype(np.float32)
    scale, bias = affine_params(g, bits)

    n_bins = np.float32(2 ** bits - 1)
    # Quantization uses the FULL fp32 scale/bias, while dequantization later
    # uses the dtype-rounded stored ones. Clamp is top-only in the kernel.
    q = round_half_away((g - bias[..., None]) / scale[..., None])
    q = np.minimum(q, n_bins).astype(np.uint32)
    return (q.reshape(rows, cols),
            scale.astype(store_dtype),
            bias.astype(store_dtype))


def pack_bits(q: np.ndarray, bits: int) -> np.ndarray:
    """Contiguous little-endian bit stream, low element in the low bits."""
    rows, cols = q.shape
    stream = np.zeros((rows, (cols * bits + 63) // 64 + 1), dtype=np.uint64)
    for i in range(cols):
        word, offset = divmod(i * bits, 64)
        val = q[:, i].astype(np.uint64)
        stream[:, word] |= val << np.uint64(offset)
        if offset + bits > 64:
            stream[:, word + 1] |= val >> np.uint64(64 - offset)
    byts = stream.view(np.uint8)[:, : cols * bits // 8]
    return np.ascontiguousarray(byts).view(np.uint32)


def report(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'DIFF'}] {label}{'  ' + detail if detail else ''}")
    return ok


def main():
    rng = np.random.default_rng(0)
    print("mlx", mx.__version__)
    all_ok = True

    print("\n== bit-exact reconstruction, float32 weights ==")
    for bits in BITS:
        for group in GROUPS:
            cols = 512
            w = rng.standard_normal((3, cols), dtype=np.float32)
            wq, sc, bi = mx.quantize(mx.array(w), group_size=group, bits=bits)
            q_np, s_np, b_np = np_quantize_affine(w, group, bits)
            packed = pack_bits(q_np, bits)
            ok = np.array_equal(np.array(wq), packed)
            s_ok = np.array_equal(np.array(sc), s_np)
            b_ok = np.array_equal(np.array(bi), b_np)
            all_ok &= report(f"bits={bits} group={group}", ok and s_ok and b_ok,
                             f"packed={ok} scales={s_ok} biases={b_ok}")

    print("\n== bit-exact reconstruction, float16 / bfloat16 weights ==")
    for mdt, ndt in ((mx.float16, np.float16),):
        for bits in (4, 8):
            w = rng.standard_normal((4, 256)).astype(ndt)
            wq, sc, bi = mx.quantize(mx.array(w), group_size=64, bits=bits)
            q_np, s_np, b_np = np_quantize_affine(w, 64, bits)
            ok = np.array_equal(np.array(wq), pack_bits(q_np, bits))
            s_ok = np.array_equal(np.array(sc), s_np)
            all_ok &= report(f"{mdt} bits={bits}", ok and s_ok,
                             f"packed={ok} scales={s_ok}")

    print("\n== adversarial groups ==")
    g = 64
    specials = {
        "all negative": -(rng.random(g).astype(np.float32) + 1.0),
        "all positive": rng.random(g).astype(np.float32) + 1.0,
        "constant": np.full(g, 0.37, np.float32),
        "all zero": np.zeros(g, np.float32),
        "one outlier": np.concatenate([[80.0], rng.standard_normal(g - 1)]).astype(np.float32),
        "symmetric tie": np.concatenate([[-1.0, 1.0], rng.standard_normal(g - 2)]).astype(np.float32),
        "tiny range": (np.full(g, 1.0) + rng.random(g) * 1e-8).astype(np.float32),
        "denormal": np.full(g, 1e-40, np.float32),
    }
    for name, row in specials.items():
        w = row.reshape(1, g)
        wq, sc, bi = mx.quantize(mx.array(w), group_size=g, bits=4)
        q_np, s_np, b_np = np_quantize_affine(w, g, 4)
        ok = (np.array_equal(np.array(wq), pack_bits(q_np, 4))
              and np.array_equal(np.array(sc), s_np)
              and np.array_equal(np.array(bi), b_np))
        all_ok &= report(f"{name:14}", ok,
                         f"scale={float(np.array(sc).ravel()[0]):+.6e} "
                         f"bias={float(np.array(bi).ravel()[0]):+.6e}")

    print("\n== cpu stream vs gpu stream ==")
    w = mx.array(rng.standard_normal((8, 512), dtype=np.float32))
    for bits in (4, 8):
        gq, gs, gb = mx.quantize(w, group_size=64, bits=bits, stream=mx.gpu)
        cq, cs, cb = mx.quantize(w, group_size=64, bits=bits, stream=mx.cpu)
        same = (np.array_equal(np.array(gq), np.array(cq))
                and np.array_equal(np.array(gs), np.array(cs))
                and np.array_equal(np.array(gb), np.array(cb)))
        all_ok &= report(f"bits={bits} cpu==gpu", same)

    print("\n== dequantize: does it use the STORED (dtype-rounded) scale? ==")
    for mdt, ndt in ((mx.float32, np.float32), (mx.float16, np.float16)):
        w = mx.array(rng.standard_normal((2, 128)).astype(ndt))
        wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
        deq = np.array(mx.dequantize(wq, sc, bi, group_size=64, bits=4))
        q_np, s_np, b_np = np_quantize_affine(np.array(w), 64, 4)
        manual = (s_np[:, :, None].astype(ndt) * q_np.reshape(2, 2, 64).astype(ndt)
                  + b_np[:, :, None].astype(ndt)).reshape(2, 128)
        err = float(np.max(np.abs(deq.astype(np.float64) - manual.astype(np.float64))))
        all_ok &= report(f"{mdt}: deq == stored_scale*q + stored_bias", err == 0.0,
                         f"max|diff|={err:.3e}")

    print("\nALL EXACT:", all_ok)


if __name__ == "__main__":
    main()
