"""How far apart do legitimate quantized implementations sit on this device?

Phase 0 calibrates K from an ensemble of legitimate implementations of the same
quantization contract, on CPU numpy. The plan pre-registers that the CPU value
is provisional and must be re-derived with device members, because a correct
Metal kernel can legitimately exceed a CPU-calibrated tolerance.

This measures the device-side half of that spread using implementations that
already exist and are all correct by construction: MLX's quantized_matmul on
GPU and on CPU, and dequantize-then-matmul on both. The contract reference is
fp64: exact s*q + b from the stored scales and biases, then an fp64 matmul.

It does not derive K. It supplies the number Phase 0's day-2 re-derivation and
the pack's tolerance both need.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

BITS, GROUP = 4, 64


def unpack(wq: np.ndarray, bits: int, cols: int) -> np.ndarray:
    """Little-endian contiguous bit stream back to integer codes."""
    byts = np.ascontiguousarray(wq).view(np.uint8)
    rows = byts.shape[0]
    stream = np.zeros((rows, byts.shape[1] // 8 + 2), dtype=np.uint64)
    packed = byts.view(np.uint8)
    for b in range(packed.shape[1]):
        stream[:, b // 8] |= packed[:, b].astype(np.uint64) << np.uint64(8 * (b % 8))
    out = np.zeros((rows, cols), dtype=np.uint32)
    mask = (1 << bits) - 1
    for i in range(cols):
        word, off = divmod(i * bits, 64)
        val = stream[:, word] >> np.uint64(off)
        if off + bits > 64:
            val = val | (stream[:, word + 1] << np.uint64(64 - off))
        out[:, i] = (val & np.uint64(mask)).astype(np.uint32)
    return out


def contract_reference(wq, sc, bi, x, out_dim, in_dim):
    """fp64 truth: the intended quantized weights, exactly, times x exactly."""
    q = unpack(np.array(wq), BITS, in_dim).astype(np.float64)
    s = np.array(sc).astype(np.float64).repeat(GROUP, axis=1)
    b = np.array(bi).astype(np.float64).repeat(GROUP, axis=1)
    w = s * q + b
    return np.array(x).astype(np.float64) @ w.T


def main():
    print("mlx", mx.__version__)
    rng = np.random.default_rng(0)

    for label, out_dim, in_dim, wdtype in (
        ("4096x4096 fp16", 4096, 4096, mx.float16),
        ("4096x4096 fp32", 4096, 4096, mx.float32),
        ("2048x2048 fp16", 2048, 2048, mx.float16),
    ):
        w = mx.array(rng.standard_normal((out_dim, in_dim))).astype(wdtype)
        x = mx.array(rng.standard_normal((1, in_dim))).astype(wdtype)
        wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=BITS)
        mx.eval(wq, sc, bi, x)

        ref = contract_reference(wq, sc, bi, x, out_dim, in_dim)
        scale = float(np.max(np.abs(ref)))

        impls = {}
        for stream, tag in ((mx.gpu, "gpu"), (mx.cpu, "cpu")):
            impls[f"quantized_matmul {tag}"] = np.array(mx.quantized_matmul(
                x, wq, sc, bi, transpose=True, group_size=GROUP, bits=BITS,
                stream=stream)).astype(np.float64)
            deq = mx.dequantize(wq, sc, bi, group_size=GROUP, bits=BITS,
                                stream=stream)
            impls[f"dequantize @ matmul {tag}"] = np.array(
                mx.matmul(x, deq.T, stream=stream)).astype(np.float64)

        print(f"\n== {label} (output scale {scale:.2f}) ==")
        errs = {}
        for name, got in impls.items():
            e = float(np.max(np.abs(got - ref)))
            errs[name] = e
            print(f"  {name:28} max|err vs contract fp64| = {e:.4e}  "
                  f"({e/scale:.2e} relative)")
        floor = max(errs.values())
        spread = max(errs.values()) / max(min(errs.values()), 1e-30)
        print(f"  ensemble floor over these four = {floor:.4e}")
        print(f"  worst/best ratio               = {spread:.1f}x")
        print(f"  pairwise max disagreement      = "
              f"{max(float(np.max(np.abs(a - b))) for a in impls.values() for b in impls.values()):.4e}")


if __name__ == "__main__":
    main()
