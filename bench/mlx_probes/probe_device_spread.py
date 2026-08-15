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

import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase0_contract_k import unpack_mlx_q  # noqa: E402

BITS, GROUP = 4, 64


def contract_reference(wq, sc, bi, x, out_dim, in_dim):
    """fp64 truth: the intended quantized weights, exactly, times x exactly."""
    q = unpack_mlx_q(np.array(wq), in_dim, BITS).astype(np.float64)
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
