"""T5 feasibility spike: one competitive fused dequant-GEMV on Metal.

The D7 decision experiment (eng review). The full loop in miniature:
a candidate MSL kernel is VERIFIED through the crash-isolated runner against
the Phase 0 quantization contract (r_contract anchor, ensemble floor, K=3),
and only then TIMED against MLX's own quantized_matmul on the decode shapes
of the model actually benchmarked (Qwen3-4B: 2560 hidden, 9728 ffn).

Context that frames the result: on this M3 Pro, llama.cpp decodes at 92% of
the achieved-bandwidth roofline (little to win) while MLX decodes at ~69%
(a real gap, and MLX is the launch channel). The spike asks whether a simple
verified kernel through mx.fast.metal_kernel can close any of MLX's gap.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from kernelverify.runners.runner import DeviceCase, KernelSpec, MetalRunner  # noqa: E402
from kernelverify.schemas.quant_contract import (  # noqa: E402
    ENSEMBLE,
    QuantContract,
    canonical_quantize,
    r_contract,
)

CONTRACT = QuantContract(bits=4, group_size=64)
K = 3.0  # Phase 0's calibrated safety factor
SHAPES = [(9728, 2560), (2560, 9728), (2560, 2560)]  # (d_out, d_in), Qwen3-4B decode
REPEATS = 30

# One simdgroup per output row. Each lane strides over packed words (8 nibbles
# per uint32; a 64-wide group spans exactly 8 words, so a word never crosses a
# group boundary). Per word: s * sum(x*q) + b * sum(x), accumulated in fp32,
# simd-reduced, lane 0 writes the row.
GEMV_MSL = """
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_grid.x;
    uint d_in = x_shape[1];
    uint words = d_in / 8;
    const device half4* x4 = (const device half4*)x;
    float acc = 0.0f;
    for (uint wi = lane; wi < words; wi += 32) {
        uint word = w_q[row * words + wi];
        uint g = wi / 8;
        float s = (float)scales[row * (d_in / 64) + g];
        float b = (float)biases[row * (d_in / 64) + g];
        float4 lo = float4(x4[wi * 2]);
        float4 hi = float4(x4[wi * 2 + 1]);
        float4 qlo = float4(float(word & 0xF), float((word >> 4) & 0xF),
                            float((word >> 8) & 0xF), float((word >> 12) & 0xF));
        float4 qhi = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                            float((word >> 24) & 0xF), float((word >> 28) & 0xF));
        float xq = metal::dot(lo, qlo) + metal::dot(hi, qhi);
        float xs = metal::dot(lo, float4(1.0f)) + metal::dot(hi, float4(1.0f));
        acc = metal::fma(s, xq, metal::fma(b, xs, acc));
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) out[row] = (T)acc;
"""


def pack_nibbles(q: np.ndarray) -> np.ndarray:
    rows, cols = q.shape
    q8 = q.reshape(rows, cols // 8, 8).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
    return np.bitwise_or.reduce(q8 << shifts, axis=2).astype(np.uint32)


def gemv_kernel() -> KernelSpec:
    return KernelSpec(name="kv_dequant_gemv", source=GEMV_MSL,
                      input_names=["x", "w_q", "scales", "biases"],
                      output_names=["out"])


def verify(runner: MetalRunner) -> bool:
    ok = True
    for d_out, d_in in SHAPES:
        rng = np.random.default_rng(7)
        w = (rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.02).astype(np.float16)
        artefact = canonical_quantize(w, CONTRACT)
        packed = pack_nibbles(artefact.q)
        for scale_tag, xs in (("unit", 1.0), ("corpus-scale", 10.0)):
            x = (rng.standard_normal((1, d_in)).astype(np.float32) * xs).astype(np.float16)
            ref = r_contract(x, artefact)
            floor = max(
                np.max(np.abs(fn(x, artefact).astype(np.float64) - ref))
                for fn in ENSEMBLE.values()
            )
            base = 4 * 9.77e-4 * float(np.max(np.abs(ref)))
            tol = max(base, K * floor)

            case = DeviceCase(
                inputs={"x": x, "w_q": packed,
                        "scales": artefact.scales, "biases": artefact.biases},
                output_shapes=[((d_out,), "float16")],
                grid=(32, d_out, 1), threadgroup=(32, 8, 1),
                template={"T": "float16"},
            )
            result = runner.run(gemv_kernel(), [case])[0]
            if not result.ok:
                print(f"  VERIFY FAIL {d_out}x{d_in} {scale_tag}: {result.error}")
                ok = False
                continue
            err = float(np.max(np.abs(result.outputs[0].astype(np.float64) - ref[0])))
            verdict = "pass" if err <= tol else "FAIL"
            if err > tol:
                ok = False
            print(f"  verify {d_out:>5}x{d_in:<5} {scale_tag:<13} "
                  f"err {err:9.3e}  tol {tol:9.3e}  {verdict}")
    return ok


def timed_mx(fn) -> float:
    fn()  # warmup + JIT
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench() -> None:
    kernel = mx.fast.metal_kernel(name="kv_dequant_gemv",
                                  input_names=["x", "w_q", "scales", "biases"],
                                  output_names=["out"], source=GEMV_MSL)
    print(f"\n{'shape':<14}{'ours us':>10}{'mlx us':>10}{'ratio':>8}{'ours GB/s':>11}")
    for d_out, d_in in SHAPES:
        rng = np.random.default_rng(7)
        w = (rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.02).astype(np.float16)
        x16 = (rng.standard_normal((1, d_in)).astype(np.float16))
        artefact = canonical_quantize(w, CONTRACT)
        packed = pack_nibbles(artefact.q)

        mx_x = mx.array(x16)
        mx_wq, mx_s, mx_b = mx.quantize(mx.array(w), group_size=64, bits=4)
        mx_packed = mx.array(packed)
        mx_scales = mx.array(artefact.scales)
        mx_biases = mx.array(artefact.biases)
        mx.eval(mx_x, mx_wq, mx_s, mx_b, mx_packed, mx_scales, mx_biases)

        def ours():
            return kernel(inputs=[mx_x, mx_packed, mx_scales, mx_biases],
                          output_shapes=[(d_out,)], output_dtypes=[mx.float16],
                          grid=(32, d_out, 1), threadgroup=(32, 8, 1),
                          template=[("T", mx.float16)])[0]

        def theirs():
            return mx.quantized_matmul(mx_x, mx_wq, mx_s, mx_b, transpose=True,
                                       group_size=64, bits=4)

        t_ours, t_mlx = timed_mx(ours), timed_mx(theirs)
        weight_bytes = packed.nbytes + artefact.scales.nbytes + artefact.biases.nbytes
        gbps = weight_bytes / t_ours / 1e9
        print(f"{d_out}x{d_in:<7}{t_ours * 1e6:>10.1f}{t_mlx * 1e6:>10.1f}"
              f"{t_mlx / t_ours:>7.2f}x{gbps:>10.1f}")


def main() -> int:
    print("correctness (runner-isolated, Phase 0 contract, K=3):")
    ok = verify(MetalRunner())
    if not ok:
        print("\nSPIKE VERDICT: kernel does not verify; no timing claims allowed")
        return 1
    bench()
    print("\n(ratio > 1.00x means the verified kernel beats mx.quantized_matmul)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
