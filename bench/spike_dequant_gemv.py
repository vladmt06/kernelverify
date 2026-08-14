"""T5 feasibility spike: one competitive fused dequant-GEMV on Metal.

The D7 decision experiment (eng review), migrated to the consolidated runner
(W4): the candidate is raw MSL specialized through `specialize()`, executed
by the crash-isolated `MetalRunner`, and judged through the pack's shared
verification path (`kernelverify.pack.verify`), which routes to the battery's
own NATIVE_OPS reference and tolerance - the r_contract anchor, ensemble
floor, K = 3, exactly the formula the pre-migration spike hand-rolled.

Import layout is deliberate: the module imports only numpy and the runner, so
the kernel assembly is testable on the consolidation branch, while the pack
and MLX imports live inside `verify()`/`bench()` and resolve once the pack
branch is on main (it merges ahead of this one). The GEMV parity gate in the
consolidation merge review runs this file end to end and compares against the
pre-migration output; fail means no merge.

One kernel body serves both doors. The runner door wraps it in an explicit
raw-MSL signature (`d_in` arrives as a scalar); the MLX timing door hands the
same body to `mx.fast.metal_kernel`, whose generated signature supplies
`x_shape` instead. The timing methodology is unchanged from the original
spike so the parity comparison is like for like.

Context that frames the result: on this M3 Pro, llama.cpp decodes at 92% of
the achieved-bandwidth roofline (little to win) while MLX decodes at ~69%
(a real gap, and MLX is the launch channel). The spike asks whether a simple
verified kernel can close any of MLX's gap.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelverify.runners import (  # noqa: E402
    Binding,
    BindingKind,
    KernelSpec,
    LaunchSpec,
    MetalRunner,
    RunCase,
    specialize,
)

BITS = 4
GROUP_SIZE = 64
K_SHAPES = [(9728, 2560), (2560, 9728), (2560, 2560)]  # (d_out, d_in), Qwen3-4B decode
REPEATS = 30

# One simdgroup per output row. Each lane strides over packed words (8 nibbles
# per uint32; a 64-wide group spans exactly 8 words, so a word never crosses a
# group boundary). Per word: s * sum(x*q) + b * sum(x), accumulated in fp32,
# simd-reduced, lane 0 writes the row. `D_IN` is spelled differently per door:
# the MLX door substitutes MLX's generated x_shape, the runner door a scalar.
GEMV_BODY = """
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_grid.x;
    uint d_in = D_IN;
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

_RUNNER_SOURCE = f"""
#include <metal_stdlib>
using namespace metal;
using T = $T;
kernel void dequant_gemv(
    device const half* x      [[buffer(0)]],
    device const uint* w_q    [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* biases [[buffer(3)]],
    device T* out             [[buffer(4)]],
    constant uint& d_in_arg   [[buffer(5)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {{
{GEMV_BODY.replace("D_IN", "d_in_arg")}
}}
"""


def gemv_template() -> KernelSpec:
    """The runner-door template; `specialize` picks the output type."""
    return KernelSpec(
        source=_RUNNER_SOURCE,
        entry_point="dequant_gemv",
        name="kv_dequant_gemv-$T",
        bindings=(Binding(BindingKind.INPUT, "x"),
                  Binding(BindingKind.INPUT, "w_q"),
                  Binding(BindingKind.INPUT, "scales"),
                  Binding(BindingKind.INPUT, "biases"),
                  Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "d_in_arg", "uint32")),
        launch=LaunchSpec(grid=(32, "d_out", 1), threadgroup=(32, 8, 1)),
    )


def gemv_case(x: np.ndarray, packed: np.ndarray, scales: np.ndarray,
              biases: np.ndarray, d_out: int, label: str = "") -> RunCase:
    d_in = x.shape[-1]
    return RunCase(
        inputs={"x": x, "w_q": packed, "scales": scales, "biases": biases},
        params={"d_out": d_out, "d_in_arg": d_in},
        output_shapes=[((d_out,), "float16")],
        label=label,
    )


# --------------------------------------------------------------------------
# verification: the pack's shared path, batched through one candidate session
# --------------------------------------------------------------------------
def verify(runner: MetalRunner) -> bool:
    from kernelverify.pack.verify import judge, qmv_inputs, reference_and_tolerance
    from kernelverify.pack.wide_qmv import pack_nibbles
    from kernelverify.schemas.quant_contract import QuantContract, canonical_quantize

    spec = specialize(gemv_template(), {"T": "half"})
    ok = True
    for d_out, d_in in K_SHAPES:
        rng = np.random.default_rng(7)
        w = (rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.02).astype(np.float16)
        artefact = canonical_quantize(w, QuantContract(bits=BITS, group_size=GROUP_SIZE))
        packed = pack_nibbles(artefact.q)

        cases, expected = [], []
        for scale_tag, xs in (("unit", 1.0), ("corpus-scale", 10.0)):
            x = (rng.standard_normal((1, d_in)).astype(np.float32) * xs).astype(np.float16)
            ref, tol = reference_and_tolerance("quantized_matmul", qmv_inputs(x, w, BITS))
            cases.append(gemv_case(x, packed, artefact.scales, artefact.biases,
                                   d_out, label=scale_tag))
            expected.append((scale_tag, ref, tol))

        for result, (scale_tag, ref, tol) in zip(runner.run(spec, cases), expected):
            if not result.ok:
                print(f"  VERIFY FAIL {d_out}x{d_in} {scale_tag}: "
                      f"{result.status.value}: {result.detail}")
                ok = False
                continue
            verdict = judge(result.outputs[0], ref, tol)
            ok = ok and verdict.ok
            print(f"  verify {d_out:>5}x{d_in:<5} {scale_tag:<13} {verdict}")
    return ok


# --------------------------------------------------------------------------
# timing: unchanged methodology from the pre-migration spike, for parity
# --------------------------------------------------------------------------
def timed_mx(fn, mx) -> float:
    fn()  # warmup + JIT
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench() -> None:
    import mlx.core as mx

    from kernelverify.pack.wide_qmv import pack_nibbles
    from kernelverify.schemas.quant_contract import QuantContract, canonical_quantize

    kernel = mx.fast.metal_kernel(name="kv_dequant_gemv",
                                  input_names=["x", "w_q", "scales", "biases"],
                                  output_names=["out"],
                                  source=GEMV_BODY.replace("D_IN", "x_shape[1]"))
    print(f"\n{'shape':<14}{'ours us':>10}{'mlx us':>10}{'ratio':>8}{'ours GB/s':>11}")
    for d_out, d_in in K_SHAPES:
        rng = np.random.default_rng(7)
        w = (rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.02).astype(np.float16)
        x16 = rng.standard_normal((1, d_in)).astype(np.float16)
        artefact = canonical_quantize(w, QuantContract(bits=BITS, group_size=GROUP_SIZE))
        packed = pack_nibbles(artefact.q)

        mx_x = mx.array(x16)
        mx_wq, mx_s, mx_b = mx.quantize(mx.array(w), group_size=GROUP_SIZE, bits=BITS)
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
                                       group_size=GROUP_SIZE, bits=BITS)

        t_ours, t_mlx = timed_mx(ours, mx), timed_mx(theirs, mx)
        weight_bytes = packed.nbytes + artefact.scales.nbytes + artefact.biases.nbytes
        gbps = weight_bytes / t_ours / 1e9
        print(f"{d_out}x{d_in:<7}{t_ours * 1e6:>10.1f}{t_mlx * 1e6:>10.1f}"
              f"{t_mlx / t_ours:>7.2f}x{gbps:>10.1f}")


def main() -> int:
    print("correctness (runner-isolated, shared pack verdict, K = 3):")
    if not verify(MetalRunner()):
        print("\nSPIKE VERDICT: kernel does not verify; no timing claims allowed")
        return 1
    bench()
    print("\n(ratio > 1.00x means the verified kernel beats mx.quantized_matmul)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
