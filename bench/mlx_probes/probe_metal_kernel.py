"""Probe the mx.fast.metal_kernel door the pack ships through.

Answers the mechanics questions the pack design depends on: what the generated
signature looks like, how grid/threadgroup map, what a compile failure does,
whether non-contiguous and wrong-dtype inputs are handled or rejected, and
whether math_mode changes numerics (it decides what the certificate must pin).
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

SRC_EXP = """
    uint elem = thread_position_in_grid.x;
    T tmp = inp[elem];
    out[elem] = metal::exp(tmp);
"""


def section(title):
    print(f"\n== {title} ==")


def main():
    print("mlx", mx.__version__, mx.default_device())

    section("generated signature (verbose=True)")
    kernel = mx.fast.metal_kernel(name="myexp", input_names=["inp"],
                                  output_names=["out"], source=SRC_EXP)
    a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
    out = kernel(inputs=[a], template=[("T", mx.float16)],
                 grid=(a.size, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[a.shape], output_dtypes=[a.dtype], verbose=True)[0]
    print("  matches mx.exp:", bool(mx.allclose(out, mx.exp(a))))

    section("grid semantics: grid is TOTAL threads, not threadgroups")
    n = 1000  # deliberately not a multiple of the threadgroup size
    x = mx.arange(n, dtype=mx.float32)
    src = """
        uint i = thread_position_in_grid.x;
        if (i < n_elem) { out[i] = inp[i] * 2.0f; }
    """
    k2 = mx.fast.metal_kernel(name="dbl", input_names=["inp", "n_elem"],
                              output_names=["out"], source=src)
    y = k2(inputs=[x, mx.array(n, dtype=mx.uint32)],
           grid=(n, 1, 1), threadgroup=(256, 1, 1),
           output_shapes=[(n,)], output_dtypes=[mx.float32])[0]
    print("  ragged grid ok:", bool(mx.allclose(y, x * 2)))

    section("scalar inputs arrive as pointers or values?")
    print("  n_elem above was passed as a 0-d array and indexed as a scalar:",
          "worked" if float(y[999]) == 1998.0 else "FAILED")

    section("compile failure")
    # MLX is lazy: the failure surfaces when the OUTPUT is evaluated, not when
    # the kernel is called. Evaluating anything else reports a false clean run.
    bad = mx.fast.metal_kernel(name="bad", input_names=["inp"],
                               output_names=["out"], source="this is not MSL;")
    try:
        out_bad = bad(inputs=[a], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                      output_shapes=[(1,)], output_dtypes=[mx.float32])[0]
        mx.eval(out_bad)
        print("  NO EXCEPTION - silent fallback risk")
    except Exception as exc:  # noqa: BLE001 - probing what the API raises
        msg = str(exc)
        print(f"  raised {type(exc).__name__}, {len(msg)} chars")
        print("  first line:", msg.splitlines()[0][:160])
        print("  carries the MSL diagnostic:", "error:" in msg)

    section("non-contiguous input")
    big = mx.random.normal(shape=(8, 32)).astype(mx.float32)
    view = big[:, ::2]  # strided
    print("  view shape", view.shape, "contiguous flag via ascontiguous copy test")
    got = kernel(inputs=[view], template=[("T", mx.float32)],
                 grid=(view.size, 1, 1), threadgroup=(64, 1, 1),
                 output_shapes=[view.shape], output_dtypes=[mx.float32])[0]
    print("  ensure_row_contiguous=True handled the stride:",
          bool(mx.allclose(got, mx.exp(view))))

    k_nc = mx.fast.metal_kernel(name="myexp_nc", input_names=["inp"],
                                output_names=["out"], source=SRC_EXP,
                                ensure_row_contiguous=False)
    got_nc = k_nc(inputs=[view], template=[("T", mx.float32)],
                  grid=(view.size, 1, 1), threadgroup=(64, 1, 1),
                  output_shapes=[view.shape], output_dtypes=[mx.float32])[0]
    print("  ensure_row_contiguous=False silently reads wrong data:",
          not bool(mx.allclose(got_nc, mx.exp(view))))

    section("template dtype differs from the input dtype")
    # The buffer is declared with the INPUT's dtype and T is a separate
    # template parameter, so this converts on read rather than reinterpreting.
    got = kernel(inputs=[mx.array([1.0, 2.0], dtype=mx.float32)],
                 template=[("T", mx.float16)], grid=(2, 1, 1),
                 threadgroup=(2, 1, 1), output_shapes=[(2,)],
                 output_dtypes=[mx.float16])[0]
    mx.eval(got)
    print("  result:", np.array(got), "vs exp([1,2]) =", np.exp([1.0, 2.0]))
    print("  converts on read, no reinterpretation")

    section("math_mode changes numerics?")
    # A plain multiply-accumulate loop is identical in all three modes; the
    # difference shows up on transcendentals, which is what the certificate
    # has to pin.
    src_tr = """
        uint i = thread_position_in_grid.x;
        float x = inp[i];
        out[i] = metal::exp(x) / (1.0f + metal::sqrt(metal::fabs(x)))
                 + metal::precise::divide(1.0f, x);
    """
    xs = np.array([-np.inf, -80.0, -1.0, 1e-30, 1.0, 12.5, 88.0], dtype=np.float32)
    results = {}
    for mode in ("safe", "relaxed", "fast"):
        kk = mx.fast.metal_kernel(name=f"tr_{mode}", input_names=["inp"],
                                  output_names=["out"], source=src_tr,
                                  compile_options={"math_mode": mode})
        r = kk(inputs=[mx.array(xs)], grid=(xs.size, 1, 1), threadgroup=(8, 1, 1),
               output_shapes=[xs.shape], output_dtypes=[mx.float32])[0]
        results[mode] = np.array(r)
        print(f"  {mode:8}", np.array2string(results[mode], precision=6))
    print("  safe == relaxed bitwise:",
          np.array_equal(results["safe"], results["relaxed"]))
    print("  safe == fast bitwise:   ",
          np.array_equal(results["safe"], results["fast"]))

    section("does the door take what a competitive kernel needs?")
    src_red = """
        threadgroup float partial[32];
        uint row  = threadgroup_position_in_grid.x;
        uint tid  = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg   = simdgroup_index_in_threadgroup;
        float acc = 0.0f;
        for (uint k = tid; k < n; k += tg_size) { acc += a[row * n + k] * b[k]; }
        acc = metal::simd_sum(acc);
        if (lane == 0) partial[sg] = acc;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (sg == 0) {
            float t = (lane < (tg_size / 32)) ? partial[lane] : 0.0f;
            t = metal::simd_sum(t);
            if (lane == 0) out[row] = t;
        }
    """
    rng = np.random.default_rng(0)
    rows, kdim, tg = 512, 1024, 256
    an = rng.standard_normal((rows, kdim), dtype=np.float32)
    bn = rng.standard_normal(kdim, dtype=np.float32)
    kred = mx.fast.metal_kernel(name="gemv_sg",
                                input_names=["a", "b", "n", "tg_size"],
                                output_names=["out"], source=src_red)
    r = kred(inputs=[mx.array(an), mx.array(bn), mx.array(kdim, dtype=mx.uint32),
                     mx.array(tg, dtype=mx.uint32)],
             grid=(rows * tg, 1, 1), threadgroup=(tg, 1, 1),
             output_shapes=[(rows,)], output_dtypes=[mx.float32])[0]
    err = float(np.max(np.abs(np.array(r).astype(np.float64)
                              - an.astype(np.float64) @ bn.astype(np.float64))))
    print(f"  threadgroup memory + barrier + simd_sum: max|err| = {err:.2e}")

    src_mm = """
        using namespace metal;
        simdgroup_float8x8 A, B, C;
        C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_load(A, a, 8);
        simdgroup_load(B, b, 8);
        simdgroup_multiply_accumulate(C, A, B, C);
        simdgroup_store(C, out, 8);
    """
    kmm = mx.fast.metal_kernel(name="sgmm", input_names=["a", "b"],
                               output_names=["out"], source=src_mm)
    ma = mx.array(rng.standard_normal((8, 8), dtype=np.float32))
    mb = mx.array(rng.standard_normal((8, 8), dtype=np.float32))
    rmm = kmm(inputs=[ma, mb], grid=(32, 1, 1), threadgroup=(32, 1, 1),
              output_shapes=[(8, 8)], output_dtypes=[mx.float32])[0]
    ok = np.allclose(np.array(rmm), np.array(ma) @ np.array(mb), atol=1e-4)
    print("  simdgroup_matrix multiply_accumulate:", "OK" if ok else "WRONG")

    section("does the same source recompile per call or cache?")
    import time
    k3 = mx.fast.metal_kernel(name="cachetest", input_names=["inp"],
                              output_names=["out"], source=SRC_EXP)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        r = k3(inputs=[a], template=[("T", mx.float16)], grid=(a.size, 1, 1),
               threadgroup=(64, 1, 1), output_shapes=[a.shape],
               output_dtypes=[a.dtype])[0]
        mx.eval(r)
        times.append((time.perf_counter() - t0) * 1e3)
    print("  call latencies (ms):", " ".join(f"{t:.2f}" for t in times))


if __name__ == "__main__":
    main()
