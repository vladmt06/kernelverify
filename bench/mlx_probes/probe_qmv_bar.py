"""Measure the bar the pack has to beat: MLX's own quantized matmul at decode shapes.

The plan's headroom number for a fused dequant-GEMV (1.2-1.5x) was read off
llama.cpp baselines. The pack ships through MLX, so the number that decides
whether the kernel is worth writing is MLX's own qmv, measured here.

Timing rules this harness follows, because getting them wrong inflates results
by an order of magnitude:

- MLX is lazy. Building N graphs and evaluating one measures graph
  construction. Every timed op is put in one graph of N copies, evaluated once,
  so the sync cost is paid once and N kernel executions are actually timed.
- A single weight buffer of a few MB lives in the system-level cache and
  reports bandwidth far above DRAM. Weights are rotated over a working set
  larger than any cache, which is the regime real decode runs in.
"""

from __future__ import annotations

import time

import mlx.core as mx

WORKING_SET_MB = 512  # comfortably past the SLC on any current M-series part


def time_graph(build, n_copies, trials=5):
    """Seconds per op: n_copies independent ops in one graph, evaluated once."""
    mx.eval(build(0))
    mx.synchronize()
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        outs = [build(i) for i in range(n_copies)]
        mx.eval(outs)
        mx.synchronize()
        best = min(best, (time.perf_counter() - t0) / n_copies)
    return best


def rotating(make, bytes_each):
    """Enough distinct buffers to exceed the cache, returned as a list."""
    count = max(2, int(WORKING_SET_MB * 1e6 // max(bytes_each, 1)) + 1)
    count = min(count, 64)
    bufs = [make() for _ in range(count)]
    mx.eval([b for tup in bufs for b in (tup if isinstance(tup, tuple) else (tup,))])
    return bufs


def quant_bytes(rows, cols, bits, group):
    """Packed weights + fp16 scales + fp16 biases."""
    return rows * cols * bits / 8 + 2 * (rows * cols / group) * 2


def measure_ceiling():
    n = 256 * 1024 * 1024 // 2  # 256 MB of fp16 per buffer
    bufs = [mx.random.normal(shape=(n,)).astype(mx.float16) for _ in range(2)]
    mx.eval(bufs)
    t = time_graph(lambda i: mx.sum(bufs[i % 2]), n_copies=8, trials=3)
    gbps = (n * 2) / t / 1e9
    print(f"  streaming fp16 sum, 256 MB per pass: {t*1e3:7.3f} ms -> {gbps:6.1f} GB/s")
    return gbps


def bench_qmv(label, out_dim, in_dim, bits, group, batch, ceiling):
    wbytes = quant_bytes(out_dim, in_dim, bits, group)

    def make():
        w = mx.random.normal(shape=(out_dim, in_dim)).astype(mx.float16)
        return mx.quantize(w, group_size=group, bits=bits)

    bufs = rotating(make, wbytes)
    x = mx.random.normal(shape=(batch, in_dim)).astype(mx.float16)
    mx.eval(x)

    def build(i):
        wq, sc, bi = bufs[i % len(bufs)]
        return mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                   group_size=group, bits=bits)

    t = time_graph(build, n_copies=max(len(bufs), 16))
    gbps = wbytes / t / 1e9
    print(f"  {label:34} {t*1e6:8.1f} us  {gbps:6.1f} GB/s "
          f"({100*gbps/ceiling:5.1f}% of ceiling, {len(bufs)} buffers)")
    return t, gbps


def main():
    print("mlx", mx.__version__, mx.default_device())
    print("\n== practical streaming ceiling (this process, this moment) ==")
    ceiling = measure_ceiling()

    print("\n== dense fp16 matvec, decode (M=1), cold weights ==")
    for out_dim, in_dim in ((4096, 4096), (14336, 4096)):
        wbytes = out_dim * in_dim * 2
        bufs = rotating(
            lambda: mx.random.normal(shape=(out_dim, in_dim)).astype(mx.float16),
            wbytes)
        x = mx.random.normal(shape=(1, in_dim)).astype(mx.float16)
        mx.eval(x)
        t = time_graph(lambda i: x @ bufs[i % len(bufs)].T,
                       n_copies=max(len(bufs), 16))
        gbps = wbytes / t / 1e9
        print(f"  [{out_dim:6d} x {in_dim:5d}] {t*1e6:8.1f} us  {gbps:6.1f} GB/s "
              f"({100*gbps/ceiling:5.1f}% of ceiling)")

    print("\n== MLX quantized matvec, affine 4-bit group 64, cold weights ==")
    for label, out_dim, in_dim in (
        ("dense 4096 x 4096", 4096, 4096),
        ("dense 14336 x 4096", 14336, 4096),
        ("Qwen3-30B-A3B attn 2048x2048", 2048, 2048),
        ("Qwen3-30B-A3B expert 768x2048", 768, 2048),
    ):
        bench_qmv(label, out_dim, in_dim, 4, 64, 1, ceiling)

    print("\n== bit width sweep, 4096 x 4096, group 64, M=1 ==")
    for bits in (2, 3, 4, 5, 6, 8):
        bench_qmv(f"{bits}-bit", 4096, 4096, bits, 64, 1, ceiling)

    print("\n== group size sweep, 4096 x 4096, 4-bit, M=1 ==")
    for group in (32, 64, 128):
        bench_qmv(f"group {group}", 4096, 4096, 4, group, 1, ceiling)

    print("\n== batch sweep, 4096 x 4096, 4-bit group 64 ==")
    for batch in (1, 2, 4, 8, 32, 128):
        bench_qmv(f"batch {batch}", 4096, 4096, 4, 64, batch, ceiling)


if __name__ == "__main__":
    main()
