"""Machine ceiling: achieved memory bandwidth and matmul throughput on this Mac.

The design doc's measurement-first rule: every headroom claim is checked
against the ceiling of the actual machine before any kernel work chases it.
Bandwidth is measured with a large copy (2 bytes moved per element) and a
large sum (1 byte-read per element counted); compute with fp16 matmul.
Median of repeats, warmup discarded; run on an otherwise-idle machine.
"""

from __future__ import annotations

import statistics
import sys
import time

import mlx.core as mx

REPEATS = 9


def timed(fn, *args) -> float:
    out = fn(*args)
    mx.eval(out)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> int:
    dev = mx.default_device()
    n = 256 * 1024 * 1024  # elements, fp16: 512 MiB per buffer
    a = mx.random.normal((n,), dtype=mx.float16)
    mx.eval(a)

    t_copy = timed(lambda x: x + 0, a)          # read n*2 + write n*2 bytes
    bw_copy = (n * 4) / t_copy / 1e9
    t_sum = timed(lambda x: mx.sum(x), a)        # read n*2 bytes
    bw_sum = (n * 2) / t_sum / 1e9

    m = 4096
    b = mx.random.normal((m, m), dtype=mx.float16)
    c = mx.random.normal((m, m), dtype=mx.float16)
    mx.eval(b, c)
    t_mm = timed(lambda x, y: x @ y, b, c)
    tflops = (2 * m ** 3) / t_mm / 1e12

    print(f"device: {dev}")
    print(f"achieved bandwidth (copy):   {bw_copy:7.1f} GB/s")
    print(f"achieved bandwidth (sum):    {bw_sum:7.1f} GB/s")
    print(f"achieved fp16 matmul:        {tflops:7.2f} TFLOPS  ({m}x{m}x{m})")
    print("M3 Pro spec bandwidth: 150 GB/s (the decode ceiling on this machine)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
