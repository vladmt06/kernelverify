"""Diagnose the batch 5-11 cost step in MLX's quantized matmul.

The naive reading of a batch sweep is that MLX loses half its bandwidth
between batch 4 and batch 8 on constant weight traffic. It does not.
`QuantizedMatmul::eval_gpu` routes batch 2-11 to `qmv_wide`, which sets
`n_tiles = ceil(M / 5)` and gives each threadgroup a tile of at most five
input vectors - and every tile re-reads the whole weight matrix. The traffic
is a step function, so the bandwidth has to be recomputed against
`ceil(M/5) * W` before it means anything.

This probe prints both columns so the step is visible rather than inferred.
"""

from __future__ import annotations

import math
import time

import mlx.core as mx

BITS, GROUP = 4, 64
OUT_DIM, IN_DIM = 4096, 4096
WORKING_SET_MB = 512
VECTOR_LIMIT = 12  # get_qmv_batch_limit(4096, 4096) on applegpu_g15s
TILE_CAP = 5       # qmv_wide's n_tiles = ceil(M / 5)


def time_graph(build, n_copies, trials=5):
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


def qbytes(rows, cols):
    return rows * cols * BITS / 8 + 2 * (rows * cols / GROUP) * 2


def main():
    info = mx.device_info()
    print(f"mlx {mx.__version__}  {info['device_name']}  {info['architecture']}")

    n_buf = 256 * 1024 * 1024 // 2
    ceil_bufs = [mx.random.normal(shape=(n_buf,)).astype(mx.float16) for _ in range(2)]
    mx.eval(ceil_bufs)
    t_ceiling = time_graph(lambda i: mx.sum(ceil_bufs[i % 2]), n_copies=8, trials=3)
    ceiling = (n_buf * 2) / t_ceiling / 1e9
    del ceil_bufs
    print(f"streaming ceiling: {ceiling:.1f} GB/s\n")

    weight_bytes = qbytes(OUT_DIM, IN_DIM)
    n_w = max(2, int(WORKING_SET_MB * 1e6 // weight_bytes) + 1)
    ws = [mx.quantize(mx.random.normal(shape=(OUT_DIM, IN_DIM)).astype(mx.float16),
                      group_size=GROUP, bits=BITS) for _ in range(n_w)]
    mx.eval([a for p in ws for a in p])

    print(f"{'M':>3} {'path':>11} {'time us':>9} {'tiles':>6} {'us/pass':>8} "
          f"{'GB/s naive':>11} {'GB/s real':>10} {'% ceiling':>10}")
    for m in range(1, 17):
        xb = mx.random.normal(shape=(m, IN_DIM)).astype(mx.float16)
        mx.eval(xb)

        def build(i, xb=xb):
            wq, sc, bi = ws[i % n_w]
            return mx.quantized_matmul(xb, wq, sc, bi, transpose=True,
                                       group_size=GROUP, bits=BITS)

        t = time_graph(build, n_copies=max(n_w, 24))
        naive = weight_bytes / t / 1e9
        if m >= VECTOR_LIMIT:
            print(f"{m:3d} {'qmm_splitk':>11} {t*1e6:9.1f} {'-':>6} {'-':>8} "
                  f"{naive:11.1f} {'-':>10} {'-':>10}")
            continue
        path = "qmv_quad" if m == 1 else "qmv_wide"
        tiles = math.ceil(m / TILE_CAP)
        real = tiles * weight_bytes / t / 1e9
        print(f"{m:3d} {path:>11} {t*1e6:9.1f} {tiles:6d} {t*1e6/tiles:8.1f} "
              f"{naive:11.1f} {real:10.1f} {100*real/ceiling:9.1f}%")

    single_pass = weight_bytes / (ceiling * 1e9)
    print(f"\none weight pass at the ceiling: {single_pass*1e6:.1f} us")
    print("that is the bound a wider vector tile is chasing at batch 6-11;")
    print("the five-vector tile already costs 23% more per pass than the "
          "four-vector one, which is the register tax a wider tile also pays.")


if __name__ == "__main__":
    main()
