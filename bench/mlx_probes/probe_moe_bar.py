"""Measure the MoE-dispatch and small-shape headroom the pack would target.

Timing rules, both of which are load-bearing:
- one graph of N copies per measurement, evaluated once (MLX is lazy, so
  timing N graph builds and one eval measures nothing);
- weights rotated over a working set larger than any cache, and the expert
  selection rotated too, so no measurement reports cache bandwidth. Any row
  reading above the streaming ceiling means the rotation failed.

Shapes are Qwen3-30B-A3B class: hidden 2048, 128 experts, 8 active, expert
intermediate 768. A 4-bit checkpoint of that model is ~17 GB and fits this
36 GB machine, which is what makes it a plausible first pack target.
"""

from __future__ import annotations

import time

import mlx.core as mx

HIDDEN, N_EXPERTS, N_ACTIVE, FFN = 2048, 128, 8, 768
BITS, GROUP = 4, 64
WORKING_SET_MB = 512


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


def qbytes(*shape):
    n = 1
    for s in shape:
        n *= s
    return n * BITS / 8 + 2 * (n / GROUP) * 2


def measure_ceiling():
    n = 256 * 1024 * 1024 // 2
    bufs = [mx.random.normal(shape=(n,)).astype(mx.float16) for _ in range(2)]
    mx.eval(bufs)
    t = time_graph(lambda i: mx.sum(bufs[i % 2]), n_copies=8, trials=3)
    return (n * 2) / t / 1e9


def pct(bytes_moved, t, ceiling):
    gbps = bytes_moved / t / 1e9
    return f"{gbps:6.1f} GB/s ({100*gbps/ceiling:5.1f}% of ceiling)"


def main():
    print("mlx", mx.__version__, mx.default_device())
    ceiling = measure_ceiling()
    print(f"streaming ceiling measured now: {ceiling:.1f} GB/s\n")

    print("== MoE expert projection at decode, one token ==")
    x = mx.random.normal(shape=(1, HIDDEN)).astype(mx.float16)
    tensor_bytes = qbytes(N_EXPERTS, FFN, HIDDEN)
    n_tensors = max(2, int(WORKING_SET_MB * 1e6 // tensor_bytes) + 1)
    packs = []
    for _ in range(n_tensors):
        w = mx.random.normal(shape=(N_EXPERTS, FFN, HIDDEN)).astype(mx.float16)
        packs.append(mx.quantize(w, group_size=GROUP, bits=BITS))
        del w
    mx.eval([a for p in packs for a in p], x)
    # A different expert set every call, so no expert stays resident.
    idxs = [mx.array([[(e * 13 + j * 29) % N_EXPERTS for j in range(N_ACTIVE)]],
                     dtype=mx.uint32) for e in range(n_tensors)]
    mx.eval(idxs)
    active_bytes = qbytes(N_ACTIVE, FFN, HIDDEN)
    print(f"  {n_tensors} expert tensors x {tensor_bytes/1e6:.0f} MB = "
          f"{n_tensors*tensor_bytes/1e6:.0f} MB working set; "
          f"{active_bytes/1e6:.2f} MB read per call")

    def gather_build(i):
        wq, sc, bi = packs[i % n_tensors]
        return mx.gather_qmm(x, wq, sc, bi, rhs_indices=idxs[i % n_tensors],
                             transpose=True, group_size=GROUP, bits=BITS,
                             sorted_indices=False)

    t_gather = time_graph(gather_build, n_copies=max(n_tensors, 24))
    print(f"  gather_qmm (MLX built-in)   {t_gather*1e6:8.1f} us  "
          f"{pct(active_bytes, t_gather, ceiling)}")

    def loop_build(i):
        wq, sc, bi = packs[i % n_tensors]
        sel = idxs[i % n_tensors].tolist()[0]
        outs = [mx.quantized_matmul(x, wq[e], sc[e], bi[e], transpose=True,
                                    group_size=GROUP, bits=BITS) for e in sel]
        return mx.concatenate(outs, axis=0)

    t_loop = time_graph(loop_build, n_copies=max(n_tensors, 8))
    print(f"  8 separate quantized_matmul {t_loop*1e6:8.1f} us  "
          f"{pct(active_bytes, t_loop, ceiling)}")
    print(f"  the built-in gather is {t_loop/t_gather:.2f}x the naive loop")

    print("\n== batch sweep, 4096 x 4096, 4-bit group 64, cold weights ==")
    wbytes = qbytes(4096, 4096)
    n_w = max(2, int(WORKING_SET_MB * 1e6 // wbytes) + 1)
    ws = []
    for _ in range(n_w):
        ws.append(mx.quantize(
            mx.random.normal(shape=(4096, 4096)).astype(mx.float16),
            group_size=GROUP, bits=BITS))
    mx.eval([a for p in ws for a in p])
    for batch in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32):
        xb = mx.random.normal(shape=(batch, 4096)).astype(mx.float16)
        mx.eval(xb)

        def build(i, xb=xb):
            wq, sc, bi = ws[i % n_w]
            return mx.quantized_matmul(xb, wq, sc, bi, transpose=True,
                                       group_size=GROUP, bits=BITS)

        tb = time_graph(build, n_copies=max(n_w, 24))
        print(f"  batch {batch:3d}  {tb*1e6:8.1f} us  {pct(wbytes, tb, ceiling)}  "
              f"{tb*1e6/batch:6.1f} us/row")

    print("\n== KV-cache attention at decode, cold cache ==")
    B, H, KVH, D = 1, 16, 2, 128
    for ctx in (4096, 16384):
        kv_bytes = 2 * B * KVH * ctx * D * 2
        n_kv = max(2, int(WORKING_SET_MB * 1e6 // kv_bytes) + 1)
        n_kv = min(n_kv, 32)
        kvs = []
        for _ in range(n_kv):
            kvs.append((mx.random.normal(shape=(B, KVH, ctx, D)).astype(mx.float16),
                        mx.random.normal(shape=(B, KVH, ctx, D)).astype(mx.float16)))
        q = mx.random.normal(shape=(B, H, 1, D)).astype(mx.float16)
        mx.eval([a for p in kvs for a in p], q)

        def build(i):
            k, v = kvs[i % n_kv]
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5)

        t = time_graph(build, n_copies=max(n_kv, 24))
        print(f"  fp16 SDPA ctx={ctx:6d} ({kv_bytes/1e6:5.1f} MB KV)  "
              f"{t*1e6:8.1f} us  {pct(kv_bytes, t, ceiling)}")


if __name__ == "__main__":
    main()
