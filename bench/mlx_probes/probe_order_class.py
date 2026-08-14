"""Does a hand-picked ensemble bound the admissible class on the pack's surface?

The phase0 lane measured, on the corpus operators, that seeded random summation
permutations exceed the hand-picked sequential-order member by up to 8.3x, so a
floor built from a few chosen implementations under-states the class it claims
to bound. This checks the same thing on the surface the pack ships into: a
quantized matvec against the MLX-affine contract.

The contract members here are all admissible under the same freedoms: any
association order, any permutation of the reduction, fp32-or-wider
intermediates. The question is only how far apart they get.

Run at fp32 so the result isolates reduction order from output-dtype rounding.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

BITS, GROUP = 4, 64
ROWS, KDIM = 64, 4096
N_PERMS = 200
SEED = 0


def unpack(wq: np.ndarray, bits: int, cols: int) -> np.ndarray:
    byts = np.ascontiguousarray(wq).view(np.uint8)
    rows = byts.shape[0]
    stream = np.zeros((rows, byts.shape[1] // 8 + 2), dtype=np.uint64)
    for b in range(byts.shape[1]):
        stream[:, b // 8] |= byts[:, b].astype(np.uint64) << np.uint64(8 * (b % 8))
    out = np.zeros((rows, cols), dtype=np.uint32)
    mask = (1 << bits) - 1
    for i in range(cols):
        word, off = divmod(i * bits, 64)
        val = stream[:, word] >> np.uint64(off)
        if off + bits > 64:
            val = val | (stream[:, word + 1] << np.uint64(64 - off))
        out[:, i] = (val & np.uint64(mask)).astype(np.uint32)
    return out


def main():
    rng = np.random.default_rng(SEED)
    w = mx.array(rng.standard_normal((ROWS, KDIM)).astype(np.float32))
    x = mx.array(rng.standard_normal((1, KDIM)).astype(np.float32))
    wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=BITS)
    mx.eval(wq, sc, bi, x)

    # The contract reference: the intended quantized weights, exactly, in fp64.
    q = unpack(np.array(wq), BITS, KDIM).astype(np.float64)
    s = np.array(sc).astype(np.float64).repeat(GROUP, axis=1)
    b = np.array(bi).astype(np.float64).repeat(GROUP, axis=1)
    w64 = s * q + b
    xn64 = np.array(x).astype(np.float64).ravel()
    ref = w64 @ xn64

    # Working-precision products, shared by every member below.
    w32 = w64.astype(np.float32)
    x32 = xn64.astype(np.float32)
    prod = (w32 * x32[None, :]).astype(np.float32)

    def err(vals):
        return float(np.max(np.abs(vals.astype(np.float64) - ref)))

    print(f"contract reference: fp64, {ROWS} rows, K = {KDIM}, "
          f"output scale {float(np.max(np.abs(ref))):.2f}\n")

    hand = {}
    hand["numpy dot (pairwise)"] = err(w32 @ x32)
    hand["sequential accumulate"] = err(np.add.accumulate(prod, axis=1)[:, -1])
    hand["reversed order"] = err(np.add.accumulate(prod[:, ::-1], axis=1)[:, -1])
    hand["magnitude ascending"] = err(
        np.take_along_axis(prod, np.argsort(np.abs(prod), axis=1), axis=1)
        .cumsum(axis=1, dtype=np.float32)[:, -1])
    hand["magnitude descending"] = err(
        np.take_along_axis(prod, np.argsort(-np.abs(prod), axis=1), axis=1)
        .cumsum(axis=1, dtype=np.float32)[:, -1])
    for width in (32, 64, 128):
        blocked = prod.reshape(ROWS, KDIM // width, width).sum(axis=2, dtype=np.float32)
        hand[f"blocked at {width}"] = err(
            np.add.accumulate(blocked, axis=1)[:, -1])

    print("hand-picked members, the kind an ensemble is built from:")
    for name, e in hand.items():
        print(f"  {name:24} {e:.4e}")
    hand_floor = max(hand.values())
    print(f"  hand-picked floor        {hand_floor:.4e}")

    perms = []
    prng = np.random.default_rng(SEED + 1)
    for _ in range(N_PERMS):
        p = prng.permutation(KDIM)
        perms.append(err(np.add.accumulate(prod[:, p], axis=1)[:, -1]))
    perms = np.array(perms)

    print(f"\n{N_PERMS} seeded random permutations, all equally admissible:")
    print(f"  min                      {perms.min():.4e}")
    print(f"  median                   {np.median(perms):.4e}")
    print(f"  max                      {perms.max():.4e}")

    print(f"\n  permutation max / hand-picked floor        "
          f"{perms.max()/hand_floor:.2f}x")
    print(f"  permutation max / sequential member        "
          f"{perms.max()/hand['sequential accumulate']:.2f}x")
    print(f"  fraction of permutations above the floor   "
          f"{100*float((perms > hand_floor).mean()):.1f}%")


if __name__ == "__main__":
    main()
