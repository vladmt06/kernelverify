"""Does the permissive ordering clause cost the pack any detection, measured?

The phase0 lane swept its ensemble from 1 to 32 members and K from 1.0 to 6.0
and found not one of 9,678 verdicts moves, so on the corpus operators the
ordering clause costs no detection and there is nothing to tighten. The claim
that a wider floor gives away detection therefore has to be shown on the
surface that makes it, not argued from another one.

This runs that test on a dequant-GEMV against the MLX-affine contract.

Two floors are computed over the same case:

    floor_device  = worst error of implementations a Metal kernel can emit
                    (blocked and tree reductions, the fused per-group form)
    floor_class   = worst error over the full admissible class, which adds
                    sequential, reversed, magnitude-sorted and seeded random
                    permutations

Then realistic kernel-side faults are seeded and each is judged under
tolerance = max(base_tol, K * floor) for both floors. A fault caught under the
device floor and absolved under the class floor is a measured detection loss
caused by admitting orderings no device emits. If that column stays empty, the
phase0 lane's position holds on this surface too and the proposal should be
dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase0_contract_k import unpack_mlx_q  # noqa: E402

BITS, GROUP = 4, 64
ROWS, KDIM = 64, 4096
N_PERMS = 64
K_ENSEMBLE = 1.5
SEED = 0


def build_case(dtype):
    rng = np.random.default_rng(SEED)
    w = mx.array(rng.standard_normal((ROWS, KDIM)).astype(dtype))
    x = mx.array(rng.standard_normal((1, KDIM)).astype(dtype))
    wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=BITS)
    mx.eval(wq, sc, bi, x)
    q = unpack_mlx_q(np.array(wq), KDIM, BITS)
    return q, np.array(sc), np.array(bi), np.array(x).ravel()


def dequantized(q, sc, bi, precision=np.float64):
    s = sc.astype(precision).repeat(GROUP, axis=1)
    b = bi.astype(precision).repeat(GROUP, axis=1)
    return s * q.astype(precision) + b


# --- correct implementations -------------------------------------------------

def impl_blocked(q, sc, bi, x, width, out_dtype):
    w = dequantized(q, sc, bi, np.float32)
    prod = (w * x.astype(np.float32)[None, :]).astype(np.float32)
    blocked = prod.reshape(ROWS, KDIM // width, width).sum(axis=2, dtype=np.float32)
    return np.add.accumulate(blocked, axis=1)[:, -1].astype(out_dtype)


def impl_tree(q, sc, bi, x, out_dtype):
    """Pairwise tree over the whole reduction: what a simdgroup reduce does."""
    w = dequantized(q, sc, bi, np.float32)
    acc = (w * x.astype(np.float32)[None, :]).astype(np.float32)
    while acc.shape[1] > 1:
        half = acc.shape[1] // 2
        acc = (acc[:, :half] + acc[:, half:2 * half]).astype(np.float32)
    return acc[:, 0].astype(out_dtype)


def impl_fused_group(q, sc, bi, x, out_dtype):
    """s * sum(code * x) + b * sum(x) per group: the standard fused form."""
    xg = x.astype(np.float32).reshape(KDIM // GROUP, GROUP)
    qg = q.reshape(ROWS, KDIM // GROUP, GROUP).astype(np.float32)
    dots = (qg * xg[None, :, :]).sum(axis=2, dtype=np.float32)
    xsum = xg.sum(axis=1, dtype=np.float32)
    per_group = (dots * sc.astype(np.float32)
                 + bi.astype(np.float32) * xsum[None, :]).astype(np.float32)
    return np.add.accumulate(per_group, axis=1)[:, -1].astype(out_dtype)


def impl_permuted(q, sc, bi, x, perm, out_dtype):
    w = dequantized(q, sc, bi, np.float32)
    prod = (w * x.astype(np.float32)[None, :]).astype(np.float32)
    return np.add.accumulate(prod[:, perm], axis=1)[:, -1].astype(out_dtype)


# --- kernel-side faults ------------------------------------------------------

def fault_bitmap_shift(q, sc, bi, x, out_dtype):
    """Codes read one nibble position along: the classic unpack fault."""
    return impl_blocked(np.roll(q, 1, axis=1), sc, bi, x, 64, out_dtype)


def fault_drop_bias(q, sc, bi, x, out_dtype):
    return impl_blocked(q, sc, np.zeros_like(bi), x, 64, out_dtype)


def fault_group_off_by_one(q, sc, bi, x, out_dtype):
    return impl_blocked(q, np.roll(sc, 1, axis=1), np.roll(bi, 1, axis=1),
                        x, 64, out_dtype)


def fault_fp16_accumulate(q, sc, bi, x, out_dtype):
    w = dequantized(q, sc, bi, np.float32)
    prod = (w * x.astype(np.float32)[None, :]).astype(np.float16)
    acc = np.zeros(ROWS, np.float16)
    for j in range(0, KDIM, 64):
        acc = (acc + prod[:, j:j + 64].sum(axis=1, dtype=np.float16)).astype(np.float16)
    return acc.astype(out_dtype)


def fault_wide_mask(q, sc, bi, x, out_dtype):
    """Unpack mask one bit too wide, so codes pick up the neighbour's low bit."""
    wide = (q | ((np.roll(q, -1, axis=1) & 1) << BITS)).astype(np.uint32)
    return impl_blocked(wide, sc, bi, x, 64, out_dtype)


def fault_scale_only_first_group(q, sc, bi, x, out_dtype):
    """Every group uses group 0's scale and bias: a broadcast indexing slip."""
    sc2 = np.repeat(sc[:, :1], sc.shape[1], axis=1)
    bi2 = np.repeat(bi[:, :1], bi.shape[1], axis=1)
    return impl_blocked(q, sc2, bi2, x, 64, out_dtype)


FAULTS = {
    "bit-map shift by one code": fault_bitmap_shift,
    "bias term dropped": fault_drop_bias,
    "group index off by one": fault_group_off_by_one,
    "fp16 accumulation": fault_fp16_accumulate,
    "unpack mask one bit wide": fault_wide_mask,
    "all groups use group 0": fault_scale_only_first_group,
}


def run(dtype, out_dtype, label):
    q, sc, bi, x = build_case(dtype)
    ref = dequantized(q, sc, bi) @ x.astype(np.float64)
    scale = float(np.max(np.abs(ref)))

    def err(v):
        return float(np.max(np.abs(v.astype(np.float64) - ref)))

    device_members = {
        "blocked 32": impl_blocked(q, sc, bi, x, 32, out_dtype),
        "blocked 64": impl_blocked(q, sc, bi, x, 64, out_dtype),
        "blocked 128": impl_blocked(q, sc, bi, x, 128, out_dtype),
        "pairwise tree": impl_tree(q, sc, bi, x, out_dtype),
        "fused per group": impl_fused_group(q, sc, bi, x, out_dtype),
    }
    floor_device = max(err(v) for v in device_members.values())

    prng = np.random.default_rng(SEED + 1)
    class_errs = [floor_device]
    ident = np.arange(KDIM)
    w32 = dequantized(q, sc, bi, np.float32)
    prod = (w32 * x.astype(np.float32)[None, :]).astype(np.float32)
    for perm in ([ident, ident[::-1],
                  np.argsort(np.abs(prod[0])), np.argsort(-np.abs(prod[0]))]
                 + [prng.permutation(KDIM) for _ in range(N_PERMS)]):
        class_errs.append(err(impl_permuted(q, sc, bi, x, perm, out_dtype)))
    floor_class = max(class_errs)

    print(f"\n=== {label} (output scale {scale:.1f}) ===")
    print(f"  floor over device-realisable members : {floor_device:.4e}")
    print(f"  floor over the admissible class      : {floor_class:.4e}"
          f"   ({floor_class/floor_device:.1f}x wider)")

    fault_errs = {name: err(fn(q, sc, bi, x, out_dtype))
                  for name, fn in FAULTS.items()}

    for base_tol in (0.0, 1e-4, 1e-3, 1e-2, 5e-2):
        tol_d = max(base_tol, K_ENSEMBLE * floor_device)
        tol_c = max(base_tol, K_ENSEMBLE * floor_class)
        caught_d = sum(e > tol_d for e in fault_errs.values())
        caught_c = sum(e > tol_c for e in fault_errs.values())
        lost = [n for n, e in fault_errs.items() if e > tol_d and e <= tol_c]
        print(f"  base_tol {base_tol:<7g} tol_device {tol_d:.3e}  "
              f"tol_class {tol_c:.3e}  caught {caught_d}/{len(FAULTS)} vs "
              f"{caught_c}/{len(FAULTS)}"
              + (f"  LOST: {', '.join(lost)}" if lost else "  no loss"))

    print("  fault errors:")
    for name, e in sorted(fault_errs.items(), key=lambda kv: -kv[1]):
        print(f"    {name:28} {e:.4e}   ({e/scale:.2e} relative)")


def main():
    print(f"mlx {mx.__version__}, numpy only past the quantizer call")
    print(f"K = {K_ENSEMBLE}, {ROWS} rows, K dim {KDIM}, "
          f"{BITS}-bit group {GROUP}, {N_PERMS} random permutations")
    run(np.float32, np.float32, "fp32 weights, fp32 output")
    run(np.float16, np.float16, "fp16 weights, fp16 output")


if __name__ == "__main__":
    main()
