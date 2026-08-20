"""Is a quantized matmul's cost a smooth dial on its input width, or a staircase?

Why this question decides candidate Q's share
---------------------------------------------
An ablation measures what the step stops paying when a region stops
happening. For candidate A that is the right question, because the sprint
could plausibly replace attention outright. For candidate Q it is the wrong
one: the sprint would RETUNE the quantized matmul, not delete it. A retuned
kernel still issues one dispatch per call site, still reads every 4-bit weight
byte, and still pays the host-side graph construction of its own operations.
A stub deletes all three, so a stub-based share credits candidate Q with time
no kernel could ever win back, at 197 call sites per step.

The fix is to move the knob from presence to size: shrink the arithmetic the
matmul does while holding the launch count, the output shape and the graph
structure fixed, then fit

    T(phi) = a + b * phi

and read the share off the SLOPE rather than an endpoint. The intercept `a`
holds launch and encode and host cost, which a retune keeps. The slope `b`
holds the arithmetic and the weight traffic, which is what a retune attacks.

That only works if the knob is a dial. A quantized kernel tiles its input, so
its cost can be a step function of the tile count rather than a line, and a
line fitted through a staircase is a number with no meaning. This probe checks
that before anything is built on it, and it needs no model.

    python bench/mlx_probes/probe_qmm_knob.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from interleave import MAX_CANARY_SPREAD, calibrate_copies, dispatch  # noqa: E402

BITS, GROUP = 4, 64
# The 0.6B's five distinct projection shapes plus the tied output head, as
# (out, in). Both dimensions matter: the head is 148 times wider than a
# projection and may well tile differently.
SHAPES = [(2048, 1024), (1024, 1024), (1024, 2048), (3072, 1024),
          (1024, 3072), (151936, 1024)]
# Batch 2 at width 97 gives 96 trained positions per row, so M = 192.
M = 192
PHIS = (1.0, 0.75, 0.5, 0.25)
ROUNDS = 5


def sliced(weight, phi: float):
    """The same weight with a contiguous prefix of its input dimension.

    Quantized along the last axis, so the packed width and the scale and bias
    columns all shrink together and the result is a valid quantized operand
    rather than a reinterpretation of one.
    """
    out_dims, in_dims = weight.shape
    in_k = max(GROUP, int(round(phi * in_dims / GROUP)) * GROUP)
    part = mx.contiguous(weight[:, :in_k])
    quantized, scales, biases = mx.quantize(part, group_size=GROUP, bits=BITS)
    mx.eval(quantized, scales, biases)
    return in_k, quantized, scales, biases


def fit(phis, times):
    """Least squares slope, intercept and R-squared on four points."""
    n = len(phis)
    mean_x = sum(phis) / n
    mean_y = sum(times) / n
    sxx = sum((x - mean_x) ** 2 for x in phis)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(phis, times))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    total = sum((y - mean_y) ** 2 for y in times)
    residual = sum((y - (intercept + slope * x)) ** 2
                   for x, y in zip(phis, times))
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return slope, intercept, r2


def main() -> int:
    print(f"M = {M}, bits {BITS}, group {GROUP}, {ROUNDS} interleaved rounds\n")
    print(f"{'shape':>16} {'phi=1.00':>9} {'0.75':>9} {'0.50':>9} {'0.25':>9} "
          f"{'R2':>7} {'slope/T(1)':>11} {'worst resid':>12}")

    for out_dims, in_dims in SHAPES:
        mx.random.seed(0)
        weight = mx.random.normal((out_dims, in_dims)).astype(mx.bfloat16)
        x = mx.random.normal((M, in_dims)).astype(mx.bfloat16)
        mx.eval(weight, x)

        arms = {}
        for phi in PHIS:
            in_k, quantized, scales, biases = sliced(weight, phi)
            xs = mx.contiguous(x[:, :in_k])
            mx.eval(xs)
            arms[phi] = (lambda _i, q=quantized, s=scales, b=biases, xx=xs:
                         mx.quantized_matmul(xx, q, scales=s, biases=b,
                                             transpose=True, group_size=GROUP,
                                             bits=BITS))

        copies = calibrate_copies(lambda c: dispatch(arms[1.0], c))
        for build in arms.values():                       # warm every arm
            dispatch(build, copies)
        samples = {phi: [] for phi in PHIS}
        for _ in range(ROUNDS):
            for phi, build in arms.items():
                samples[phi].append(dispatch(build, copies) / copies)
        median = {phi: statistics.median(rows)
                  for phi, rows in samples.items()}

        reference = samples[1.0]
        canary = max(reference) / min(reference)
        if canary > MAX_CANARY_SPREAD:
            print(f"{out_dims}x{in_dims:>6} REJECTED, canary spread "
                  f"{canary:.2f}x > {MAX_CANARY_SPREAD}x")
            continue

        times = [median[phi] for phi in PHIS]
        slope, intercept, r2 = fit(list(PHIS), times)
        worst = max(abs(y - (intercept + slope * x)) for x, y in
                    zip(PHIS, times)) / abs(slope) if slope else float("nan")
        row = " ".join(f"{median[phi] * 1e6:>9.2f}" for phi in PHIS)
        print(f"{f'{out_dims}x{in_dims}':>16} {row} {r2:>7.4f} "
              f"{slope / median[1.0]:>11.4f} {worst:>12.4f}", flush=True)

    print("\nR-squared near 1 and every residual small against the slope means "
          "the knob is a cost dial and candidate Q's share can be read off the "
          "slope, holding the launch count fixed.")
    print("A knee, a staircase, or flat points at small phi mean the knob "
          "quantises against the kernel's tiling, and Q keeps only a stub "
          "endpoint, which is an upper bound rather than a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
