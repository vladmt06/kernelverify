"""What one BUILD of an arm costs, against what clause 46 assumed it costs.

Amendment 10 clause 46 prices a build at three warm-up steps and prices those
steps at the timed-step rate, giving 890 seconds for one build of every arm
in both contexts. It separately lists graph compilation as an item the
subtotal does not carry.

Those two are not independent. `prepare_arm` traces at the FIRST warm-up
call, so compilation happens INSIDE warm-up one and is not a separate item at
all: it is a multiplier on the first of the three steps clause 46 counted at
the plain rate. If warm-up one is much more expensive than a timed step, the
890 seconds is a floor rather than the figure.

This times the three warm-ups separately, and the construction around them,
on freshly built arms.

    python bench/mlx_probes/probe_build_cost.py [builds]
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

import profile_knobs as pk  # noqa: E402
from probe_block_lifecycle import rig, stock_arm  # noqa: E402


def build_phases(model, optimizer, state, batch, arm):
    """prepare_arm's own body, with a clock between its phases.

    Copied rather than instrumented in place, because a timer inside the
    production builder would be a measurement device in the thing measured.
    """
    from metalrunner import seams

    installation = seams.Installation()
    t0 = time.perf_counter()
    try:
        for seam, wrap in arm.seams.items():
            installation.install(seam, wrap)
        step, traces = pk.build_step(model, optimizer, state)
        t1 = time.perf_counter()
        compiled = pk.CompiledArm(label=arm.label, phi=arm.phi, step=step,
                                  state=state, traces=traces,
                                  traced_by_warmup=0)
        warmups = []
        for _ in range(pk.WARMUPS):
            start = time.perf_counter()
            pk.one_step(compiled, batch)
            warmups.append((time.perf_counter() - start) * 1000.0)
    finally:
        installation.remove()
    compiled.traced_by_warmup = len(traces)
    total = (time.perf_counter() - t0) * 1000.0
    return compiled, {"construct_ms": (t1 - t0) * 1000.0,
                      "warmups_ms": warmups, "total_ms": total,
                      "traces": len(traces)}


def main() -> int:
    builds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    model, optimizer, state, batch = rig()
    print(f"{builds} fresh builds, {pk.WARMUPS} warm-ups each, compiled")

    held, rows = [], []
    for b in range(builds):
        compiled, row = build_phases(model, optimizer, state, batch,
                                     stock_arm(f"B{b}"))
        held.append(compiled)
        rows.append(row)
        w = row["warmups_ms"]
        print(f"  build {b:2d}  construct {row['construct_ms']:7.3f}   "
              f"warm-ups " + "  ".join(f"{v:8.3f}" for v in w) +
              f"   total {row['total_ms']:8.3f} ms   traces {row['traces']}")

    # A timed round on the arms now that every one of them is warm.
    samples = pk.timed_rounds(held, batch, rounds=9)
    step_ms = statistics.median(
        v * 1000.0 for vals in samples.values() for v in vals)

    first = statistics.median(r["warmups_ms"][0] for r in rows)
    rest = statistics.median(v for r in rows for v in r["warmups_ms"][1:])
    total = statistics.median(r["total_ms"] for r in rows)
    print(f"\n  timed step, median over {len(held)} warm arms   {step_ms:8.3f} ms")
    print(f"  warm-up 1 (the one that traces and compiles) {first:8.3f} ms  "
          f"= {first / step_ms:6.2f} timed steps")
    print(f"  warm-ups 2 and 3, median                     {rest:8.3f} ms  "
          f"= {rest / step_ms:6.2f} timed steps")
    print(f"  one whole build, median                      {total:8.3f} ms  "
          f"= {total / step_ms:6.2f} timed steps")
    print(f"\nclause 46 prices a build at 3.00 timed steps; measured here it is "
          f"{total / step_ms:.2f}, a factor of {total / (3 * step_ms):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
