"""Price the wide_qmv routing boundary at the E2E dispatch shapes.

The routing boundary (kernelverify/pack/wide_qmv.py, MIN/MAX_PROFITABLE_M)
ships as the 5..11 default ruled by D3.1 until this probe prices it where
the serving path actually dispatches: the six Qwen3-4B decode shapes, ours
against mx.quantized_matmul, M = 1..16, 3-bit (the block's arm, D1; 2-bit
is cut, D4).

Verification precedes timing, enforced: the extended pack gate must be
green in this same process before a single ratio is printed, exactly as
bench/pack_wide_qmv.py enforces for the microbenchmark shapes.

Timing discipline, inherited measurement by measurement:

- interleaved arms within every round, always (an A/B in separate passes
  measures the clock, not the kernels);
- batched dispatches of at least MIN_SAMPLE_MS, so nothing sub-millisecond
  is timed as an absolute claim;
- weights rotated over a working set larger than any cache;
- the reference arm's own spread is the excursion detector: past
  MAX_CANARY_SPREAD the point is REJECTED as the machine moving;
- a point whose ratio interval [min_mlx/max_ours, max_mlx/min_ours]
  contains 1.0 is REFUSED: its spread swallows its ratio, so it may not
  claim a direction (the outside voice measured per-op probes as volatile;
  every point states both arms' spreads for exactly this reason);
- the idle gate runs before and after, and the run refuses a busy or
  unplugged machine the way measure_baselines does.

Machine coordination is not code: this probe records timing only inside a
coordinated quiet slot, one measuring lane at a time.

The full samples, spreads, machine state and shape provenance land in
bench/results/, committed, so the boundary the pricing justifies can cite
a file rather than a memory.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

import machine_state  # noqa: E402
import pack_wide_qmv as gate  # noqa: E402

# Single-sourced from the gate module: the constants the probe records must
# be the constants the shared sampler enforces (a recorded protocol that
# differs from the enforced one is the ADR 0004 two-halves mistake again).
from pack_wide_qmv import MAX_CANARY_SPREAD, ROUNDS  # noqa: E402

from kernelverify.pack.wide_qmv import build, launch_config  # noqa: E402
from kernelverify.runners import MetalRunner  # noqa: E402

BITS = 3                      # the block's arm (D1); 2-bit is cut (D4)
M_SWEEP = tuple(range(1, 17))  # the serving range B = 1..16
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class PricedPoint:
    """One (shape, M) cell: both arms' samples and the verdict they earn."""

    site: str
    d_out: int
    d_in: int
    m: int
    r: int
    verdict: str               # WIN | LOSS | REFUSED | REJECTED
    ratio: float               # median(mlx) / median(ours); > 1 means we win
    ratio_lo: float            # worst pairing of the samples
    ratio_hi: float            # best pairing of the samples
    spread_ours: float         # max/min over rounds
    spread_mlx: float
    ours_us: float
    mlx_us: float
    samples_ours_us: tuple
    samples_mlx_us: tuple


def classify(ours: list, mlx: list, *, site: str = "", d_out: int = 0,
             d_in: int = 0, m: int = 0, r: int = 0) -> PricedPoint:
    """The verdict for one point, from the two arms' per-round samples.

    REJECTED: the reference arm's spread says the clock moved; nothing about
    the kernels can be read off this round. REFUSED: the ratio interval
    contains 1.0, so the point's own spread could explain its direction.
    Only an interval clear of 1.0 on one side may claim WIN or LOSS.
    """
    spread_ours = max(ours) / min(ours)
    spread_mlx = max(mlx) / min(mlx)
    ratio = statistics.median(mlx) / statistics.median(ours)
    ratio_lo = min(mlx) / max(ours)
    ratio_hi = max(mlx) / min(ours)
    if spread_mlx > MAX_CANARY_SPREAD:
        verdict = "REJECTED"
    elif ratio_lo > 1.0:
        verdict = "WIN"
    elif ratio_hi < 1.0:
        verdict = "LOSS"
    else:
        verdict = "REFUSED"
    return PricedPoint(
        site=site, d_out=d_out, d_in=d_in, m=m, r=r, verdict=verdict,
        ratio=ratio, ratio_lo=ratio_lo, ratio_hi=ratio_hi,
        spread_ours=spread_ours, spread_mlx=spread_mlx,
        ours_us=statistics.median(ours) * 1e6,
        mlx_us=statistics.median(mlx) * 1e6,
        samples_ours_us=tuple(s * 1e6 for s in ours),
        samples_mlx_us=tuple(s * 1e6 for s in mlx))


def price_shape(kernel, shape, m_sweep, rounds) -> list[PricedPoint]:
    """Interleaved ours-vs-stock pricing of one dispatch shape across M."""
    d_out, d_in = shape.d_out, shape.d_in
    sets = gate.weight_sets(d_out, d_in, BITS)

    points = []
    for m in m_sweep:
        x = mx.random.normal(shape=(m, d_in)).astype(mx.float16)
        mx.eval(x)
        grid, threadgroup, r = launch_config(d_out, m)

        def ours(i):
            wq, sc, bi = sets[i % len(sets)]
            return kernel(inputs=[x, wq, sc, bi],
                          output_shapes=[(m, d_out)],
                          output_dtypes=[mx.float16],
                          grid=grid, threadgroup=threadgroup,
                          template=[("T", mx.float16), ("M", m), ("R", r),
                                    ("BITS", BITS)])[0]

        def theirs(i):
            wq, sc, bi = sets[i % len(sets)]
            return mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                       group_size=64, bits=BITS)

        a_samples, b_samples = gate.interleaved_samples(ours, theirs, rounds)
        points.append(classify(a_samples, b_samples, site=shape.name,
                               d_out=d_out, d_in=d_in, m=m, r=r))
    return points


def render(points: list[PricedPoint]) -> None:
    site = None
    for p in points:
        if p.site != site:
            site = p.site
            print(f"\n  {site}  {p.d_out} x {p.d_in}, {BITS}-bit group 64")
            print(f"  {'M':>3} {'R':>3} {'ours us':>9} {'mlx us':>9} "
                  f"{'ratio':>7} {'interval':>15} {'spread o/m':>12} verdict")
        interval = f"[{p.ratio_lo:.2f},{p.ratio_hi:.2f}]"
        print(f"  {p.m:>3} {p.r:>3} {p.ours_us:>9.1f} {p.mlx_us:>9.1f} "
              f"{p.ratio:>6.2f}x {interval:>15} "
              f"{p.spread_ours:>5.2f}/{p.spread_mlx:<5.2f} {p.verdict}")


def summarize(points: list[PricedPoint]) -> None:
    """Per-shape win zones from decisive points only; refusals listed."""
    print("\nper-shape win zones (WIN points only; REFUSED points cannot "
          "support either side):")
    for site in dict.fromkeys(p.site for p in points):
        wins = [p.m for p in points if p.site == site and p.verdict == "WIN"]
        undecided = [p.m for p in points
                     if p.site == site and p.verdict in ("REFUSED", "REJECTED")]
        if wins:
            zone = f"M {min(wins)}..{max(wins)}"
            gaps = sorted(set(range(min(wins), max(wins) + 1)) - set(wins))
        else:
            zone = "none"
            gaps = []
        note = f", gaps {gaps}" if gaps else ""
        und = f", undecided {undecided}" if undecided else ""
        print(f"  {site:24s} {zone}{note}{und}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="plumbing check: smallest shape, M in {1, 8}, "
                             "2 rounds, no verification, nothing recorded")
    args = parser.parse_args(argv)

    shapes = gate.E2E_SHAPES
    print(f"E2E dispatch shapes from {gate.E2E_PROVENANCE}")
    kernel = build(mx)

    if args.smoke:
        smallest = min(shapes, key=lambda s: s.d_out * s.d_in)
        points = price_shape(kernel, smallest, (1, 8), rounds=2)
        render(points)
        print("\nSMOKE ONLY: no verification ran, nothing was recorded, and "
              "no number above is a claim.")
        return 0

    fp = machine_state.fingerprint()
    before = machine_state.idle_check(fp["cores"])
    if not before["idle"]:
        print(f"machine not idle: {before}")
        print("boundary pricing refuses a busy machine; re-run in the "
              "coordinated quiet slot")
        return 1

    print("verification first: the pack gate must be green at EVERY tile "
          "width this probe times before any ratio is printed")
    if not gate.verify(MetalRunner(), e2e_m=list(M_SWEEP)).ok:
        print("\nVERDICT: kernel does not verify at the E2E shapes; no "
              "pricing claim permitted")
        return 1

    points = []
    for shape in shapes:
        points.extend(price_shape(kernel, shape, M_SWEEP, ROUNDS))
    render(points)
    summarize(points)

    # The final gate runs BEFORE any write. A file under the canonical name
    # IS the claim that the machine stayed idle end to end; a run that went
    # busy quarantines its samples under a REFUSED name instead, so the
    # evidence survives for diagnosis without ever looking binding.
    after = machine_state.idle_check(fp["cores"])
    stamp = _dt.date.today().isoformat()
    suffix = ".json" if after["idle"] else ".REFUSED.json"
    out = RESULTS_DIR / f"qmv-boundary-pricing-{stamp}{suffix}"
    out.write_text(json.dumps({
        "probe": "price_qmv_boundary",
        "date": stamp,
        "bits": BITS,
        "rounds": ROUNDS,
        "min_sample_ms": gate.MIN_SAMPLE_MS,
        "working_set_mb": gate.WORKING_SET_MB,
        "max_canary_spread": MAX_CANARY_SPREAD,
        "shape_provenance": gate.E2E_PROVENANCE,
        "machine": fp,
        "idle_before": before,
        "idle_after": after,
        "points": [asdict(p) for p in points],
    }, indent=2))
    print(f"\nrecorded: {out}")
    if not after["idle"]:
        print("WARNING: machine went busy during the run; recording "
              "quarantined under a REFUSED name and binds nothing; re-run "
              "in the coordinated quiet slot")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
