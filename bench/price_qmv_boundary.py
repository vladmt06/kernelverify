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
coordinated quiet slot, one measuring lane at a time. In code, the machine
discipline is the shared vocabulary (bench/memory_guard.py and
bench/machine_state.py, single-sourced under ruling D1, never copied):

- the ONE machine-wide measurement lock is taken for the life of the run,
  BEFORE the idle gate samples - two co-armed idle gates are how the 03:29
  two-harness collapse started;
- a phys_footprint budget and a low-memory check run between verification
  tile-widths and between timing rounds, through the guard-callback seam of
  bench/pack_wide_qmv.py; the checks may NOT live in a private copy of
  those loops, because a sampler that diverges from the gate's is the
  ADR 0004 two-halves mistake;
- a guard refusal exits with the vocabulary's own code and writes NOTHING.
  The probe has no checkpoint: its samples are worthless unless the whole
  protocol completed on a clean machine, so discarding the run and the slot
  on refusal is INTENDED, not a port omission from the serving harness.

Boundary interpretation is PRE-REGISTERED (ruling D3, decided 2026-08-15
before any number existed): a boundary cell that comes back REFUSED or
REJECTED under the probe's own interval rule is DROPPED from the routed
window - the window may narrow, it never keeps an unevidenced cell. The
reconciliation step reads this rule from here; MIN/MAX_PROFITABLE_M stay
at the shipped 5..11 default until that step runs against the recorded
numbers.

The full samples, spreads, machine state and shape provenance land in
bench/results/, committed, so the boundary the pricing justifies can cite
a file rather than a memory. A run that went busy under the closing idle
gate lands under a REFUSED quarantine name instead, and binds nothing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

import machine_state  # noqa: E402
import pack_wide_qmv as gate  # noqa: E402
from machine_state import MeasurementLock  # noqa: E402

# The refusal vocabulary, single-sourced (D1): the exit codes, footprint
# reader and low-memory check are the SAME objects every heavy harness uses,
# never a probe-local copy with its own numbering.
from memory_guard import (  # noqa: E402
    EXIT_BUDGET_REFUSAL,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    BudgetExceeded,
    BudgetGuard,
    LowMemoryRefusal,
    machine_ram_gb,
    phys_footprint_gb,
    positive_float_arg,
    require_available_memory,
)

# Single-sourced from the gate module: the constants the probe records must
# be the constants the shared sampler enforces (a recorded protocol that
# differs from the enforced one is the ADR 0004 two-halves mistake again).
from pack_wide_qmv import MAX_CANARY_SPREAD, ROUNDS  # noqa: E402

from kernelverify.pack.wide_qmv import build, launch_config  # noqa: E402
from kernelverify.runners import MetalRunner  # noqa: E402

BITS = 3                      # the block's arm (D1); 2-bit is cut (D4)
M_SWEEP = tuple(range(1, 17))  # the serving range B = 1..16
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The pricing budget, from the 2026-08-15 instrumentation passes on this
# machine (bench/price_qmv_boundary.py --instrument, lock held, foreground):
# the completed pass measured a 22.50 GB lifetime peak phys_footprint
# (checkpointed peak identical; the peak is the lm_head verification group,
# whose fp64/fp32 whole-matrix transients at 151936x2560 dominate) with an
# 8.14 GB MLX allocator-cache peak accumulated across the six shapes'
# rotated timing working sets. 26.0 = 22.50 + ~15% margin - deliberately
# NOT the serving calibration's 24 GB, which describes a different
# harness's allocations. mx.get_cache_memory() is recorded beside the
# footprint at every guard checkpoint so a refusal can be told apart from
# legitimate cache growth after the fact. (The first pass refused at
# 22.14 GB under its own protective cap: the then-unchunked serial ensemble
# members' (m, d_out, d_in) products were ratcheting malloc's dirty pages
# per tile width - that holder is fixed, and this budget describes the
# fixed code.)
PRICING_BUDGET_GB = 26.0

# The instrumentation pass is bounded twice: by a protective footprint cap
# (half the machine by default, a safety net rather than a claim) and by a
# wall cap, so an unexpectedly slow verification cannot occupy the machine
# open-endedly. The wall cap is sized to cover the lm_head verification
# group, whose 32 fp64 references at 151936x2560 dominate the pass.
INSTRUMENT_WALL_CAP_S = 3600.0


class WallCapExceeded(RuntimeError):
    """The instrumentation pass ran past its wall cap: stop measuring."""


class PricingGuard:
    """The probe's memory checks, entered through the shared gate's
    guard-callback seam between verification tile-widths and timing rounds.

    Checks, in order: this process's phys_footprint against the budget
    (BudgetExceeded); the machine's available memory against the room this
    run may still legitimately grow into, budget minus current footprint
    (LowMemoryRefusal) - the remaining headroom, not the full budget, or the
    check would refuse mid-run as our own growth eats the machine's number.
    mx.get_cache_memory() is recorded beside the footprint at every
    checkpoint. A refusal discards the run: no checkpoint, nothing written.
    """

    def __init__(self, budget_gb: float, instrument: bool = False,
                 wall_cap_s: float | None = None):
        self.budget = BudgetGuard(budget_gb)
        self.instrument = instrument
        self.wall_cap_s = wall_cap_s
        self.started = time.monotonic()
        self.peak_seen_gb = 0.0
        self.peak_cache_gb = 0.0

    def __call__(self, cell: str) -> None:
        if (self.wall_cap_s is not None
                and time.monotonic() - self.started > self.wall_cap_s):
            raise WallCapExceeded(
                f"wall cap {self.wall_cap_s:.0f}s exceeded at cell {cell}")
        current = self.budget.check(cell)
        self.peak_seen_gb = max(self.peak_seen_gb, current)
        cache_gb = mx.get_cache_memory() / 1e9
        self.peak_cache_gb = max(self.peak_cache_gb, cache_gb)
        remaining = max(self.budget.budget_gb - current, 0.0)
        require_available_memory(remaining, cell)
        if self.instrument:
            print(f"  [instrument] {cell}: footprint {current:.2f} GB, "
                  f"mx cache {cache_gb:.2f} GB")


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


def _round_guard(guard, cell: str):
    """The cell's guard in the form the shared sampler calls: the sampler
    names the round, this names the cell that round belongs to. None stays
    None, so an unguarded run keeps the sampler's no-check path."""
    if guard is None:
        return None

    def check_round(round_label: str) -> None:
        guard(f"{cell} {round_label}")

    return check_round


def price_shape(kernel, shape, m_sweep, rounds, guard=None) -> list[PricedPoint]:
    """Interleaved ours-vs-stock pricing of one dispatch shape across M.

    ``guard`` runs before each cell and between the rounds inside it (through
    the shared sampler's own seam), so an over-budget or low-memory refusal
    lands between measurements, never inside one."""
    d_out, d_in = shape.d_out, shape.d_in
    sets = gate.weight_sets(d_out, d_in, BITS)

    points = []
    for m in m_sweep:
        cell = f"{shape.name} {d_out}x{d_in} M={m}"
        if guard is not None:
            guard(cell)
        cell_guard = _round_guard(guard, cell)
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

        a_samples, b_samples = gate.interleaved_samples(ours, theirs, rounds,
                                                        guard=cell_guard)
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


def instrument(kernel, shapes, guard: PricingGuard) -> int:
    """ONE bounded foreground instrumentation pass: the full verification
    sweep plus, per shape, the timing path's dominant allocations (the
    rotated working set and two sampled rounds at the widest tile), with
    footprint and mx cache printed at every guard checkpoint. Its output is
    a memory measurement for setting PRICING_BUDGET_GB - no number it
    produces is a timing claim. Nothing is recorded."""
    print("instrumentation pass: memory only, wall-capped "
          f"{guard.wall_cap_s:.0f}s, footprint cap "
          f"{guard.budget.budget_gb:.1f} GB (protective, not a claim)")
    if not gate.verify(MetalRunner(), e2e_m=list(M_SWEEP), guard=guard).ok:
        print("\nVERDICT: kernel does not verify; instrumentation stopped")
        return 1
    # No cache clearing between shapes, deliberately: the priced run frees
    # each shape's rotated working set into MLX's allocator cache and never
    # clears it, so instrumentation must let the cache accumulate the same
    # way or the budget it produces refuses the real run mid-window.
    m_widest = max(M_SWEEP)
    for shape in shapes:
        price_shape(kernel, shape, (m_widest,), rounds=2, guard=guard)
    current, lifetime_peak = phys_footprint_gb()
    print(f"\ninstrumented peak: footprint {lifetime_peak:.2f} GB lifetime "
          f"(checkpointed peak {guard.peak_seen_gb:.2f} GB, final "
          f"{current:.2f} GB), mx cache peak {guard.peak_cache_gb:.2f} GB")
    print("set PRICING_BUDGET_GB from this lifetime peak plus margin")
    return 0


def _budget_gb_arg(text: str) -> float:
    return positive_float_arg(text, "budget", "decimal GB")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="plumbing check: smallest shape, M in {1, 8}, "
                             "2 rounds, no verification, nothing recorded")
    parser.add_argument("--instrument", action="store_true",
                        help="bounded foreground memory instrumentation "
                             "(lock held, wall-capped, nothing recorded); "
                             "its measured peak sets PRICING_BUDGET_GB")
    parser.add_argument("--budget-gb", type=_budget_gb_arg, default=None,
                        help="phys_footprint budget in decimal GB; crossing "
                             f"it refuses and exits {EXIT_BUDGET_REFUSAL} "
                             f"(default {PRICING_BUDGET_GB} from the "
                             "instrumentation pass)")
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

    # The ONE machine-wide lock, taken before anything else samples the
    # machine: two heavy harnesses whose idle gates co-fired is exactly the
    # 03:29 collapse, so even the idle gate waits behind the lock.
    lock = MeasurementLock("price_qmv_boundary")
    acquired, detail = lock.acquire()
    if not acquired:
        print(f"REFUSAL (exit {EXIT_LOCK_HELD}): machine measurement lock "
              f"{detail}; one heavy measurement at a time")
        return EXIT_LOCK_HELD
    try:
        if args.instrument:
            budget = args.budget_gb or machine_ram_gb() / 2
            guard = PricingGuard(budget, instrument=True,
                                 wall_cap_s=INSTRUMENT_WALL_CAP_S)
            try:
                return instrument(kernel, shapes, guard)
            except (BudgetExceeded, WallCapExceeded, LowMemoryRefusal) as exc:
                print(f"\ninstrumentation stopped: {exc}")
                return 1

        fp = machine_state.fingerprint()
        before = machine_state.idle_check(fp["cores"])
        if not before["idle"]:
            print(f"machine not idle: {before}")
            print("boundary pricing refuses a busy machine; re-run in the "
                  "coordinated quiet slot")
            return 1

        return priced_run(kernel, shapes, fp, before,
                          args.budget_gb or PRICING_BUDGET_GB)
    finally:
        lock.release()


def priced_run(kernel, shapes, fp: dict, before: dict,
               budget_gb: float) -> int:
    """Verification, pricing and recording, inside the held lock.

    A guard refusal deliberately discards the run AND the slot: the probe
    has no checkpoint (unlike the serving calibration), because a partial
    pricing sweep supports no boundary claim - so nothing is written and
    the distinct exit code is the whole story. INTENDED, not an omission.
    """
    guard = PricingGuard(budget_gb)
    try:
        print("verification first: the pack gate must be green at EVERY tile "
              "width this probe times before any ratio is printed")
        if not gate.verify(MetalRunner(), e2e_m=list(M_SWEEP),
                           guard=guard).ok:
            print("\nVERDICT: kernel does not verify at the E2E shapes; no "
                  "pricing claim permitted")
            return 1

        points = []
        for shape in shapes:
            points.extend(price_shape(kernel, shape, M_SWEEP, ROUNDS,
                                      guard=guard))
    except BudgetExceeded as exc:
        print(f"\nREFUSAL (exit {EXIT_BUDGET_REFUSAL}): {exc}")
        print("run discarded (no checkpoint, nothing written); re-arm the "
              "slot after raising the budget or freeing the machine")
        return EXIT_BUDGET_REFUSAL
    except LowMemoryRefusal as exc:
        print(f"\nREFUSAL (exit {EXIT_LOW_MEMORY}): {exc}")
        print("run discarded (no checkpoint, nothing written); another "
              "process is eating the machine - free it and re-arm the slot")
        return EXIT_LOW_MEMORY
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
