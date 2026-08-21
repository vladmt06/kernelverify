"""Does an independent calibration block have to REBUILD its arms?

Why this exists
---------------
Amendment 10 clause 46 prices the calibration's timed rounds at 16.07 hours
and then says, in writing, that it does not know whether a block rebuilds its
arms or reuses them, and that the answer moves the subtotal from 16.32 hours
to 25.71. It registers that as an obligation on step 10 rather than guessing.
This is the measurement that discharges it.

What the two designs actually differ in
---------------------------------------
A null block's contrast is between two arms that ought to agree. The demand
`C` is an order statistic over block contrasts, and it has to control a
FUTURE null exceedance: the binding run builds its own arms once and times
them, so the binding run's contrast carries exactly one draw of whatever
varies at build time, plus round noise.

    REUSE     all blocks share one pair of builds. Every block contrast
              carries the SAME build offset, so the offset enters `R` as a
              constant and never enters the scatter that sets `C`.
    REBUILD   every block owns its builds, so block contrasts scatter over
              the build-offset distribution the binding run will draw from.

So the whole question is empirical and narrow: does one build of a step have
a persistent time offset against another build of the SAME step? If it does
not, reuse is sound and the budget is 16.32 hours. If it does, a reuse
calibration measures one draw of an offset and reports it as a distribution,
and the budget is 25.71.

The arrangement
---------------
Four identical stock arms, no seam anywhere, all interleaved in one set of
rounds so drift is common to all four:

    P1, P2   built once before the first block and never rebuilt
    F1, F2   rebuilt at the start of every block

Both pairs are read in the same rounds, so the reuse and the rebuild designs
are compared on one machine at one moment rather than across two runs.

    python bench/mlx_probes/probe_block_lifecycle.py [blocks] [rounds]
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402
from mlx_lm.utils import load  # noqa: E402

import profile_knobs as pk  # noqa: E402

MODEL = ROOT / "bench" / ".models" / "qwen3-0.6b-4bit-g64"
ADAPTED_LAYERS = 4
BATCH, WIDTH = 2, 97
ALPHA = 0.05


def rig():
    model, _ = load(str(MODEL))
    model.freeze()
    linear_to_lora_layers(model, ADAPTED_LAYERS,
                          {"rank": 8, "scale": 20.0, "dropout": 0.0})
    model.train()
    optimizer = optim.Adam(learning_rate=1e-5)
    pk.settle_optimizer(model, optimizer)
    mx.random.seed(0)
    tokens = mx.random.randint(0, 1000, (BATCH, WIDTH))
    lengths = mx.repeat(mx.array([[WIDTH // 2, WIDTH - 1]], dtype=mx.int32),
                        BATCH, axis=0)
    mx.eval(tokens, lengths)
    state = [model.state, optimizer.state, mx.random.state]
    return model, optimizer, state, (tokens, lengths)


def stock_arm(label: str) -> pk.Arm:
    """Identical to every other arm here: no seam, phi 1, a fresh closure."""
    return pk.Arm(label=label, phi=1.0, seams={})


def critical(values, alpha: float = ALPHA) -> float:
    """Clause 33's estimator: the ceil((m+1)(1-alpha))-th smallest."""
    import math
    ordered = sorted(values)
    m = len(ordered)
    index = min(m, math.ceil((m + 1) * (1 - alpha)))
    return ordered[index - 1]


def main() -> int:
    blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 9

    model, optimizer, state, batch = rig()
    print(f"model {MODEL.name}, batch {BATCH}, width {WIDTH}, "
          f"{blocks} blocks of {rounds} rounds, compiled")

    persistent = [pk.prepare_arm(model, optimizer, state, batch, stock_arm(n))
                  for n in ("P1", "P2")]

    reuse, rebuild, cross, fresh_minus_persistent = [], [], [], []
    per_round_pp, per_round_ff = [], []
    for b in range(blocks):
        fresh = [pk.prepare_arm(model, optimizer, state, batch,
                                stock_arm(f"F1-{b}"))
                 if i == 0 else
                 pk.prepare_arm(model, optimizer, state, batch,
                                stock_arm(f"F2-{b}"))
                 for i in range(2)]
        arms = [persistent[0], fresh[0], persistent[1], fresh[1]]
        samples = pk.timed_rounds(arms, batch, rounds=rounds)
        ms = {k: [v * 1000.0 for v in vals] for k, vals in samples.items()}
        p1, p2 = ms["P1"], ms["P2"]
        f1, f2 = ms[f"F1-{b}"], ms[f"F2-{b}"]
        reuse.append(abs(statistics.median(p1) - statistics.median(p2)))
        rebuild.append(abs(statistics.median(f1) - statistics.median(f2)))
        cross.append(abs(statistics.median(p1) - statistics.median(f1)))
        fresh_minus_persistent.append(
            statistics.median(f1 + f2) - statistics.median(p1 + p2))
        per_round_pp.extend(a - c for a, c in zip(p1, p2))
        per_round_ff.extend(a - c for a, c in zip(f1, f2))
        print(f"  block {b:2d}  reuse |P1-P2| {reuse[-1]:7.3f}   "
              f"rebuild |F1-F2| {rebuild[-1]:7.3f}   "
              f"|P1-F1| {cross[-1]:7.3f}   "
              f"fresh-persistent {fresh_minus_persistent[-1]:+7.3f} ms")
        del fresh

    def line(name, values):
        print(f"{name:>28}  R (median) {statistics.median(values):7.3f}   "
              f"C (alpha {ALPHA}) {critical(values):7.3f}   "
              f"min {min(values):7.3f}  max {max(values):7.3f}")

    print()
    line("reuse block contrasts", reuse)
    line("rebuild block contrasts", rebuild)
    line("persistent vs fresh", cross)

    pos = sum(1 for d in per_round_pp if d > 0)
    print(f"\nper-round sign test, P1 faster than P2 in "
          f"{len(per_round_pp) - pos} of {len(per_round_pp)} rounds "
          f"(a persistent offset shows as a lopsided count)")
    pos_f = sum(1 for d in per_round_ff if d > 0)
    print(f"per-round sign test, F1 faster than F2 in "
          f"{len(per_round_ff) - pos_f} of {len(per_round_ff)} rounds")
    print(f"\nfresh minus persistent, median over blocks "
          f"{statistics.median(fresh_minus_persistent):+.3f} ms, "
          f"positive in {sum(1 for d in fresh_minus_persistent if d > 0)} "
          f"of {blocks} blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
