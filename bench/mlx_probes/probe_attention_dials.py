"""Which of the three attention dials measures candidate A, by the registered rule.

Why attention needs a choice at all
-----------------------------------
Every other candidate has one obvious dial. A quantized projection shrinks on
its input width; the output head and the loss shrink together on the
vocabulary. Attention is two matmuls with a softmax between them, and no
single dial shrinks all three without changing something else, so Amendment 5
clause 15 registers THREE dials, a price statistic, and a tie-break, all
written before any of the three was measured.

    rank 1   key and value sequence length, hand-built mask at every arm
             both matmuls and the softmax move
    rank 2   head dimension on queries, keys and values, output padded back
             both matmuls move
    rank 3   head dimension on queries and keys only
             the first matmul moves

The price is the fitted scaffold slope as a fraction of that dial's own knob
slope, and the smallest wins. Two prices within 0.01 of each other are a tie
and a tie goes to the higher rank. A dial that cannot place three distinct
settings on the ladder is not eligible whatever its price.

Why the price is a slope and not one number
-------------------------------------------
A scaffold whose own cost CHANGES with the dial enters the fitted slope, which
is the number the share is read off. Checking the scaffold at one setting
cannot see that. So each dial's scaffold arm is fitted over the same ladder as
its knob, and what is compared is slope against slope.

What each scaffold arm can and cannot price, stated because it decides things
-----------------------------------------------------------------------------
The length dial's machinery is an array mask, and the array mask's cost is
proportional to the score matrix it covers. Its scaffold arm applies the array
mask at FULL size, so what it prices lands in the OFFSET; the part that moves
with the dial sits inside the knob slope and the slope statistic cannot
separate it. This probe therefore reports that dial's offset as a fraction of
its slope alongside the registered price, as a diagnostic the criterion does
not read.

The head-dimension dial's machinery is a pad that GROWS as the dial shrinks,
so its scaffold arm writes exactly as many padded columns as the real arm and
throws them away. At the full setting it writes none, which is the one arm of
that dial structurally unlike the rest.

The queries-and-keys dial has no scaffold at all beyond two slices, which a
lazy graph drops because nothing reads them.

None of the three scaffold arms prices the operand slice itself, because a
dead slice is eliminated before it costs anything. That is a known limit of
clause 5's scaffold as registered, and the arithmetic bounds it: a slice
writes as many elements as the operand, and the matmul it feeds does a factor
of the sequence length more work, so at width 96 the slice is about one
percent of what it is being compared against.

What it measured, 2026-08-20, and why the answer is not yet usable
------------------------------------------------------------------
Qwen3-0.6B, 4 adapted layers, batch 2, five rounds, three warm-ups, compiled,
all 28 arms interleaved inside one set of rounds.

    queries   kv-length   head-dim-qkv   head-dim-qk   chosen
       96       0.101        0.937          0.440      kv-length, on price
      384       0.221        0.027          0.029      head-dim-qkv, on rank

The criterion returns a different dial at each width, and clause 15 registers
no width. The dial is not a presentation choice: at 384 queries the length
dial reads a share of 0.060 and both head-dimension dials read 0.023, against
an unmarked ablation of 0.068 in the same rounds. So at that width the
criterion prefers a dial reading a third of what removing attention outright
costs.

The reason nothing separates cleanly is that attention is 2 to 6 percent of
the step at these widths, so the dial moves about two milliseconds out of 88,
or 23 out of 390, and every scaffold slope measured so far sits inside the
machine's own jitter. The best knob fit reached an R-squared of 0.946 where
clause 6's gate demands 0.99, so all three dials would be refused as readings
even while the criterion happily ranks them.

The width where this becomes measurable is the long registered one, where
attention's score matrix is quadratic and everything else is linear and its
share reaches roughly a fifth of the step. Running there is minutes rather
than seconds, so it takes a granted window.

    python bench/mlx_probes/probe_attention_dials.py
    python bench/mlx_probes/probe_attention_dials.py --width 385
"""

from __future__ import annotations

import argparse
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
from memory_guard import BudgetGuard, require_available_memory  # noqa: E402
from probe_attention_ablation import MODEL  # noqa: E402

BATCH = 2
ADAPTED_LAYERS = 4
BUDGET_GB = 14.0


def rig(width: int):
    """One model, one settled optimizer, one fixed batch.

    The optimizer is settled before any arm is built, which is clause 25's
    second requirement: Adam allocates its moments on the first update, and
    that growth forces a second trace which could land after a seam is gone.
    """
    model, _ = load(str(MODEL))
    model.freeze()
    linear_to_lora_layers(model, ADAPTED_LAYERS,
                          {"rank": 8, "scale": 20.0, "dropout": 0.0})
    model.train()
    optimizer = optim.Adam(learning_rate=1e-5)
    pk.settle_optimizer(model, optimizer)

    mx.random.seed(0)
    tokens = mx.random.randint(0, 1000, (BATCH, width))
    lengths = mx.repeat(mx.array([[width // 2, width - 1]], dtype=mx.int32),
                        BATCH, axis=0)
    mx.eval(tokens, lengths)
    state = [model.state, optimizer.state, mx.random.state]
    return model, optimizer, state, (tokens, lengths)


def build_arms(model, optimizer, state, batch, width: int) -> tuple[list, dict]:
    """Every arm of all three dials, plus one stock arm they all share.

    Built together and timed together on purpose. The three prices are
    compared against EACH OTHER, so timing one dial's arms in their own set of
    rounds would let the machine drift between the sets and put that drift
    inside the comparison. One stock arm serves all three for the same reason.
    """
    arms = [pk.prepare_arm(model, optimizer, state, batch,
                           pk.Arm(label="stock", phi=1.0))]
    plan: dict[str, dict] = {}
    for dial in pk.ATTENTION_DIALS:
        knob = pk.ATTENTION_KNOBS[dial]
        prepared = knob.prepare(model, width)
        settings = pk.realisable_settings(prepared["attention"])
        entry = {"settings": settings, "knob": {}, "scaffold": {}}
        for phi in settings:
            for kind, build in (("knob", knob.arm), ("scaffold", knob.scaffold)):
                label = f"{dial}/{kind}@{phi:.2f}"
                arms.append(pk.prepare_arm(
                    model, optimizer, state, batch,
                    pk.Arm(label=label, phi=phi, seams=build(prepared, phi))))
                entry[kind][label] = phi
        label = f"{dial}/ablated"
        arms.append(pk.prepare_arm(
            model, optimizer, state, batch,
            pk.Arm(label=label, phi=0.0, seams=knob.ablate(prepared))))
        entry["ablated"] = label
        plan[dial] = entry
    return arms, plan


def price_one_dial(samples, median, plan, dial: str) -> dict:
    """One dial's knob line, scaffold line and price, off the shared rounds."""
    entry = plan[dial]
    knob_fit = pk.pooled_fit(samples, entry["knob"])
    scaffold_fit = pk.pooled_fit(samples, entry["scaffold"])
    top = max(entry["settings"])
    offset = median[f"{dial}/knob@{top:.2f}"] - median["stock"]

    return {
        "dial": dial,
        "settings": len(entry["settings"]),
        "knob": knob_fit,
        "scaffold": scaffold_fit,
        "offset": offset,
        "price": abs(scaffold_fit.slope) / abs(knob_fit.slope),
        "offset_over_slope": abs(offset) / abs(knob_fit.slope),
        "residue": knob_fit.intercept - median[entry["ablated"]],
        "share": knob_fit.slope / median["stock"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=97,
                        help="batch width in tokens; attention sees one less")
    parser.add_argument("--rounds", type=int, default=pk.ROUNDS)
    args = parser.parse_args()

    guard = BudgetGuard(BUDGET_GB)
    guard.check("startup")
    require_available_memory(BUDGET_GB, "startup")

    model, optimizer, state, batch = rig(args.width)
    print(f"Qwen3-0.6B, {ADAPTED_LAYERS} adapted layers, batch {BATCH} x "
          f"{args.width} tokens, attention sees "
          f"{pk.attention_width(args.width)} queries, {args.rounds} rounds, "
          f"{pk.WARMUPS} warm-ups, compiled")

    arms, plan = build_arms(model, optimizer, state, batch, args.width)
    guard.check("arms built")
    samples = pk.timed_rounds(arms, batch, rounds=args.rounds)
    median = {label: statistics.median(times)
              for label, times in samples.items()}
    print(f"    stock {median['stock'] * 1e3:.2f} ms over "
          f"{len(arms)} interleaved arms")

    readings = []
    for rank, dial in enumerate(pk.ATTENTION_DIALS, start=1):
        reading = price_one_dial(samples, median, plan, dial)
        readings.append(reading)
        print(f"\n--- rank {rank}: {dial} "
              f"({reading['settings']} realisable settings)")
        for label in sorted(l for l in median if l.startswith(f"{dial}/")):
            print(f"    {label:<28} {median[label] * 1e3:8.2f} ms")
        knob, scaffold = reading["knob"], reading["scaffold"]
        print(f"    knob      slope {knob.slope * 1e3:8.2f} ms  "
              f"intercept {knob.intercept * 1e3:8.2f} ms  "
              f"R-squared {knob.r_squared:.4f}  "
              f"worst residual {abs(knob.max_residual) / abs(knob.slope):.4f} "
              f"of slope")
        print(f"    scaffold  slope {scaffold.slope * 1e3:8.2f} ms  "
              f"R-squared {scaffold.r_squared:.4f}")
        print(f"    PRICE {reading['price']:.4f}  "
              f"(offset {reading['offset'] * 1e3:+.2f} ms, "
              f"{reading['offset_over_slope']:.4f} of slope)")
        print(f"    share by slope {reading['share']:.4f}  "
              f"residue {reading['residue'] * 1e3:+.2f} ms")

    prices = {r["dial"]: r["price"] for r in readings}
    settings = {r["dial"]: r["settings"] for r in readings}
    ruling = pk.choose_dial(prices, pk.ATTENTION_DIALS, settings)

    print("\n=== Amendment 5 clause 15, applied")
    for dial in pk.ATTENTION_DIALS:
        mark = "  <- chosen" if dial == ruling["chosen"] else ""
        print(f"    {dial:<14} price {prices[dial]:.4f}  "
              f"settings {settings[dial]}{mark}")
    print(f"    tied within {pk.DIAL_TIE}: {ruling['tied']}")
    print(f"    refused for too few settings: {ruling['refused'] or 'none'}")
    print(f"    decided by completeness rather than by price: "
          f"{ruling['by_completeness']}")
    print(f"    CANDIDATE A'S DIAL IS {ruling['chosen']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
