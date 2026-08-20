"""The smallest paired difference in step time this harness can resolve.

Why this exists
---------------
`probe_attention_ablation.py` reports attention at 0.042 of the step by
differencing two arms, and that number is only worth its digits if the
difference is larger than what two IDENTICAL arms would show. Nobody had
measured that. The stock arm's median moved from 132.6 ms to 141.7 ms across
four separate runs of the same probe, so the across-run scatter is around 7%,
and a 0.042 share at width 97 is a delta of about 5.9 ms, which is 4% of the
step. Those two numbers are close enough together that the whole argument
rests on which of them is the right comparison.

So this runs the null experiment: the same arm against itself, interleaved
exactly the way the real arms are, and reports the difference that ought to be
zero.

Three arms, because there are two different things to separate
--------------------------------------------------------------
    stock-a / stock-b   the same absent installation twice. Their difference
                        is the floor: pure timing noise under interleaving,
                        with no seam and no replacement involved.
    stock-passthrough   a seam really installed, whose replacement calls the
                        original and returns it unchanged. Its difference from
                        stock-a is what INSTALLING costs, separately from what
                        any replacement does. A wrapper adds a Python frame
                        per call, 28 times per step here, and the ablation
                        arms all pay it while the stock arm does not.

Without the third arm, the seam's own cost would sit inside every ablation
delta and be reported as part of the region.

What the number has to clear
----------------------------
Section 4.3 calls a difference under two percentage points of gain a tie.
Differentiating the registered formula,

    dgain/df = (1 - 1/r) / (1 - f*(1 - 1/r))^2

so the resolution in f that a 0.02 tie band demands is about 0.008 at
candidate Q's likely regime, which on a 133 ms step is about 1 ms. That is the
binding requirement, and it is derived from the pre-registration rather than
chosen here.

    python bench/mlx_probes/probe_step_resolution.py
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

import profile_rules as rules  # noqa: E402
from memory_guard import BudgetGuard, require_available_memory  # noqa: E402
from probe_attention_ablation import MODEL, SEAM, build, timed  # noqa: E402

BATCH, WIDTH, ROUNDS = 2, 97, 9
BUDGET_GB = 12.0

# What a 0.02 tie in gain demands of f, and of milliseconds on this step.
# Derived in the module docstring from the registered formula.
TIE_RESOLUTION_MS = 1.0


def passthrough():
    """A seam installed whose replacement changes nothing at all."""
    def wrap(original):
        def call(queries, keys, values, *args, **kwargs):
            return original(queries, keys, values, *args, **kwargs)
        return call
    return wrap


ARMS = {"stock-a": None, "stock-b": None, "stock-passthrough": passthrough()}


def main() -> int:
    guard = BudgetGuard(BUDGET_GB)
    guard.check("startup")
    require_available_memory(BUDGET_GB, "startup")

    model, _ = load(str(MODEL))
    model.freeze()
    linear_to_lora_layers(model, rules.LORA_LAYERS,
                          {"rank": rules.LORA_RANK, "scale": 20.0,
                           "dropout": 0.0})
    model.train()
    optimizer = optim.Adam(learning_rate=1e-5)
    step, state = build(model, optimizer)

    mx.random.seed(0)
    tokens = mx.random.randint(0, 1000, (BATCH, WIDTH))
    lengths = mx.repeat(mx.array([[WIDTH // 2, WIDTH - 1]], dtype=mx.int32),
                        BATCH, axis=0)
    mx.eval(tokens, lengths)
    batch = (tokens, lengths)

    for install in ARMS.values():
        timed(step, state, batch, install)
    samples = {name: [] for name in ARMS}
    for _ in range(ROUNDS):
        for name, install in ARMS.items():
            samples[name].append(timed(step, state, batch, install))

    median = {name: statistics.median(rows) for name, rows in samples.items()}
    spread = {name: (max(rows) - min(rows)) / statistics.median(rows) * 100
              for name, rows in samples.items()}

    print(f"model {MODEL.name}, batch {BATCH}, width {WIDTH}, "
          f"{ROUNDS} interleaved rounds\n")
    print(f"{'arm':>18} {'median ms':>10} {'spread %':>9}")
    for name in ARMS:
        print(f"{name:>18} {median[name] * 1e3:>10.2f} {spread[name]:>9.2f}")

    null = abs(median["stock-a"] - median["stock-b"])
    seam = median["stock-passthrough"] - median["stock-a"]
    print(f"\nnull difference, two identical arms   {null * 1e3:>7.3f} ms  "
          f"({null / median['stock-a'] * 100:.3f}% of the step)")
    print(f"cost of installing a seam at all      {seam * 1e3:>7.3f} ms  "
          f"({seam / median['stock-a'] * 100:.3f}% of the step)")
    print(f"what a 0.02 tie in gain demands       "
          f"{TIE_RESOLUTION_MS:>7.3f} ms")

    if null * 1e3 < TIE_RESOLUTION_MS:
        print("\nThe harness resolves a difference at the registered tie band.")
    elif null * 1e3 < 4.0:
        print("\nCandidates with a large share are resolvable and a small one "
              "is not; any small share measured this way is a bound rather "
              "than a measurement, and must be reported as one.")
    else:
        print("\nThe null difference is the same size as the deltas being "
              "reported. Nothing measured by differencing two arms of this "
              "harness means anything until the step is longer or the rounds "
              "are more numerous.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
