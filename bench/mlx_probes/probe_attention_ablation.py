"""Measure attention's share of a training step from OUTSIDE the step.

Why this exists
---------------
The Day 1 profile times regions with marks: an identity `mx.custom_function`
that calls `mx.eval` and then reads the clock, one at each end of a region.
Every such mark costs time, and because the eval happens BEFORE the timestamp,
a region's exit mark charges its own cost to the region while the plain step
that forms the share's denominator pays none of it. The overstatement grows
with how many marks a region carries.

That prediction is testable and this probe tests it, by measuring the same
quantity with no instrument inside the step at all: replace attention with
something of the same output shape that does almost none of the work, and read
the whole step's time, exactly the way stock's own time is read.

Measured 2026-08-20 on qwen3-0.6b-4bit-g64, batch 2, LoRA on the last 16 of 28
blocks, at width 97:

    ablation                    0.047 of the step
    marking one block, scaled   0.296
    marking every block         0.208

Both marked figures exceed the unmarked one by four to six times. The size of
the gap matches the fence count: dense marking fires 44 mark pairs on
attention per step, and at roughly half a millisecond of fence apiece that is
most of what the marked spans contain.

Why the ablation is believed
----------------------------
Not because it is cleaner, but because it reproduces a scaling law it has no
way of knowing. Attention's score matrix is quadratic in the sequence length
while every other region in the step is linear, so attention's share must be
flat at short widths, where the linear part of attention dominates, and then
climb. Measured:

    width  attention ms  share
      97       6.59      0.047
     193      13.59      0.051
     385      49.88      0.089
     769     198.43      0.160

Doubling the width multiplies attention by 2.06, then 3.67, then 3.98: linear
at the short end and quadratic at the long end, with the crossover where the
geometry puts it. Nothing in an ablation knows what curve it ought to trace.

The three confounds, each measured rather than argued away
----------------------------------------------------------
The replacement must not let MLX drop the key and value projections. If it
returned only `queries`, nothing would consume keys or values and a lazy graph
would never schedule them, so the delta would quietly include their cost too.
The `ablated` arm therefore consumes both through a full reduction multiplied
by zero, which keeps the value identical to `queries` and both tensors on the
graph in each direction. That the reduction is not folded away is visible in
the numbers: the arm that drops it is 15 ms faster.

The replacement is not free. Its cost is measured by the `stock+reductions`
arm, which adds exactly the same reduction beside stock attention where it
changes nothing else, and is 1.07 ms. That correction is a measurement rather
than an assumption, which is what the first version of this probe lacked.

The delta covers the backward as well. With attention replaced, the gradient
reaches q_proj directly, so what the step stops paying is attention's forward
and backward together, which is what a share is defined over.

What this probe does NOT settle
-------------------------------
Removing an operation changes the graph, and a scheduler is free to behave
differently on the smaller one. This measures what the step stops paying when
attention stops happening, which is the quantity the gain formula needs, but
it is not the same as what the step spends on attention while everything else
is also in flight. Those two can differ wherever work overlaps.

    python bench/mlx_probes/probe_attention_ablation.py
"""

from __future__ import annotations

import statistics
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.nn.utils import average_gradients  # noqa: E402
from mlx.utils import tree_map  # noqa: E402
from mlx_lm.tuner.trainer import _clear_cache, default_loss  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402
from mlx_lm.utils import load  # noqa: E402

import profile_instrument as pi  # noqa: E402
import profile_rules as rules  # noqa: E402
from memory_guard import (  # noqa: E402
    BudgetGuard,
    phys_footprint_gb,
    require_available_memory,
)
from metalrunner import seams  # noqa: E402

MODEL = ROOT / "bench" / ".models" / "qwen3-0.6b-4bit-g64"
BATCH, ROUNDS = 2, 5
WIDTHS = (97, 193, 385, 769)
# An instrumented or wide step forces intermediates resident that a plain one
# releases, so this refuses before the operating system does. An earlier
# version of this sweep reached width 1025 at batch 4 with no budget and was
# killed by the OS, losing every width it had already measured.
BUDGET_GB = 12.0


def cheap_attention(keep_alive: bool):
    """A stand-in of the right shape that does almost none of the work."""
    def wrap(_original):
        def call(queries, keys, values, *args, **kwargs):
            if not keep_alive:
                return queries
            alive = keys.sum() * 0.0 + values.sum() * 0.0
            return queries + alive
        return call
    return wrap


def stock_plus_reductions():
    """Stock attention plus exactly the reduction the ablated arm adds."""
    def wrap(original):
        def call(queries, keys, values, *args, **kwargs):
            alive = keys.sum() * 0.0 + values.sum() * 0.0
            return original(queries, keys, values, *args, **kwargs) + alive
        return call
    return wrap


ARMS = {
    "stock": None,
    "stock+reductions": stock_plus_reductions(),
    "ablated": cheap_attention(keep_alive=True),
    "ablated-no-kv": cheap_attention(keep_alive=False),
}
SEAM = pi.REGIONS["attn-core"].seam


def build(model, optimizer):
    """mlx-lm's own step, mirrored; see bench/profile_stock.py for why."""
    loss_value_and_grad = nn.value_and_grad(model, default_loss)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            grad = average_gradients(grad)
            optimizer.update(model, grad)
            grad = None
        return lvalue, toks, grad

    return step, state


def timed(step, state, batch, install):
    """One step with one arm installed, timed where mlx-lm times its own."""
    mx.disable_compile()
    installation = None
    try:
        if install is not None:
            installation = seams.Installation()
            installation.install(SEAM, install)
        start = time.perf_counter()
        lvalue, toks, grad = step(batch, None, True)
        mx.eval(state, lvalue, toks, grad)
        _clear_cache(0)
        return time.perf_counter() - start
    finally:
        if installation is not None:
            installation.remove()
        mx.enable_compile()


def main() -> int:
    model, _ = load(str(MODEL))
    model.freeze()
    linear_to_lora_layers(model, rules.LORA_LAYERS,
                          {"rank": rules.LORA_RANK, "scale": 20.0,
                           "dropout": 0.0})
    model.train()
    optimizer = optim.Adam(learning_rate=1e-5)
    step, state = build(model, optimizer)
    guard = BudgetGuard(BUDGET_GB)

    print(f"model {MODEL.name}, batch {BATCH}, depth "
          f"{len(model.model.layers)}, adapted {rules.LORA_LAYERS}\n")
    print(f"{'width':>6} {'stock ms':>9} {'attn ms':>9} {'f_attn':>8} "
          f"{'f_attn+kv':>10} {'peak GB':>8}", flush=True)

    for width in WIDTHS:
        guard.check(f"width {width}")
        require_available_memory(BUDGET_GB, f"width {width}")
        mx.random.seed(0)
        tokens = mx.random.randint(0, 1000, (BATCH, width))
        lengths = mx.repeat(mx.array([[width // 2, width - 1]],
                                     dtype=mx.int32), BATCH, axis=0)
        mx.eval(tokens, lengths)
        batch = (tokens, lengths)

        for install in ARMS.values():
            timed(step, state, batch, install)
        # Interleaved within the round, so a clock excursion hits every arm
        # rather than whichever happened to run first.
        samples = {name: [] for name in ARMS}
        for _ in range(ROUNDS):
            for name, install in ARMS.items():
                samples[name].append(timed(step, state, batch, install))
        median = {name: statistics.median(rows)
                  for name, rows in samples.items()}

        stock = median["stock"]
        reductions = median["stock+reductions"] - stock
        attention = (stock - median["ablated"]) + reductions
        with_kv = stock - median["ablated-no-kv"]
        # Printed per width rather than at the end, so a refusal costs the
        # widths not yet measured rather than every width already measured.
        print(f"{width:>6} {stock * 1e3:>9.2f} {attention * 1e3:>9.2f} "
              f"{attention / stock:>8.4f} {with_kv / stock:>10.4f} "
              f"{phys_footprint_gb()[1]:>8.2f}", flush=True)

    print("\nf_attn must rise across these rows, because attention is "
          "quadratic in the width and every other region is linear. If it "
          "does not, the ablation is measuring something other than "
          "attention.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
