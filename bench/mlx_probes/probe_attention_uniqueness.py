"""Does attention sit alone on the step's critical path, or under other work?

Why this matters more than the number it corrects
-------------------------------------------------
`probe_attention_ablation.py` measures what the step stops paying when
attention stops happening. That is the quantity the registered gain formula
needs, but it is not the same as what the step SPENDS on attention, and the
two differ exactly where work overlaps. MLX permits overlap: its Metal backend
inserts a barrier only where a dispatch reads a buffer an earlier dispatch in
the same encoder wrote, so independent dispatches are free to run together,
and a training step contains genuinely independent pairs - the query, key and
value projections, the gate and up projections, and the LoRA branch against
the base matmul it is added to.

If attention runs partly underneath other work, then no partition of the
step's wall time into regions exists at all, and section 3.3's requirement
that the named regions plus a remainder equal the step is asking for something
that is not there. That would explain the fenced instrument reporting a share
of 1.401 without any arithmetic error: a sum of isolated latencies over a
pipelined critical path is a work-to-span ratio, and nothing bounds it by one.

The construction
----------------
Run attention TWICE, with a real data dependency between the two calls so the
copies cannot overlap with each other:

    out1 = original(q, k, v, ...)
    q2   = q + 0.0 * out1
    return original(q2, k, v, ...)

The multiply by literal zero makes the second call wait for the first while
leaving the returned value bit-identical to stock, which is asserted rather
than assumed. Then

    kappa = (T_doubled - T_stock) / attention_cost

where `attention_cost` is the ablation's own figure. A second copy of
attention that must run serially adds one attention's worth of time to the
critical path. So kappa near 1 means attention was already alone on the
critical path and the attribution is unique. Kappa above 1 means the first
copy was partly hidden under other work and the ablation's figure understates
what attention costs in isolation, which is the honest direction: a faster
kernel cannot recover time the step was not spending.

Two biases, both stated rather than corrected. The doubled call reads query,
key and value tensors that are already cache-hot, so it is cheaper than the
first and kappa is biased low, making this a lower bound on hidden work. And
doubling changes the memory profile, which at a short width is small and at a
long one is not.

What it measured, 2026-08-20
----------------------------
    width   attention ms   doubled adds   kappa
      97        6.51           7.26        1.11
     385       49.47          54.20        1.10
      97        6.56           6.10        0.93
     385       53.33          56.94        1.07

kappa is 1.0 within the run-to-run scatter at both widths, so attention was
already alone on the critical path and its share is a unique quantity here.
The worry that a pipelined step admits no attribution at all is measured away
for this region rather than argued away.

It is also an independent check on the ablation, in the opposite direction:
removing attention saves 6.5 ms and adding a second serial copy costs 6.1 to
7.3 ms. Two opposite perturbations agreeing to within a few percent is a much
stronger statement than either alone, and both disagree with the marked
instrument's 0.208 and 0.296 by four to six times.

One warm-up per arm is NOT enough, and this probe found that the hard way. At
a single warm-up the same construction returned kappa 7.71 and -0.00 on one
run and 0.67 and 0.82 on the next: not a noisy measurement but a meaningless
one. Three warm-ups and nine rounds bring the arm spreads to 1.5 to 2.3% and
kappa to 1.0 plus or minus 0.1. Any probe here that differences two arms
should assume the same, because a single warm pass leaves allocation and
graph-cache costs inside the first timed round of whichever arm ran first.

    python bench/mlx_probes/probe_attention_uniqueness.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx_lm.tuner.trainer import default_loss  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402
from mlx_lm.utils import load  # noqa: E402

import profile_rules as rules  # noqa: E402
from memory_guard import BudgetGuard, require_available_memory  # noqa: E402
from metalrunner import seams  # noqa: E402
from probe_attention_ablation import (  # noqa: E402
    MODEL,
    SEAM,
    build,
    cheap_attention,
    stock_plus_reductions,
    timed,
)

BATCH, ROUNDS, WARMUPS = 2, 9, 3
WIDTHS = (97, 385)
BUDGET_GB = 12.0


def doubled_attention():
    """Stock attention run twice, serially, returning stock's exact value."""
    def wrap(original):
        def call(queries, keys, values, *args, **kwargs):
            first = original(queries, keys, values, *args, **kwargs)
            # A real data dependency, so the second call cannot overlap the
            # first, and a value change of exactly zero, so the step still
            # computes what stock computes.
            waited = queries + 0.0 * first
            return original(waited, keys, values, *args, **kwargs)
        return call
    return wrap


ARMS = {
    "stock": None,
    "stock+reductions": stock_plus_reductions(),
    "ablated": cheap_attention(keep_alive=True),
    "doubled": doubled_attention(),
}


def loss_of(model, batch, install):
    """The loss on unchanged weights, so two arms can be compared exactly."""
    value_and_grad = nn.value_and_grad(model, default_loss)
    installation = None
    mx.disable_compile()
    try:
        if install is not None:
            installation = seams.Installation()
            installation.install(SEAM, install)
        (loss, _toks), _grad = value_and_grad(model, *batch)
        mx.eval(loss)
        return loss
    finally:
        if installation is not None:
            installation.remove()
        mx.enable_compile()


def main() -> int:
    guard = BudgetGuard(BUDGET_GB)
    model, _ = load(str(MODEL))
    model.freeze()
    linear_to_lora_layers(model, rules.LORA_LAYERS,
                          {"rank": rules.LORA_RANK, "scale": 20.0,
                           "dropout": 0.0})
    model.train()
    optimizer = optim.Adam(learning_rate=1e-5)
    step, state = build(model, optimizer)

    print(f"model {MODEL.name}, batch {BATCH}, adapted {rules.LORA_LAYERS}\n")
    print(f"{'width':>6} {'stock ms':>9} {'attn ms':>9} {'doubled +ms':>12} "
          f"{'kappa':>7}  loss identical")

    for width in WIDTHS:
        guard.check(f"width {width}")
        require_available_memory(BUDGET_GB, f"width {width}")
        mx.random.seed(0)
        tokens = mx.random.randint(0, 1000, (BATCH, width))
        lengths = mx.repeat(mx.array([[width // 2, width - 1]],
                                     dtype=mx.int32), BATCH, axis=0)
        mx.eval(tokens, lengths)
        batch = (tokens, lengths)

        # Checked before any timing and on weights nothing has updated, so a
        # difference here is the construction and not the optimizer.
        identical = bool(mx.array_equal(loss_of(model, batch, None),
                                        loss_of(model, batch, ARMS["doubled"])))

        for _ in range(WARMUPS):
            for install in ARMS.values():
                timed(step, state, batch, install)
        samples = {name: [] for name in ARMS}
        for _ in range(ROUNDS):
            for name, install in ARMS.items():
                samples[name].append(timed(step, state, batch, install))
        median = {name: statistics.median(rows)
                  for name, rows in samples.items()}

        stock = median["stock"]
        reductions = median["stock+reductions"] - stock
        attention = (stock - median["ablated"]) + reductions
        added = median["doubled"] - stock
        kappa = added / attention if attention > 0 else float("nan")
        spread = {name: (max(rows) - min(rows)) / statistics.median(rows) * 100
                  for name, rows in samples.items()}
        print(f"{width:>6} {stock * 1e3:>9.2f} {attention * 1e3:>9.2f} "
              f"{added * 1e3:>12.2f} {kappa:>7.2f}  {identical}  "
              f"spread stock {spread['stock']:.1f}% doubled "
              f"{spread['doubled']:.1f}%", flush=True)

    print("\nkappa near 1 means attention was already alone on the critical "
          "path, so its share is a unique quantity at this width.")
    print("kappa above 1 means the first copy was partly hidden under other "
          "work, the timeline does not decompose, and the ablation figure is "
          "a lower bound on what attention costs in isolation.")
    print("The doubled call reads cache-hot inputs, so kappa is biased low "
          "and is itself a lower bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
