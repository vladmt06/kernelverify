"""Candidate Q's share of the real step, by shrinking every projection at once.

`probe_qmm_knob.py` established that a quantized matmul's cost is a clean
linear dial on its input width, in a microbenchmark. That is a necessary
condition and not a sufficient one: in a microbenchmark one matmul runs alone,
and in a training step 197 of them run among everything else, where laziness,
allocation and overlap all get a say. This drives the same knob through the
real step and reads candidate Q's share off the slope.

Why the slope and not an endpoint. A retuned kernel still issues one dispatch
per call site, still reads every weight byte, and still pays the host-side
construction of its own operations. Removing the matmul deletes all three, so
an endpoint credits the sprint with time no kernel could win back, 197 times
per step. Fitting T(phi) = a + b*phi puts that fixed cost in the intercept and
leaves the slope holding only what a retune can attack.

Four properties of the knob, each load-bearing
----------------------------------------------
The output shape does not move, so the residual adds still typecheck and
`LoRALinear.__call__`'s separate low-rank branch is untouched.

The launch count does not move: one `quantized_matmul` per call at every phi.

The backward keeps its structure without a pad, because the vjp of a slice
scatters into a zeros tensor of the original shape, so the gradient with
respect to x comes back at full width for free.

The slice is applied at EVERY arm including phi = 1.0, so the scaffold's cost
enters every arm rather than only the small ones. The `noslice` arm prices
that scaffold against stock separately.

What it measured, 2026-08-20
----------------------------
Batch 2, width 97, 196 quantized projections, seven interleaved rounds, three
warm-ups.

    arm        median ms
    stock        135.99
    noslice      135.80
    phi=1.00     136.31
    phi=0.75     125.76
    phi=0.50     112.08
    phi=0.25      98.25

    fit T(phi) = a + b*phi, R-squared 0.9962
      intercept   86.14 ms   a retune keeps this
      slope       51.14 ms   a retune attacks this
      f_Q by slope     0.3760
      f_Q by endpoint  0.3700

Both readings agree to under two percent of themselves, and both scaffold
prices are negligible against the slope: applying the slice costs +0.51 ms and
installing the seam at all costs -0.19 ms, against a 51 ms slope and a 0.155
ms resolution floor.

One scope difference from the pre-registration, stated because it matters.
This seam is `QuantizedLinear.__call__`, so it covers the projections and NOT
the tied output head, which is a `QuantizedEmbedding` and is reached through
`as_linear`. Section 3.3 registers candidate Q as the projections plus the
head, so this 0.376 is the projections alone and the registered f_Q would be
larger by the head's share. That also makes this figure disjoint from
candidate L's, which is what allows the sum check below.

The check the marked instrument could never pass
------------------------------------------------
    disjoint region                              share
    head plus cross-entropy   (candidate L)      0.245
    attention core            (candidate A)      0.045
    quantized projections     (candidate Q)      0.376
    sum                                          0.666
    unattributed remainder                       0.334

Three disjoint regions accounting for two thirds of the step and leaving a
third for the LoRA adapters, the norms, the elementwise work and the optimizer
update is a reading that can be true. The same three regions by the registered
marked instrument sum to 1.857, which cannot be, and section 3.3's
reconciliation would have had to reject the profile without ever saying why.

    python bench/mlx_probes/probe_qmm_knob_insitu.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx_lm.tuner.trainer import _clear_cache  # noqa: E402
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
from probe_attention_ablation import MODEL, build, timed  # noqa: E402
from probe_qmm_knob import GROUP, fit  # noqa: E402

BATCH, WIDTH, ROUNDS, WARMUPS = 2, 97, 7, 3
PHIS = (1.0, 0.75, 0.5, 0.25)
BUDGET_GB = 14.0
SEAM = pi.REGIONS["qmm"].seam


def ladder(model, phi: float) -> dict:
    """Every quantized projection, cut to a prefix of its input dimension.

    Keyed by module identity, materialised once outside any timed region. A
    dimension is rounded to a whole number of quantization groups, because a
    partial group has no scale of its own and the operand would not be a valid
    quantized weight.
    """
    slices = {}
    for _name, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear):
            continue
        weight = module["weight"]
        _out_dims, packed = weight.shape
        in_dims = packed * 32 // module.bits
        in_k = max(GROUP, int(round(phi * in_dims / GROUP)) * GROUP)
        packed_k = in_k * module.bits // 32
        biases = module.get("biases")
        entry = (
            in_k,
            mx.contiguous(weight[:, :packed_k]),
            mx.contiguous(module["scales"][:, :in_k // module.group_size]),
            (mx.contiguous(biases[:, :in_k // module.group_size])
             if biases is not None else None),
        )
        mx.eval(*[part for part in entry[1:] if part is not None])
        slices[id(module)] = entry
    return slices


def knob(slices, apply_slice: bool):
    def wrap(_original):
        def call(self, x):
            in_k, weight, scales, biases = slices[id(self)]
            operand = x[..., :in_k] if apply_slice else x
            y = mx.quantized_matmul(
                operand, weight, scales=scales, biases=biases, transpose=True,
                group_size=self.group_size, bits=self.bits, mode=self.mode)
            return y + self["bias"] if "bias" in self else y
        return call
    return wrap


def timed_knob(step, state, batch, slices, apply_slice=True):
    mx.disable_compile()
    installation = seams.Installation()
    try:
        installation.install(SEAM, knob(slices, apply_slice))
        start = time.perf_counter()
        lvalue, toks, grad = step(batch, None, True)
        mx.eval(state, lvalue, toks, grad)
        _clear_cache(0)
        return time.perf_counter() - start
    finally:
        installation.remove()
        mx.enable_compile()


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

    ladders = {phi: ladder(model, phi) for phi in PHIS}
    guard.check("ladder resident")
    projections = len(ladders[1.0])

    def arms():
        yield "stock", lambda: timed(step, state, batch, None)
        yield "noslice", lambda: timed_knob(step, state, batch, ladders[1.0],
                                            apply_slice=False)
        for phi in PHIS:
            yield f"phi={phi:.2f}", (
                lambda p=phi: timed_knob(step, state, batch, ladders[p]))

    names = [name for name, _ in arms()]
    for _ in range(WARMUPS):
        for _name, run in arms():
            run()
    samples = {name: [] for name in names}
    for _ in range(ROUNDS):
        for name, run in arms():
            samples[name].append(run())
    median = {name: statistics.median(rows) for name, rows in samples.items()}

    print(f"model {MODEL.name}, batch {BATCH}, width {WIDTH}, "
          f"{projections} quantized projections, {ROUNDS} interleaved rounds, "
          f"{WARMUPS} warm-ups\n")
    print(f"{'arm':>10} {'median ms':>10}")
    for name in names:
        print(f"{name:>10} {median[name] * 1e3:>10.2f}")

    stock = median["stock"]
    times = [median[f"phi={phi:.2f}"] for phi in PHIS]
    slope, intercept, r2 = fit(list(PHIS), times)
    scaffold = median["phi=1.00"] - median["noslice"]
    install_cost = median["noslice"] - stock

    print(f"\nfit T(phi) = a + b*phi:  R-squared {r2:.4f}")
    print(f"  intercept a  {intercept * 1e3:>8.2f} ms   a retune keeps this")
    print(f"  slope b      {slope * 1e3:>8.2f} ms   a retune attacks this")
    print(f"\nf_Q by slope     {slope / stock:.4f}   the defensible share")
    print(f"f_Q by endpoint  {(stock - median['phi=0.25']) / (0.75 * stock):.4f}"
          f"   the same read off two points")
    print(f"\nthe slice itself costs   {scaffold * 1e3:>+7.2f} ms  "
          f"(phi=1.00 against noslice)")
    print(f"installing at all costs  {install_cost * 1e3:>+7.2f} ms  "
          f"(noslice against stock)")
    print(f"peak footprint {phys_footprint_gb()[1]:.2f} GB")
    print("\nBoth scaffold prices should be small against the slope. Where "
          "they are not, only the endpoint survives and it is an upper bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
