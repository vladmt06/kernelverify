"""Candidate L's share of the step, measured by shrinking the vocabulary.

Why a knob rather than a deletion
---------------------------------
Candidate L is a streamed, mask-aware output head plus cross-entropy. Like the
quantized matmul, it would be retuned rather than deleted: the kernel still
reads every weight byte of a 151936-row head and still reduces over whatever
logits it produces. Removing the pair entirely would credit the sprint with
the dispatch and the host-side graph construction that any replacement keeps.

So the knob is the vocabulary. Slicing the head's output rows shrinks the
matmul's arithmetic AND the logits tensor the cross-entropy reduces over, in
one dial, and those two are exactly candidate L's registered region set. The
launch count is unchanged: one `quantized_matmul` and one `cross_entropy` per
step whatever the width. Fit

    T(phi) = a + b * phi

and read L's share off the slope, which is the part a retune can attack.

Why the targets are taken modulo the sliced vocabulary
------------------------------------------------------
Cross-entropy indexes its targets into the logits, so a target beyond the
sliced width would be out of range. Its cost depends on the width of the
logits and not on which column a target names, so `targets % V` keeps the
work honest while keeping the indices valid. The loss VALUE changes, which is
why no arm here is compared against stock's loss; only times are compared.

Two things this shares with every other probe in this directory
--------------------------------------------------------------
Three warm-ups per arm, not one. At a single warm pass the uniqueness probe
returned two mutually contradictory answers on consecutive runs, because the
first timed round of whichever arm ran first still carried allocation and
graph-cache cost.

Arms interleaved within a round, so a clock excursion lands on all of them
rather than on whichever ran first, and the resolution floor from
`probe_step_resolution.py` (0.155 ms, 0.115% of the step) is what any
difference here has to clear.

What it measured, 2026-08-20
----------------------------
Batch 2, width 97, nine interleaved rounds, three warm-ups.

    arm        median ms   vocabulary
    stock        141.46      151936
    phi=1.00     143.78      151936
    phi=0.75     135.55      113952
    phi=0.50     125.87       75968
    phi=0.25     118.15       37984

    fit T(phi) = a + b*phi, R-squared 0.9983
      intercept  109.19 ms   what a retune keeps
      slope       34.64 ms   what a retune attacks
      f_L by slope     0.2449
      f_L by endpoint  0.2197

The knob is a clean dial and the two readings agree to about a tenth of
themselves.

One bias is real and is reported rather than absorbed. The phi = 1.00 arm is
2.33 ms slower than stock, which is fifteen times the 0.155 ms resolution
floor, so the scaffold is not free: the arm holds a second copy of a 78 MB
head weight and reaches it through a Python frame the stock path does not
have. That is 6.7% of the slope, and it biases f_L upward.

Where this sits against the marked instrument
---------------------------------------------
    candidate   marks/step   marked f   unmarked f   error
    L                2         0.263       0.245       +7%
    A               32         0.193       0.045     +329%
    Q              197         1.401     not yet

The error tracks the mark count, which is the diagnosis predicting itself. Two
marks contaminate almost nothing, so the registered instrument was very nearly
right about candidate L and would have been believed. Thirty-two marks
overstate by a factor of four. A hundred and ninety-seven produce a number
above one, which is the only reason the fault was noticed at all.

    python bench/mlx_probes/probe_loss_knob.py
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

import profile_instrument as pi  # noqa: E402
import profile_rules as rules  # noqa: E402
from memory_guard import BudgetGuard, require_available_memory  # noqa: E402
from metalrunner import seams  # noqa: E402
from probe_attention_ablation import MODEL, build, timed  # noqa: E402
from probe_qmm_knob import fit  # noqa: E402

BATCH, WIDTH, ROUNDS, WARMUPS = 2, 97, 9, 3
PHIS = (1.0, 0.75, 0.5, 0.25)
BUDGET_GB = 12.0

HEAD_SEAM = pi.REGIONS["head-matmul"].seam
LOSS_SEAM = pi.REGIONS["cross-entropy"].seam


def head_slice(model, phi: float):
    """The tied head's weights, cut to a prefix of the vocabulary.

    Materialised once, outside any timed region, so the arms differ in the
    work they do and not in when they paid for their operands.
    """
    embedding = model.model.embed_tokens
    rows = embedding["weight"].shape[0]
    kept = max(1, int(round(phi * rows)))
    weight = mx.contiguous(embedding["weight"][:kept])
    scales = mx.contiguous(embedding["scales"][:kept])
    biases = embedding.get("biases")
    biases = mx.contiguous(biases[:kept]) if biases is not None else None
    mx.eval(weight, scales, *( [biases] if biases is not None else [] ))
    return kept, weight, scales, biases


def head_arm(kept, weight, scales, biases):
    def wrap(_original):
        def call(self, x):
            return mx.quantized_matmul(
                x, weight, scales=scales, biases=biases, transpose=True,
                group_size=self.group_size, bits=self.bits, mode=self.mode)
        return call
    return wrap


def loss_arm(kept):
    def wrap(original):
        def call(logits, targets, *args, **kwargs):
            return original(logits, targets % kept, *args, **kwargs)
        return call
    return wrap


def timed_pair(step, state, batch, kept, weight, scales, biases):
    """One step with BOTH of candidate L's regions cut to the same width."""
    mx.disable_compile()
    installation = seams.Installation()
    try:
        installation.install(HEAD_SEAM, head_arm(kept, weight, scales, biases))
        installation.install(LOSS_SEAM, loss_arm(kept))
        import time

        from mlx_lm.tuner.trainer import _clear_cache

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

    ladders = {phi: head_slice(model, phi) for phi in PHIS}
    guard.check("ladder resident")

    def run(phi):
        kept, weight, scales, biases = ladders[phi]
        return timed_pair(step, state, batch, kept, weight, scales, biases)

    for _ in range(WARMUPS):
        timed(step, state, batch, None)
        for phi in PHIS:
            run(phi)
    stock_samples, samples = [], {phi: [] for phi in PHIS}
    for _ in range(ROUNDS):
        stock_samples.append(timed(step, state, batch, None))
        for phi in PHIS:
            samples[phi].append(run(phi))

    stock = statistics.median(stock_samples)
    median = {phi: statistics.median(rows) for phi, rows in samples.items()}
    slope, intercept, r2 = fit(list(PHIS), [median[phi] for phi in PHIS])

    print(f"model {MODEL.name}, batch {BATCH}, width {WIDTH}, "
          f"{ROUNDS} interleaved rounds, {WARMUPS} warm-ups\n")
    print(f"{'arm':>16} {'median ms':>10} {'vocab':>8}")
    print(f"{'stock':>16} {stock * 1e3:>10.2f} "
          f"{model.model.embed_tokens['weight'].shape[0]:>8}")
    for phi in PHIS:
        print(f"{f'phi={phi:.2f}':>16} {median[phi] * 1e3:>10.2f} "
              f"{ladders[phi][0]:>8}")

    print(f"\nfit T(phi) = a + b*phi:  R-squared {r2:.4f}")
    print(f"  intercept a  {intercept * 1e3:>8.2f} ms   "
          f"launch, encode and host cost a retune keeps")
    print(f"  slope b      {slope * 1e3:>8.2f} ms   "
          f"arithmetic and traffic a retune attacks")
    print(f"\nf_L by slope     {slope / stock:.4f}   the defensible share")
    print(f"f_L by endpoint  {(stock - median[0.25]) / (0.75 * stock):.4f}   "
          f"the same quantity read off two points, for comparison")
    print(f"\nphi=1.00 against stock: {(median[1.0] - stock) * 1e3:+.2f} ms, "
          f"which is the scaffold's own price and should be near the "
          f"0.155 ms resolution floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
