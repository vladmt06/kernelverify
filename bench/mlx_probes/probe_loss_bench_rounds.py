"""Does candidate L's bench resolve its own scaffold offset, and at what cost.

Amendment 12 clause 49 records that at the long width the bench's scaffold
offset did not resolve at the registered five rounds: across four runs the
offset against stock ranged 49 to 338 ms and against the reference 4 to 174,
and in one of the four the ordering reversed. It registers that as an
obligation on step 10 and says nothing in the document permits a bench to
take a different round count from the step.

Clause 49 also leaves a second question open: candidate L's SHORT width fit
sat on both sides of clause 21's 0.99 linearity floor across three runs, and
a bench that fails that gate leaves candidate L with no credited denominator.

Both were measured on the 0.6B proxy's own supervised counts. This measures
them at the pinned 4B dimensions with the supervised counts mlx-lm's own
iterator produces at the registered seed, at two round counts, repeated, so
what moves with rounds is separable from what moves between runs.

It reports and rules nothing. Clause 21's `3R` gate cannot be evaluated here
at all, because `R` comes from step 10 and does not exist.

    python bench/mlx_probes/probe_loss_bench_rounds.py [repeats]
"""

from __future__ import annotations

import statistics
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

import ceiling_sweep as cs  # noqa: E402
import profile_knobs as pk  # noqa: E402
import profile_rules as rules  # noqa: E402

MODEL = ROOT / "bench" / ".models" / "qwen3-4b-4bit-g64"
DATA = ROOT / "bench" / ".data"
HIDDEN, VOCAB = 2560, 151936
BATCH, SEED = rules.CELLS["B"]["batch"], 7
ROUND_COUNTS = (5, 15)
WIDTHS_ARG = None


def supervised_at(width: str) -> tuple[int, int]:
    """The batch mlx-lm's own iterator yields first, and its supervised rows.

    The same derivation `profile_stock._fixed_batch` uses, so the count is the
    one the profile's own context will carry rather than a band-level average.
    """
    import mlx.core as mx
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import iterate_batches

    tokenizer = load_tokenizer(MODEL)
    args = types.SimpleNamespace(
        data=str(DATA / rules.WIDTHS[width]["data"]), train=True, test=False,
        mask_prompt=True, prompt_feature="prompt",
        completion_feature="completion", text_feature="text",
        chat_feature="messages")
    train_set, _, _ = load_dataset(args, tokenizer)
    batches = iterate_batches(dataset=CacheDataset(train_set),
                              batch_size=BATCH, max_seq_length=rules.SEQ_LEN,
                              loop=False, seed=SEED,
                              comm_group=mx.distributed.init())
    tokens, lengths = next(batches)
    mx.eval(tokens, lengths)
    targets = tokens[:, 1:]
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    return int(mask.sum().item()), int(targets.shape[0] * targets.shape[1])


def read(samples, roles, rounds):
    """The bench's own reduction, the same call the sweep makes.

    `R` comes from step 10 and does not exist, so a placeholder of 1 ms goes
    in and NO gate that reads it is evaluated here. The offset and the fit
    below are the raw readings; clause 21's `3R` limit and clause 26's `10R`
    excursion are not testable until the addendum exists.
    """
    return pk.reduce_width(samples, roles, resolution_floor=1.0,
                           rounds=rounds, family=pk.FLOOR)


def main() -> int:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"4B dimensions: head ({VOCAB}, {HIDDEN}), batch {BATCH}, "
          f"{repeats} repeats per round count")

    widths = ((sys.argv[2],) if len(sys.argv) > 2 else rules.WIDTH_ORDER)
    for width in widths:
        supervised, total = supervised_at(width)
        print(f"\n{width}: {supervised} supervised of {total} rows "
              f"({supervised / total:.3f})")
        context = {"supervised": supervised, "supervised_of": total}
        for rounds in ROUND_COUNTS:
            offsets, r2s, slopes, arm_ms = [], [], [], []
            for _ in range(repeats):
                samples, roles = cs.run_loss_bench(
                    context, width=width, hidden=HIDDEN, vocab=VOCAB,
                    rounds=rounds)
                reading = read(samples, roles, rounds)["L"]
                offsets.append(reading.scaffold_offset)
                r2s.append(reading.pooled.r_squared)
                slopes.append(reading.slope)
                arm_ms.append(statistics.median(
                    v for vals in samples.values() for v in vals))

            def span(vals):
                return f"{min(vals):9.3f} to {max(vals):9.3f}"
            print(f"  {rounds:2d} rounds   "
                  f"offset {span(offsets)}   "
                  f"spread {max(offsets) - min(offsets):8.3f} ms")
            print(f"              slope  {span(slopes)}   "
                  f"R2 {min(r2s):.5f} to {max(r2s):.5f}   "
                  f"clears 0.99 in "
                  f"{sum(1 for v in r2s if v >= pk.R_SQUARED_FLOOR)}"
                  f"/{repeats}   arm {statistics.median(arm_ms):8.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
