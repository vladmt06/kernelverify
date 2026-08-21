"""What one arm actually costs at the 4B target, against the model that priced it.

Amendment 10 clause 46 prices the calibration's step context through
Amendment 6's cost model, 0.4 seconds at the short width and 4.0 at the long,
and says in its own text that "that portion is a MODEL and not a measurement".
The whole 16.07-hour subtotal, and every window budget built on it, rests on
those two numbers.

This measures them, on the pinned 4B at the deciding cell's batch, on the two
pinned bands, with mlx-lm's own iterator supplying the batch. It measures one
STOCK arm only, so it prices the arm and dials nothing and computes no share.

It also times the three warm-ups separately, because clause 46 prices a build
at three plain steps and the first warm-up is the one that traces and
compiles, which clause 46 lists as an absent item rather than as part of this
one.

    python bench/mlx_probes/probe_target_arm_price.py [rounds]
"""

from __future__ import annotations

import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

import profile_knobs as pk  # noqa: E402
import profile_rules as rules  # noqa: E402

MODEL = ROOT / "bench" / ".models" / "qwen3-4b-4bit-g64"
DATA = ROOT / "bench" / ".data"
BATCH = rules.CELLS["B"]["batch"]
SEED = 0

# Amendment 6's model, the figures clause 46 priced the subtotal through.
MODELLED_MS = {"short": 400.0, "long": 4000.0}


def target():
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load

    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model, tokenizer = load(str(MODEL))
    mx.random.seed(SEED)
    model.freeze()
    linear_to_lora_layers(model, rules.LORA_LAYERS,
                          {"rank": rules.LORA_RANK, "scale": 20.0,
                           "dropout": 0.0})
    model.train()
    optimizer = optim.Adam(learning_rate=1e-5)
    pk.settle_optimizer(model, optimizer)
    state = [model.state, optimizer.state, mx.random.state]
    return model, tokenizer, optimizer, state


def fixed_batch(tokenizer, width: str):
    """mlx-lm's own iterator, one batch, frozen: the real width and mask."""
    import mlx.core as mx
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import iterate_batches

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
    return (tokens, lengths), int(tokens.shape[1])


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    model, tokenizer, optimizer, state = target()
    print(f"model {MODEL.name}, batch {BATCH}, adapted {rules.LORA_LAYERS}, "
          f"rank {rules.LORA_RANK}, {rounds} rounds, compiled")

    for width in rules.WIDTH_ORDER:
        batch, seen = fixed_batch(tokenizer, width)
        expected = rules.WIDTHS[width]["batch_width"]
        note = "" if seen == expected else f"  (registered {expected})"
        from metalrunner import seams
        installation = seams.Installation()
        t0 = time.perf_counter()
        step, traces = pk.build_step(model, optimizer, state)
        compiled = pk.CompiledArm(label=f"stock-{width}", phi=1.0, step=step,
                                  state=state, traces=traces,
                                  traced_by_warmup=0)
        warmups = []
        for _ in range(pk.WARMUPS):
            start = time.perf_counter()
            pk.one_step(compiled, batch)
            warmups.append((time.perf_counter() - start) * 1000.0)
        installation.remove()
        compiled.traced_by_warmup = len(traces)
        build_ms = (time.perf_counter() - t0) * 1000.0

        samples = pk.timed_rounds([compiled], batch, rounds=rounds)
        ms = sorted(v * 1000.0 for v in samples[compiled.label])
        step_ms = statistics.median(ms)
        modelled = MODELLED_MS[width]
        print(f"\n{width}: batch width {seen}{note}")
        print(f"  warm-ups            " + "  ".join(f"{v:9.1f}" for v in warmups)
              + " ms")
        print(f"  timed rounds        min {ms[0]:9.1f}   median {step_ms:9.1f}"
              f"   max {ms[-1]:9.1f} ms")
        print(f"  spread over median  {(ms[-1] - ms[0]) / step_ms:9.4f}")
        print(f"  one whole build     {build_ms:9.1f} ms = "
              f"{build_ms / step_ms:5.2f} timed steps "
              f"(clause 46 assumes 3.00)")
        print(f"  MEASURED / MODELLED {step_ms:9.1f} / {modelled:.1f} ms = "
              f"{step_ms / modelled:5.3f}")
        del compiled, step, traces
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
