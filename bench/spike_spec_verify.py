"""Does routing a speculative-verification step buy anything at the step level?

Pre-registered in docs/research/2026-08-18-spec-verify-meeting-point-spike.md,
which fixes the outcomes before this file was written. Read it first.

The narrow question: a K=6 verification presents 7 tokens of ONE stream to
every projection, shape (1, 7, d_in). `wide_qmv` routes M = 5..9 and the
2026-08-17 A/B measured 14.61% at M=6 and 15.64% at M=8, so M=7 is inside the
window and between two wins - but serve_sub4bit's gate refuses the sequence
spelling before it ever computes M.

There is NO kernel-level question here and this harness must not pretend one
exists: `_fused` flattens with `x.reshape(-1, d_in)`, so the two spellings
produce the same kernel call on the same array. What differs is everything
around the projection - 7 query positions against the cache rather than one -
so the honest question is whether the STEP gets faster, and it can answer no
even though the projection is faster in isolation.

This harness installs its OWN interception rather than serve_sub4bit's, and
imports nothing from it that could change its behaviour. That harness produced
the published grid and was re-measured on 2026-08-17; a spike must not be able
to move a published number by editing the thing that produced it.

Exit vocabulary is memory_guard's, as everywhere: 0 measured, 1 measured and
stopped, 3 budget, 4 lock held, 5 low memory, 8 precondition, 9 not idle.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from machine_state import MeasurementLock, spread_pct  # noqa: E402
from memory_guard import (  # noqa: E402
    EXIT_BUDGET_REFUSAL,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    EXIT_NOT_IDLE,
    EXIT_PRECONDITION,
    BudgetExceeded,
    BudgetGuard,
    LowMemoryRefusal,
    budget_gb_arg,
    require_available_memory,
)
from kernelverify.pack.wide_qmv import launch_config, should_dispatch  # noqa: E402
from serve_sub4bit import (  # noqa: E402
    GROUP_SIZE,
    MODEL_3BIT,
    MODELS_ROOT,
    PINNED_BITS,
    SUPPORTED_BITS,
    NotIdle,
    PreconditionFailed,
    load_model,
    make_prompts,
    provenance,
    require_idle,
    verify_pins,
)
from serve_sub4bit import _KERNEL  # noqa: E402

LOCK_NAME = "spike_spec_verify"
BUDGET_GB = 24.0
ROUNDS = 5
PROMPT_T = 64
DRAFT_K = 6                 # the paper's optimum; verification is K + 1 tokens
VERIFY_M = DRAFT_K + 1


class _SeqRoutedLinear:
    """Like serve_sub4bit's, with ONE rule changed and the change named.

    serve_sub4bit refuses any 3-D input whose sequence length is not 1, before
    it computes M. That check is what this spike exists to question, so here
    the flattened M is computed first and `should_dispatch` decides - which
    still refuses everything outside M = 5..9, prefill chunks included, since a
    real prefill is hundreds of tokens.
    """

    __slots__ = ("inner", "patch")

    def __init__(self, inner, patch):
        self.inner = inner
        self.patch = patch

    def _ineligible(self, x) -> str | None:
        inner = self.inner
        if not str(getattr(inner, "mode", "affine")).endswith("affine"):
            return "mode"
        if inner.bits not in SUPPORTED_BITS or inner.group_size != GROUP_SIZE:
            return f"quant-{inner.bits}b-gs{inner.group_size}"
        if "bias" in inner:
            return "bias-term"
        if x.dtype != mx.float16 or inner.scales.dtype != mx.float16:
            return "dtype"
        d_in = x.shape[-1]
        if d_in % 64:
            return f"din-{d_in}"
        m = x.size // d_in          # the flattened rows, however they are spelled
        d_out = inner.scales.shape[0]
        if not should_dispatch(m, inner.bits, d_out, d_in):
            return f"m-{m}-outside-dispatch-{d_out}x{d_in}"
        return None

    def _fused(self, x):
        inner = self.inner
        d_in = x.shape[-1]
        x2 = x.reshape(-1, d_in)
        m = x2.shape[0]
        d_out = inner.scales.shape[0]
        grid, tg, r = launch_config(d_out, m)
        out = _KERNEL(
            inputs=[x2, inner.weight, inner.scales, inner.biases],
            output_shapes=[(m, d_out)], output_dtypes=[x.dtype],
            grid=grid, threadgroup=tg,
            template=[("T", x.dtype), ("BITS", inner.bits),
                      ("M", m), ("R", r)])[0]
        return out.reshape(*x.shape[:-1], d_out)

    def __call__(self, x):
        why = self._ineligible(x)
        if why is None:
            self.patch.calls += 1
            return self._fused(x)
        self.patch.fallbacks[why] = self.patch.fallbacks.get(why, 0) + 1
        return self.inner(x)


class SeqPatch:
    """The same install/uninstall shape as serve_sub4bit's, kept separate."""

    def __init__(self, model):
        self.calls = 0
        self.fallbacks: dict[str, int] = {}
        self._sites: list[tuple] = []
        self._wrap(model)
        self.n_wrapped = len(self._sites)

    def _wrap(self, node):
        if isinstance(node, (nn.Module, dict)):
            entries = [(node, k, v) for k, v in node.items()]
        elif isinstance(node, (list, tuple)):
            entries = [(node, i, v) for i, v in enumerate(node)]
        else:
            return
        for parent, key, value in entries:
            if isinstance(value, nn.QuantizedLinear):
                self._replace(parent, key, _SeqRoutedLinear(value, self))
                self._sites.append((parent, key, value))
            elif isinstance(value, (nn.Module, list, tuple, dict)):
                self._wrap(value)

    @staticmethod
    def _replace(parent, key, value):
        if isinstance(parent, nn.Module):
            setattr(parent, key, value)
        else:
            parent[key] = value

    def uninstall(self):
        for parent, key, original in self._sites:
            self._replace(parent, key, original)
        self._sites = []


def verify_step(model, prompt, steps: int) -> float:
    """Time `steps` verification-shaped passes: VERIFY_M tokens, one stream.

    The cache is primed once outside the timed window, exactly as the A/B
    excludes prefill and the first token.
    """
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(prompt, cache=cache)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    block = mx.tile(y[:, None], (1, VERIFY_M))
    logits = model(block, cache=cache)      # warm this shape's compile
    mx.eval(logits)

    t0 = time.perf_counter()
    for _ in range(steps):
        logits = model(block, cache=cache)
        mx.eval(logits)
    return time.perf_counter() - t0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-gb", type=budget_gb_arg, default=BUDGET_GB)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        manifest = verify_pins(MODELS_ROOT, [MODEL_3BIT])
    except RuntimeError as e:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): {e}")
        return EXIT_PRECONDITION

    lock = MeasurementLock(LOCK_NAME)
    acquired, detail = lock.acquire()
    if not acquired:
        print(f"REFUSAL (exit {EXIT_LOCK_HELD}): machine measurement lock "
              f"{detail}; one heavy measurement at a time")
        return EXIT_LOCK_HELD
    try:
        require_idle("spike-start")
        guard = BudgetGuard(args.budget_gb)
        guard("spike-start")
        print(json.dumps(provenance(manifest)))

        model, tok = load_model(MODEL_3BIT)
        prompt = make_prompts(tok, PROMPT_T, 1)

        patch = SeqPatch(model)
        routed_calls_probe = patch.calls
        s_time = verify_step(model, prompt, 1)       # one pass to count routing
        routed = patch.calls - routed_calls_probe
        fallbacks = dict(patch.fallbacks)
        patch.uninstall()
        print(json.dumps({"verify_m": VERIFY_M, "wrapped": patch.n_wrapped,
                          "routed_calls_one_pass": routed,
                          "fallbacks_one_pass": fallbacks}))
        if routed == 0:
            print("STOP: the verification shape routed nothing even with the "
                  "eligibility rule relaxed; the spike's premise is wrong and "
                  "no timing would be readable")
            return 1

        # Interleaved by hand, the way mde() does it, and NOT through
        # interleave.interleaved_samples: that sampler evaluates pure array
        # expressions it can replay, and a forward pass mutates a KV cache, so
        # it cannot be replayed. Interleaving within the round is the property
        # that matters and it is preserved here - a clock excursion lands on
        # both arms, and the per-arm spread below is what would expose one.
        spec_ms, stock_ms = [], []
        for i in range(ROUNDS):
            guard(f"round {i + 1}/{ROUNDS}")
            require_available_memory(2.0, f"round {i + 1}")

            p = SeqPatch(model)
            spec_ms.append(verify_step(model, prompt, args.steps) * 1e3
                           / args.steps)
            p.uninstall()

            stock_ms.append(verify_step(model, prompt, args.steps) * 1e3
                            / args.steps)
        guard("done")
        m_spec, m_stock = statistics.median(spec_ms), statistics.median(stock_ms)
        sp_spec, sp_stock = spread_pct(spec_ms), spread_pct(stock_ms)
        ratio = m_stock / m_spec
        gain_pct = (ratio - 1.0) * 100.0
        floor = max(sp_spec, sp_stock)
        if abs(gain_pct) <= floor:
            verdict = "INCONCLUSIVE"
        elif gain_pct > 0:
            verdict = "GO"
        else:
            verdict = "NO-GO"
        print(json.dumps({
            "verify_m": VERIFY_M, "rounds": ROUNDS,
            "spec_ms_median": round(m_spec, 4),
            "stock_ms_median": round(m_stock, 4),
            "spec_spread_pct": round(sp_spec, 3),
            "stock_spread_pct": round(sp_stock, 3),
            "ratio_stock_over_spec": round(ratio, 4),
            "gain_pct": round(gain_pct, 3),
            "noise_floor_pct": round(floor, 3),
            "verdict": verdict,
        }))
        return 0
    except NotIdle as e:
        print(f"REFUSAL (exit {EXIT_NOT_IDLE}): {e}")
        return EXIT_NOT_IDLE
    except PreconditionFailed as e:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): {e}")
        return EXIT_PRECONDITION
    except BudgetExceeded as e:
        print(f"REFUSAL (exit {EXIT_BUDGET_REFUSAL}): {e}")
        return EXIT_BUDGET_REFUSAL
    except LowMemoryRefusal as e:
        print(f"REFUSAL (exit {EXIT_LOW_MEMORY}): {e}")
        return EXIT_LOW_MEMORY
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
