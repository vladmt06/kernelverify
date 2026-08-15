"""The mlx-lm end-to-end spike: does the fused kv_attention kernel move
decode tokens/s on Qwen3-4B?

Pre-registered (lane C ruling, design doc vlad-next-lanes-design-20260814.md,
D4/D9/D12); the findings live in docs/research/2026-08-15-mlx-e2e-findings.md
and the MDE derivation and demotion rule are written there BEFORE any A/B
number is recorded.

The patch is runtime-only: no mlx-lm file is edited. Two seams are hooked at
import time and controlled by a flag, so one process hosts every arm:

- `QuantizedKVCache.update_and_fetch` stashes the step's own post-rope
  full-precision K/V on the cache instance before delegating; the contract's
  new entry enters the fused kernel at full precision.
- `qwen3.scaled_dot_product_attention` (the name qwen3.py imported) becomes a
  router: below the cache update, per ruling D12.1. When the fused arm is on
  and the call is eligible (decode-shaped L=1, B=1, quantized cache at group
  size 64, supported bits and head dim, no mask arithmetic), it launches the
  one-dispatch kernel on the PADDED cache buffers with the logical length as
  a scalar - never slicing, which would charge the stock path's growing
  contiguity copies to our arm. Anything else falls through to stock, and
  every fallback is counted by reason: an arm that silently stopped being
  the arm is the first thing this harness must catch.

Arms (ruling D4):
  A  fused kernel, quantized KV cache (the candidate)
  B  stock mlx-lm, quantized KV cache (the incumbent; primary ratio is A/B)
  C  stock mlx-lm, fp16 cache (labeled third arm, never merged into A/B;
     demotion rule D12.7: if C beats A, the result is demoted regardless
     of A/B)

Method: generation-level A/B, arms interleaved within every round (separate
passes measure the clock, not the kernels - AGENTS.md), driven through
`generate_step` directly so no EOS logic can cut a round short. Decode tps
excludes prefill and the first token. Arm B is the round canary: rows whose
B spread exceeds MAX_SPREAD_PCT are withheld. Timing modes refuse to run
unless `machine_state.idle_check` is clean before and after (D12.2).

T=1024 is banned (ruling D9): the old shape overruns the compile-time
softmax buffer after one generated token. Generation lengths keep the final
step's logical cache length at or under TCAP.

Modes:
  --smoke   correctness wiring: counters must show every decode step fused,
            and a shadow round reports the fused-vs-stock output gap.
  --mde     the pre-measurement arithmetic: per-op attention times at the
            true E2E shapes, arm-B per-token time and noise floor; prints
            the numbers the findings doc's MDE section transcribes.
  --ab      the pre-registered probe: interleaved 3-arm rounds over
            {4,8} bits x {512,896} prompt tokens.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import machine_state  # noqa: E402
import mlx.core as mx  # noqa: E402

from interleave import calibrate_copies, dispatch  # noqa: E402
from machine_state import spread_pct  # noqa: E402
from kernelverify.pack.kv_attention import (  # noqa: E402
    SUPPORTED_BITS,
    SUPPORTED_DH,
    build,
    launch_config,
    should_dispatch,
)

MODEL = "mlx-community/Qwen3-4B-4bit"
GROUP_SIZE = 64
CONFIGS = [(bits, t) for bits in (8, 4) for t in (512, 896)]
# Final decode step reads logical length T + G <= TCAP (the pipelined
# lookahead step in generate_step is the +G, not +G-1).
GEN_TOKENS = {512: 480, 896: 120}
ROUNDS = {512: 5, 896: 9}
MAX_SPREAD_PCT = machine_state.MAX_SPREAD_PCT

PROMPT_SEED = (
    "The verification of GPU kernels is a study in mismatched incentives: "
    "the benchmark that ships with a kernel was written by whoever wrote the "
    "kernel, and it passes. What escapes is decided by the pair of bug and "
    "test case, never by the bug alone, so the only honest measure of a test "
    "policy is how it scores against a fault population it did not choose. "
)

# ---------------------------------------------------------------------------
# the runtime patch
# ---------------------------------------------------------------------------
from mlx_lm.models import qwen3  # noqa: E402
from mlx_lm.models.cache import QuantizedKVCache  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402

_ORIG_SDPA = qwen3.scaled_dot_product_attention
_ORIG_UPDATE = QuantizedKVCache.update_and_fetch
_KERNEL = build(mx)

SPIKE = {"fused": False, "shadow": False, "calls": 0, "fallback": {},
         "shadow_fused_ref": 0.0, "shadow_stock_ref": 0.0,
         "shadow_ref_mag": 0.0}


def hard_fallbacks() -> dict:
    """Fallbacks that mean the fused arm stopped being the fused arm.
    Prefill chunks (L > 1) route to stock by design and are whitelisted."""
    return {k: v for k, v in SPIKE["fallback"].items()
            if not k.startswith("prefill-")}


def _stashing_update(self, keys, values):
    self._spike_new_k = keys
    self._spike_new_v = values
    return _ORIG_UPDATE(self, keys, values)


def _ineligible(queries, cache, scale, mask, sinks):
    """None when the fused kernel takes this call; otherwise the reason.
    L is checked before anything else so prefill chunks are labeled as
    prefill regardless of the cache's state at that moment."""
    b, h, l, d = queries.shape
    if l != 1:
        return f"prefill-L{l}"
    if not isinstance(cache, QuantizedKVCache):
        return "cache-not-quantized"
    if getattr(cache, "_spike_new_k", None) is None:
        return "no-stashed-new-entry"
    if b != 1:
        return f"batch-B{b}"
    if d not in SUPPORTED_DH:
        return f"head-dim-{d}"
    if cache.bits not in SUPPORTED_BITS or cache.group_size != GROUP_SIZE:
        return f"quant-{cache.bits}b-gs{cache.group_size}"
    if mask is not None and mask != "causal":
        return "mask-array"
    if sinks is not None:
        return "sinks"
    if abs(scale * (d ** 0.5) - 1.0) > 1e-6:
        return "nonstandard-scale"
    tc = cache.offset - 1
    if tc < 1 or not should_dispatch(tc):
        return f"t-{tc}-outside-dispatch"
    return None


def _fused(queries, cache):
    b, h, l, d = queries.shape
    tc = cache.offset - 1
    hkv = cache.keys[0].shape[1]
    drop_b = lambda a: a.reshape(*a.shape[1:])  # (1, HKV, T_pad, w) -> 3-D
    grid, tg = launch_config(b, h)
    out = _KERNEL(
        inputs=[queries.reshape(b, h, d),
                *(drop_b(a) for a in cache.keys),
                *(drop_b(a) for a in cache.values),
                cache._spike_new_k.reshape(b, hkv, d),
                cache._spike_new_v.reshape(b, hkv, d)],
        t_cached=tc,
        output_shapes=[(b, h, d)], output_dtypes=[queries.dtype],
        grid=grid, threadgroup=tg,
        template=[("T", queries.dtype), ("BITS", cache.bits), ("DH", d)])[0]
    return out.reshape(b, h, 1, d)


def _shadow_ref(queries, cache):
    """fp64 composition of the contract semantics on the dequantized cache:
    the yardstick that separates 'our wiring is wrong' from 'stock's own
    fp16-score arithmetic sits far from the contract'."""
    _b, h, _l, d = queries.shape
    tc = cache.offset - 1
    bits, gs = cache.bits, cache.group_size
    hkv = cache.keys[0].shape[1]
    r = h // hkv
    f64 = lambda a: np.array(a.astype(mx.float32), dtype=np.float64)
    deq = lambda triplet: np.repeat(
        f64(mx.dequantize(*triplet, group_size=gs,
                          bits=bits))[0, :, :tc], r, axis=0)   # (H, tc, D)
    kk, vv = deq(cache.keys), deq(cache.values)
    q = f64(queries)[0, :, 0]                                  # (H, D)
    nk = np.repeat(f64(cache._spike_new_k)[0, :, 0], r, axis=0)
    nv = np.repeat(f64(cache._spike_new_v)[0, :, 0], r, axis=0)
    scale = d ** -0.5
    sc = np.concatenate(
        [np.einsum("hd,htd->ht", q, kk),
         np.einsum("hd,hd->h", q, nk)[:, None]], axis=1) * scale
    sc -= sc.max(axis=1, keepdims=True)
    p = np.exp(sc)
    p /= p.sum(axis=1, keepdims=True)
    return np.einsum("ht,htd->hd", p[:, :tc], vv) + p[:, tc:] * nv


def _router(queries, keys, values, cache=None, scale=1.0, mask=None,
            sinks=None):
    if SPIKE["fused"]:
        why = _ineligible(queries, cache, scale, mask, sinks)
        if why is None:
            SPIKE["calls"] += 1
            out = _fused(queries, cache)
            if SPIKE["shadow"]:
                # The ref must be taken BEFORE stock runs: stock's
                # `queries *= scale` mutates the caller's array in place,
                # and a ref taken after it double-scales the scores.
                ref = _shadow_ref(queries, cache)
                stock = _ORIG_SDPA(queries, keys, values, cache=cache,
                                   scale=scale, mask=mask, sinks=sinks)
                gap = lambda a: float(np.max(np.abs(
                    np.array(a.astype(mx.float32),
                             dtype=np.float64)[0, :, 0] - ref)))
                SPIKE["shadow_fused_ref"] = max(SPIKE["shadow_fused_ref"],
                                                gap(out))
                SPIKE["shadow_stock_ref"] = max(SPIKE["shadow_stock_ref"],
                                                gap(stock))
                SPIKE["shadow_ref_mag"] = max(SPIKE["shadow_ref_mag"],
                                              float(np.max(np.abs(ref))))
                return stock
            return out
        SPIKE["fallback"][why] = SPIKE["fallback"].get(why, 0) + 1
    return _ORIG_SDPA(queries, keys, values, cache=cache, scale=scale,
                      mask=mask, sinks=sinks)


def install_patch():
    qwen3.scaled_dot_product_attention = _router
    QuantizedKVCache.update_and_fetch = _stashing_update


def reset_counters():
    SPIKE.update(calls=0, fallback={}, shadow_fused_ref=0.0,
                 shadow_stock_ref=0.0)


# ---------------------------------------------------------------------------
# generation driver
# ---------------------------------------------------------------------------
def load_model():
    from mlx_lm import load
    model, tokenizer = load(MODEL)
    # The checkpoint activations are bfloat16; the frozen contract's dtypes
    # are float16/float32, so the whole model is cast to fp16 for EVERY arm.
    # Same lane width on Metal, so tokens/s transfers; recorded in findings.
    model.set_dtype(mx.float16)
    return model, tokenizer


def make_prompt(tokenizer, t: int) -> list[int]:
    ids = tokenizer.encode(PROMPT_SEED * 40)
    if len(ids) < t:
        raise RuntimeError(f"prompt seed too short: {len(ids)} < {t}")
    return ids[:t]


def decode_tps(model, prompt_ids, gen_tokens: int, kv_bits: int | None,
               fused: bool) -> float:
    """One generation; returns decode tokens/s over gen_tokens - 1 steps
    (prefill and the first token are excluded from the timed window)."""
    SPIKE["fused"] = fused
    kwargs = dict(max_tokens=gen_tokens,
                  sampler=lambda x: mx.argmax(x, axis=-1))
    if kv_bits is not None:
        kwargs.update(kv_bits=kv_bits, kv_group_size=GROUP_SIZE,
                      quantized_kv_start=0)
    gen = generate_step(mx.array(prompt_ids), model, **kwargs)
    next(gen)                        # prefill + first token
    n = 0
    t0 = time.perf_counter()
    for _tok, _lp in gen:
        n += 1
    dt = time.perf_counter() - t0
    SPIKE["fused"] = False
    mx.clear_cache()
    return n / dt


def require_idle(label: str) -> dict:
    state = machine_state.idle_check()
    if not state["idle"]:
        print(f"NOT QUIET ({label}): " + "; ".join(state["blockers"]))
        print("Timing refused (ruling D12.2). Do non-timing prep, then rerun "
              "in a quiet window.")
        raise SystemExit(2)
    return state


def provenance(model) -> dict:
    import mlx_lm
    return {"model": MODEL, "n_layers": len(model.layers),
            "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
            "machine": machine_state.fingerprint()}


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def smoke(model, tokenizer) -> int:
    """Wiring correctness, no timing: every decode step must go fused, and
    the shadow gap says the fused output sits where stock sits (the oracle
    verdicts live in the pack gate; this catches integration mistakes -
    transposed heads, stale stash, wrong stride)."""
    prompt = make_prompt(tokenizer, 64)
    n_layers = len(model.layers)

    reset_counters()
    SPIKE["shadow"] = True
    decode_tps(model, prompt, 8, kv_bits=8, fused=True)
    SPIKE["shadow"] = False
    calls, hard = SPIKE["calls"], hard_fallbacks()
    steps = calls // n_layers
    print(f"shadow round: {calls} fused calls over {steps} decode-shaped "
          f"steps of {n_layers} layers; fallbacks: "
          f"{dict(SPIKE['fallback']) or 'none'} (prefill is whitelisted)")
    # A wiring mistake (transposed heads, stale stash, wrong stride) sits at
    # the scale of the output itself; the kernel's honest distance is a few
    # fp16 ulps of the largest output, mag * 2^-10 each. Gate at 4 ulps.
    bound = 4e-3 * max(1.0, SPIKE["shadow_ref_mag"])
    print(f"max gap to the fp64 contract reference: fused "
          f"{SPIKE['shadow_fused_ref']:.5f} (gate {bound:.5f} at output "
          f"magnitude {SPIKE['shadow_ref_mag']:.1f}), stock "
          f"{SPIKE['shadow_stock_ref']:.5f} (stock differs legitimately: "
          f"fp16 scores, and it quantizes the step's own entry)")
    ok = (calls > 0 and calls % n_layers == 0 and not hard
          and SPIKE["shadow_fused_ref"] < bound)

    # Non-shadow pass: the fused output actually feeds the model.
    reset_counters()
    decode_tps(model, prompt, 8, kv_bits=4, fused=True)
    ok = ok and SPIKE["calls"] > 0 and not hard_fallbacks()
    print(f"fused round (4-bit): {SPIKE['calls']} calls, hard fallbacks: "
          f"{hard_fallbacks() or 'none'}")
    print("SMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def attention_op_probe(model, bits: int, t: int) -> dict:
    """Per-op stock vs fused attention time at the true E2E shape, padded
    buffers and all, for the MDE arithmetic. Interleaved and batched past
    5 ms like every comparative probe in this repo."""
    n_layers = len(model.layers)
    h, hkv, dh = 32, 8, 128
    t_pad = (t + 255) // 256 * 256
    from mlx_lm.models.base import quantized_scaled_dot_product_attention

    rng = np.random.default_rng(t + bits)
    n_sets = 8
    sets = []
    for _ in range(n_sets):
        kq = mx.quantize(mx.array(rng.standard_normal((1, hkv, t_pad, dh))
                                  .astype(np.float16)), GROUP_SIZE, bits)
        vq = mx.quantize(mx.array(rng.standard_normal((1, hkv, t_pad, dh))
                                  .astype(np.float16)), GROUP_SIZE, bits)
        mx.eval(*kq, *vq)
        sets.append((kq, vq))
    q = mx.array(rng.standard_normal((1, h, 1, dh)).astype(np.float16) * 0.1)
    nk = mx.array(rng.standard_normal((1, hkv, dh)).astype(np.float16) * 0.1)
    nv = mx.array(rng.standard_normal((1, hkv, dh)).astype(np.float16))
    mx.eval(q, nk, nv)
    scale = dh ** -0.5
    grid, tg = launch_config(1, h)

    def stock(i):
        kq, vq = sets[i % n_sets]
        sliced_k = tuple(x[..., :t, :] for x in kq)
        sliced_v = tuple(x[..., :t, :] for x in vq)
        return quantized_scaled_dot_product_attention(
            q, sliced_k, sliced_v, scale=scale, mask=None,
            group_size=GROUP_SIZE, bits=bits)

    def fused(i):
        kq, vq = sets[i % n_sets]
        drop_b = lambda a: a.reshape(*a.shape[1:])
        return _KERNEL(
            inputs=[q.reshape(1, h, dh), *(drop_b(a) for a in kq),
                    *(drop_b(a) for a in vq), nk, nv],
            t_cached=t, output_shapes=[(1, h, dh)],
            output_dtypes=[mx.float16], grid=grid, threadgroup=tg,
            template=[("T", mx.float16), ("BITS", bits), ("DH", dh)])[0]

    copies = calibrate_copies(lambda c: dispatch(stock, c))
    a_s, b_s = [], []
    for _ in range(7):
        a_s.append(dispatch(fused, copies) / copies)
        b_s.append(dispatch(stock, copies) / copies)
    return {"bits": bits, "t": t, "n_layers": n_layers,
            "stock_us": statistics.median(b_s) * 1e6,
            "fused_us": statistics.median(a_s) * 1e6,
            "stock_spread_pct": round(spread_pct(b_s), 1),
            "op_ratio": statistics.median(b_s) / statistics.median(a_s)}


def mde(model, tokenizer) -> int:
    """The numbers the findings doc's MDE section is written from: per-op
    attention times, arm-B decode tps and its round-to-round noise floor."""
    require_idle("before")
    print(json.dumps(provenance(model)))
    for bits, t in CONFIGS:
        probe = attention_op_probe(model, bits, t)
        prompt = make_prompt(tokenizer, t)
        g = GEN_TOKENS[t]
        tps = [decode_tps(model, prompt, g, kv_bits=bits, fused=False)
               for _ in range(4)]
        per_tok_ms = 1e3 / statistics.median(tps)
        attn_ms = probe["n_layers"] * probe["stock_us"] / 1e3
        saving_ms = (probe["n_layers"]
                     * (probe["stock_us"] - probe["fused_us"]) / 1e3)
        print(json.dumps({
            **{k: round(v, 2) if isinstance(v, float) else v
               for k, v in probe.items()},
            "armB_tps": round(statistics.median(tps), 2),
            "armB_noise_floor_pct": round(spread_pct(tps), 2),
            "per_token_ms": round(per_tok_ms, 3),
            "attn_share_pct": round(100 * attn_ms / per_tok_ms, 1),
            "expected_tps_gain_pct": round(100 * saving_ms / per_tok_ms, 1),
        }))
    after = machine_state.idle_check()
    if not after["idle"]:
        print("WARNING: machine went non-idle during the probe: "
              + "; ".join(after["blockers"]))
        return 1
    return 0


def ab(model, tokenizer) -> int:
    """The pre-registered probe. Interleaved [A, B, C] per round; arm B is
    the canary; rows over the spread limit are withheld, not published."""
    require_idle("before")
    print(json.dumps(provenance(model)))
    exit_code = 0
    for bits, t in CONFIGS:
        prompt = make_prompt(tokenizer, t)
        g, rounds = GEN_TOKENS[t], ROUNDS[t]
        a_v, b_v, c_v = [], [], []
        reset_counters()
        for _ in range(rounds):
            a_v.append(decode_tps(model, prompt, g, kv_bits=bits, fused=True))
            b_v.append(decode_tps(model, prompt, g, kv_bits=bits, fused=False))
            c_v.append(decode_tps(model, prompt, g, kv_bits=None, fused=False))
        row = {"bits": bits, "t": t, "gen": g, "rounds": rounds,
               "fused_calls": SPIKE["calls"],
               "hard_fallbacks": hard_fallbacks()}
        sp = spread_pct(b_v)
        if hard_fallbacks():
            row["verdict"] = "INVALID: arm A fell back to stock"
            exit_code = 1
        elif sp > MAX_SPREAD_PCT:
            row["verdict"] = (f"WITHHELD: canary spread {sp:.1f}% > "
                              f"{MAX_SPREAD_PCT}%")
            exit_code = 1
        else:
            med = statistics.median
            row.update(
                armA_tps=round(med(a_v), 2), armB_tps=round(med(b_v), 2),
                armC_fp16_tps=round(med(c_v), 2),
                armA_spread_pct=round(spread_pct(a_v), 2),
                armB_spread_pct=round(sp, 2),
                armC_spread_pct=round(spread_pct(c_v), 2),
                ratio_AB=round(med(a_v) / med(b_v), 4),
                demotion_C_beats_A=med(c_v) > med(a_v))
        print(json.dumps(row))
    after = machine_state.idle_check()
    if not after["idle"]:
        print("WARNING: machine went non-idle during the probe: "
              + "; ".join(after["blockers"]))
        exit_code = 1
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--mde", action="store_true")
    mode.add_argument("--ab", action="store_true")
    args = parser.parse_args(argv)

    install_patch()
    model, tokenizer = load_model()
    if args.smoke:
        return smoke(model, tokenizer)
    if args.mde:
        return mde(model, tokenizer)
    return ab(model, tokenizer)


if __name__ == "__main__":
    raise SystemExit(main())
