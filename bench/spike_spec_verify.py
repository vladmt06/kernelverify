"""Does routing a speculative-verification step buy anything at the step level?

Pre-registered in docs/research/2026-08-18-spec-verify-meeting-point-spike.md,
which fixes the outcomes before this file was written. Read it first.

The narrow question: a K=6 verification presents 7 tokens of ONE stream to
every projection, shape (1, 7, d_in). `wide_qmv` routes M = 5..9 and the
2026-08-17 A/B's binding grid measured 14.69% at M=6 and 15.65% at M=8, so M=7 is inside the
window and between two wins. When this spike was registered,
serve_sub4bit's gate refused the sequence spelling before it ever computed M.

There is NO kernel-level question here and this harness must not pretend one
exists: `_fused` flattens with `x.reshape(-1, d_in)`, so the two spellings
produce the same kernel call on the same array. What differs is everything
around the projection - 7 query positions against the cache rather than one -
so the honest question is whether the STEP gets faster, and it can answer no
even though the projection is faster in isolation.

The rule this spike questioned is now the shared serving rule under the section
4 amendment of 2026-08-18: `serve_sub4bit` computes the flattened M for every
input rank and leaves `should_dispatch` to decide. This harness now installs
that shared patch, so a rerun measures the interception used by serving.

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

from machine_state import (  # noqa: E402
    MeasurementLock,
    dispersion_verdict,
    spread_pct,
)
from memory_guard import (  # noqa: E402
    EXIT_BUDGET_REFUSAL,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    EXIT_NOT_IDLE,
    EXIT_PRECONDITION,
    BudgetExceeded,
    LowMemoryRefusal,
    budget_gb_arg,
)
from serve_sub4bit import (  # noqa: E402
    MODEL_3BIT,
    MODELS_ROOT,
    PINNED_BITS,
    NotIdle,
    PreconditionFailed,
    ServeGuard,
    install_patch,
    load_model,
    make_prompts,
    provenance,
    check_idle_after,
    require_idle,
    verify_pins,
)

LOCK_NAME = "spike_spec_verify"
BUDGET_GB = 24.0
ROUNDS = 5
PROMPT_T = 64
DRAFT_K = 6                 # the paper's optimum; verification is K + 1 tokens
VERIFY_M = DRAFT_K + 1
PROBE_PASSES = 2            # verify_step's compile warm-up, plus one timed step


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
        # ServeGuard, not BudgetGuard: it is the callable that takes the
        # footprint budget AND the machine memory gate in the order
        # price_qmv_boundary fixed, and restating half of it here is the
        # two-halves mistake AGENTS.md names.
        guard = ServeGuard(args.budget_gb)
        guard("spike-start")
        print(json.dumps(provenance(manifest)))

        model, tok = load_model(MODEL_3BIT)
        prompt = make_prompts(tok, PROMPT_T, 1)

        # The premise probe. verify_step runs a compile warm-up pass and then
        # `steps` timed ones, so a probe at steps=1 presents the verification
        # shape TWICE; the expected routed count is per pass times the passes,
        # and calling it "one pass" is what an earlier draft of the record got
        # wrong.
        patch = install_patch(model)
        routed_calls_probe = patch.calls
        verify_step(model, prompt, 1)
        routed = patch.calls - routed_calls_probe
        fallbacks = dict(patch.fallbacks)
        patch.uninstall()
        routed_per_pass = routed // PROBE_PASSES
        print(json.dumps({"verify_m": VERIFY_M, "wrapped": patch.n_wrapped,
                          "probe_passes": PROBE_PASSES,
                          "routed_calls_probe": routed,
                          "routed_calls_per_pass": routed_per_pass,
                          "fallbacks_probe": fallbacks}))
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

            p = install_patch(model)
            spec_ms.append(verify_step(model, prompt, args.steps) * 1e3
                           / args.steps)
            # A round that silently went part-stock would still produce a
            # number, and the median would absorb it. The count is exact and
            # cheap, so check it rather than trust it: one warm pass plus
            # `steps` timed ones, every wrapped site routing in each.
            want = routed_per_pass * (args.steps + PROBE_PASSES - 1)
            if p.calls != want:
                print(f"STOP: round {i + 1} routed {p.calls} calls, expected "
                      f"{want}; the spec arm was not fully routed and no "
                      f"timing from it is readable. Fallbacks: {p.fallbacks}")
                p.uninstall()
                return 1
            p.uninstall()

            stock_ms.append(verify_step(model, prompt, args.steps) * 1e3
                            / args.steps)
        guard("done")
        m_spec, m_stock = statistics.median(spec_ms), statistics.median(stock_ms)
        sp_spec, sp_stock = spread_pct(spec_ms), spread_pct(stock_ms)
        ratio = m_stock / m_spec
        gain_pct = (ratio - 1.0) * 100.0
        # The relative rule below only compares the gain to the arms' own
        # spread, so it would call a 20% gain readable between two arms that
        # each disperse 12%. machine_state's absolute gate is what the rest of
        # the repo uses for exactly that case: a median over samples that
        # disagree by more than MAX_SPREAD_PCT means nothing, however large the
        # gap between the two medians happens to be.
        for label, sp in (("spec", sp_spec), ("stock", sp_stock)):
            d = dispersion_verdict(sp)
            if d["over_spread_limit"]:
                print(f"STOP: the {label} arm's samples disperse {sp:.3f}%, "
                      f"over the {d['max_spread_pct']}% limit; its median is "
                      f"not a readable quantity and no verdict follows from it")
                return 1

        floor = max(sp_spec, sp_stock)
        if abs(gain_pct) <= floor:
            verdict = "INCONCLUSIVE"
        elif gain_pct > 0:
            verdict = "GO"
        else:
            verdict = "NO-GO"
        print(json.dumps({
            "verify_m": VERIFY_M, "rounds": ROUNDS,
            # --steps is settable and each step adds VERIFY_M cache
            # positions, so two runs at different values measure
            # different workloads; the value belongs in the record.
            "steps": args.steps,
            "spec_ms_median": round(m_spec, 4),
            "stock_ms_median": round(m_stock, 4),
            "spec_spread_pct": round(sp_spec, 3),
            "stock_spread_pct": round(sp_stock, 3),
            "ratio_stock_over_spec": round(ratio, 4),
            "gain_pct": round(gain_pct, 3),
            "noise_floor_pct": round(floor, 3),
            "verdict": verdict,
        }))
        # Clean before AND after, which is what the findings doc registers and
        # what marked the A/B's run 2 non-binding when its closing sample
        # caught WindowServer at 15% CPU.
        return check_idle_after(0)
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
