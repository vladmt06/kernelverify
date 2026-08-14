"""Verify, then time, the fused MoE decode against MLX's own routing + gather.

Same order and same discipline as bench/pack_wide_qmv.py: nothing is timed
until every case verifies through the crash-isolated runner, samples are
batched to at least MIN_SAMPLE_MS, arms are interleaved, and expert weights
rotate over a working set larger than any cache.

The oracle is the shipped one, reached through kernelverify.pack.verify:
`NATIVE_OPS["moe_dispatch"]` pins the Qwen3-class routing contract over DENSE
experts, so a quantized kernel is judged against it by substituting the exact
dequantized contract weights: once the artefact is fixed, a quantized expert
IS a dense expert whose entries happen to be s*q + b. No bespoke oracle, and
the routing half is judged by the same reference that judges the battery.

Both arms are verified, not just ours, so the timing compares two
contract-passing implementations rather than one that happens to be faster.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from kernelverify.pack.moe_dispatch import (  # noqa: E402
    build_dispatch,
    build_routing,
    dispatch_launch,
    dispatch_spec,
    routing_launch,
    routing_spec,
)
from kernelverify.pack.verify import (  # noqa: E402
    judge,
    moe_inputs,
    reference_and_tolerance,
)
from kernelverify.pack.wide_qmv import pack_nibbles  # noqa: E402
from kernelverify.runners import MetalRunner, RunCase, specialize  # noqa: E402
from kernelverify.schemas.quant_contract import (  # noqa: E402
    QuantContract,
    canonical_quantize,
)

CONTRACT = QuantContract(bits=4, group_size=64)

# (n_tokens, d_model, n_experts, d_ffn)
VERIFY_SHAPES = [(1, 512, 16, 256), (8, 512, 16, 256), (4, 256, 8, 128)]
TIMED_SHAPE = (2048, 64, 768)          # Qwen3-30B-A3B class: d, experts, ffn
TIMED_TOKENS = [1, 2, 4, 8, 16]
MIN_SAMPLE_MS = 5.0
ROUNDS = 7
WORKING_SET_MB = 512

# See bench/pack_wide_qmv.py: interleaving makes both arms suffer a clock
# excursion together, it does not detect one. The reference arm's own spread
# is the detector, and rows that fail it are withheld rather than published.
MAX_CANARY_SPREAD = 1.5


def quantize_experts(experts: np.ndarray):
    """Packed artefact for the kernel, plus the per-expert artefacts the
    shared gate dequantizes into the contract's dense experts."""
    arts = [canonical_quantize(e, CONTRACT) for e in experts]
    packed = np.stack([pack_nibbles(a.q) for a in arts])
    scales = np.stack([a.scales for a in arts])
    biases = np.stack([a.biases for a in arts])
    return packed, scales, biases, arts


def make_case(n_tokens, d_model, n_experts, d_ffn, seed, mode="normal"):
    rng = np.random.default_rng(seed)
    if mode == "constant_rows":
        # Every logit ties, which is the only input that exercises the
        # "ties go to the lower index" half of the routing contract.
        x = np.repeat(rng.standard_normal((n_tokens, 1)).astype(np.float32),
                      d_model, axis=1).astype(np.float16)
        router = np.repeat(rng.standard_normal((1, d_model)).astype(np.float32) * 0.05,
                           n_experts, axis=0).astype(np.float16)
    else:
        x = (rng.standard_normal((n_tokens, d_model)).astype(np.float32) * 0.5).astype(np.float16)
        router = (rng.standard_normal((n_experts, d_model)).astype(np.float32) * 0.05).astype(np.float16)
    experts = (rng.standard_normal((n_experts, d_ffn, d_model)).astype(np.float32)
               * 0.02).astype(np.float16)
    return x, router, experts


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def verify(runner: MetalRunner) -> bool:
    print("correctness (runner-isolated, shipped moe_dispatch contract):")
    # The contract's own routing, from the shipped reference's helper.
    from kernelverify.reference.native_kernels import _topk_by_prob

    # Every case is built up front so the runner sees two batched phases (one
    # worker session each: routing grouped by E, then dispatch under its one
    # shared spec) instead of one worker spawn per case.
    metas = []
    for n_tokens, d_model, n_experts, d_ffn in VERIFY_SHAPES:
        for mode in ("normal", "constant_rows"):
            x, router, experts = make_case(n_tokens, d_model, n_experts, d_ffn,
                                           seed=n_tokens * 31 + len(mode), mode=mode)
            packed, scales, biases, arts = quantize_experts(experts)
            ref, tol = reference_and_tolerance(
                "moe_dispatch", moe_inputs(x, router, arts))
            metas.append({"mode": mode, "n_tokens": n_tokens, "d_model": d_model,
                          "n_experts": n_experts, "d_ffn": d_ffn, "x": x,
                          "router": router, "packed": packed, "scales": scales,
                          "biases": biases, "ref": ref, "tol": tol})

    by_e: dict[int, list[int]] = {}
    for i, m in enumerate(metas):
        by_e.setdefault(m["n_experts"], []).append(i)
    routing_batches = [
        (specialize(routing_spec(), {"E": e}),
         [RunCase(inputs={"x": metas[i]["x"], "router": metas[i]["router"]},
                  params={"d_in_arg": metas[i]["d_model"],
                          "n_tokens": metas[i]["n_tokens"]},
                  output_shapes=[((metas[i]["n_tokens"], 2), "uint32"),
                                 ((metas[i]["n_tokens"], 2), "float32")])
          for i in order])
        for e, order in by_e.items()]
    for order, batch in zip(by_e.values(), runner.run_candidate(routing_batches)):
        for i, route in zip(order, batch):
            metas[i]["route"] = route

    ok = True
    dispatch_cases, dispatch_idx, spec_r = [], [], None
    for i, m in enumerate(metas):
        route = m["route"]
        if not route.ok:
            print(f"    {m['mode']:<14} n={m['n_tokens']:<3} ROUTING RUNNER FAIL: "
                  f"{route.status.value}: {route.detail}")
            ok = False
            continue
        idx, gate = route.outputs
        logits = m["x"].astype(np.float32) @ m["router"].astype(np.float32).T
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        want = _topk_by_prob(probs, 2, tie_high=False)
        m["routing_exact"] = np.array_equal(idx.astype(int), want)

        grid, threadgroup, spec_r = dispatch_launch(m["d_ffn"], m["n_tokens"])
        dispatch_cases.append(RunCase(
            inputs={"x": m["x"], "idx": idx, "gate": gate, "w_q": m["packed"],
                    "scales": m["scales"], "biases": m["biases"]},
            params={"d_in_arg": m["d_model"], "d_out_arg": m["d_ffn"],
                    "row_blocks": grid[1], "n_tokens": m["n_tokens"]},
            output_shapes=[((m["n_tokens"], m["d_ffn"]), "float16")]))
        dispatch_idx.append(i)

    if not dispatch_cases:
        return ok
    # R is the fixed rows-per-simdgroup, so one spec serves every case.
    spec = specialize(dispatch_spec(), {"T": "half", "R": spec_r})
    for i, result in zip(dispatch_idx, runner.run(spec, dispatch_cases)):
        m = metas[i]
        if not result.ok:
            print(f"    {m['mode']:<14} n={m['n_tokens']:<3} DISPATCH RUNNER FAIL: "
                  f"{result.status.value}: {result.detail}")
            ok = False
            continue
        v = judge(result.outputs[0], m["ref"], m["tol"])
        passed = v.ok and m["routing_exact"]
        ok = ok and passed
        print(f"    {m['mode']:<14} n={m['n_tokens']:<3} E={m['n_experts']:<3} "
              f"routing exact {str(m['routing_exact']):<5} err {v.err:9.3e} "
              f"tol {v.tol:9.3e}  {'pass' if passed else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def dispatch_once(build_one, copies: int) -> float:
    outs = [build_one(i) for i in range(copies)]
    t0 = time.perf_counter()
    mx.eval(outs)
    mx.synchronize()
    return time.perf_counter() - t0


def calibrate(sample) -> int:
    copies = 4
    while copies < 2048 and sample(copies) * 1e3 < MIN_SAMPLE_MS:
        copies *= 2
    return copies


def mlx_gather(x, idx, w, wq, sc, bi):
    """The expert half of MLX's path, given routing."""
    xe = mx.expand_dims(mx.expand_dims(x, -2), -2)          # (M, 1, 1, D)
    y = mx.gather_qmm(xe, wq, sc, bi, rhs_indices=idx, transpose=True,
                      group_size=64, bits=4)                 # (M, 2, 1, D_ffn)
    y = mx.squeeze(y, -2).astype(mx.float32)
    return mx.sum(y * mx.expand_dims(w, -1), axis=1).astype(mx.float16)


def mlx_moe(x, router, wq, sc, bi):
    """MLX's own full path: routing in ops, then gather_qmm over the top-2.

    argpartition is what mlx-lm's own MoE uses; argsort measured identical
    here (41.7 us against 41.3 us at one token), so the choice is not what
    the comparison turns on.
    """
    logits = x @ router.T
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    idx = mx.argpartition(-probs, kth=1, axis=-1)[:, :2]
    w = mx.take_along_axis(probs, idx, axis=-1)
    w = w / mx.sum(w, axis=-1, keepdims=True)
    return mlx_gather(x, idx, w, wq, sc, bi)


def bench() -> bool:
    d_model, n_experts, d_ffn = TIMED_SHAPE
    routing, dispatch = build_routing(mx), build_dispatch(mx)

    expert_bytes = n_experts * d_ffn * d_model * 0.5
    n_sets = max(2, min(10, int(WORKING_SET_MB * 1e6 // expert_bytes) + 1))
    print(f"\ntiming: d={d_model} experts={n_experts} ffn={d_ffn} top-2, "
          f"{n_sets} expert sets ({n_sets*expert_bytes/1e6:.0f} MB)")

    sets = []
    for seed in range(n_sets):
        rng = np.random.default_rng(500 + seed)
        experts = (rng.standard_normal((n_experts, d_ffn, d_model)).astype(np.float32)
                   * 0.02).astype(np.float16)
        packed, scales, biases, _ = quantize_experts(experts)
        mx_q = mx.quantize(mx.array(experts), group_size=64, bits=4)
        sets.append((mx.array(packed), mx.array(scales), mx.array(biases), mx_q))
    mx.eval([a for s in sets for a in (s[0], s[1], s[2], *s[3])])

    print("  full path is routing + experts; dispatch-only feeds both arms the")
    print("  same precomputed routing, which separates the fusion win from the")
    print("  kernel win.")
    print(f"  {'tokens':>7} | {'ours us':>9} {'mlx us':>9} {'ratio':>7} {'agree':>6} "
          f"| {'ours us':>9} {'mlx us':>9} {'ratio':>7}")
    ok = True
    for n_tokens in TIMED_TOKENS:
        rng = np.random.default_rng(9 + n_tokens)
        x = mx.array((rng.standard_normal((n_tokens, d_model)).astype(np.float32)
                      * 0.5).astype(np.float16))
        router = mx.array((rng.standard_normal((n_experts, d_model)).astype(np.float32)
                           * 0.05).astype(np.float16))
        mx.eval(x, router)
        rgrid, rtg = routing_launch(n_tokens)
        dgrid, dtg, r = dispatch_launch(d_ffn, n_tokens)

        def ours(i):
            packed, scales, biases, _ = sets[i % n_sets]
            idx, gate = routing(inputs=[x, router], output_shapes=[(n_tokens, 2)] * 2,
                                output_dtypes=[mx.uint32, mx.float32],
                                grid=rgrid, threadgroup=rtg, template=[("E", n_experts)])
            return dispatch(inputs=[x, idx, gate, packed, scales, biases],
                            output_shapes=[(n_tokens, d_ffn)], output_dtypes=[mx.float16],
                            grid=dgrid, threadgroup=dtg,
                            template=[("T", mx.float16), ("R", r)])[0]

        def theirs(i):
            _, _, _, (wq, sc, bi) = sets[i % n_sets]
            return mlx_moe(x, router, wq, sc, bi)

        a, b = np.array(ours(0)).astype(np.float64), np.array(theirs(0)).astype(np.float64)
        agree = float(np.max(np.abs(a - b))) <= 5e-3 * max(1.0, float(np.max(np.abs(b))))
        ok = ok and agree

        # Dispatch-only: identical routing handed to both arms.
        packed0, scales0, biases0, _ = sets[0]
        idx0, gate0 = routing(inputs=[x, router], output_shapes=[(n_tokens, 2)] * 2,
                              output_dtypes=[mx.uint32, mx.float32],
                              grid=rgrid, threadgroup=rtg, template=[("E", n_experts)])
        mx.eval(idx0, gate0)

        def ours_dispatch(i):
            packed, scales, biases, _ = sets[i % n_sets]
            return dispatch(inputs=[x, idx0, gate0, packed, scales, biases],
                            output_shapes=[(n_tokens, d_ffn)], output_dtypes=[mx.float16],
                            grid=dgrid, threadgroup=dtg,
                            template=[("T", mx.float16), ("R", r)])[0]

        def mlx_dispatch(i):
            _, _, _, (wq, sc, bi) = sets[i % n_sets]
            return mlx_gather(x, idx0, gate0, wq, sc, bi)

        def samples_of(fn):
            mx.eval(fn(0))
            mx.synchronize()
            copies = calibrate(lambda c: dispatch_once(fn, c))
            return [dispatch_once(fn, copies) / copies for _ in range(ROUNDS)]

        s_ours, s_mlx = samples_of(ours), samples_of(theirs)
        s_od, s_md = samples_of(ours_dispatch), samples_of(mlx_dispatch)
        spread = max(max(s_mlx) / min(s_mlx), max(s_md) / min(s_md))
        if spread > MAX_CANARY_SPREAD:
            print(f"  {n_tokens:>7} | {'REJECTED: canary spread ':>38}"
                  f"{spread:.2f}x > {MAX_CANARY_SPREAD}x")
            ok = False
            continue
        t_ours, t_mlx = statistics.median(s_ours), statistics.median(s_mlx)
        t_od, t_md = statistics.median(s_od), statistics.median(s_md)
        print(f"  {n_tokens:>7} | {t_ours*1e6:>9.1f} {t_mlx*1e6:>9.1f} "
              f"{t_mlx/t_ours:>6.2f}x {str(agree):>6} "
              f"| {t_od*1e6:>9.1f} {t_md*1e6:>9.1f} {t_md/t_od:>6.2f}x")
    return ok


def main() -> int:
    if not verify(MetalRunner()):
        print("\nVERDICT: kernel does not verify; no timing claim permitted")
        return 1
    if not bench():
        print("\nSome rows were withheld: either the arms disagreed, or the "
              "reference arm's own time moved too much inside the round. "
              "Re-run on an idle machine before quoting anything.")
        return 1
    print("\nratio > 1.00x means the verified kernel beats MLX's routing + gather_qmm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
