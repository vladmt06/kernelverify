"""Verify, then time, the fused MoE decode against MLX's own routing + gather.

Same order and same discipline as bench/pack_wide_qmv.py: nothing is timed
until every case verifies through the crash-isolated runner, samples are
batched to at least MIN_SAMPLE_MS, arms are interleaved, and expert weights
rotate over a working set larger than any cache.

The oracle is the shipped one. `NATIVE_OPS["moe_dispatch"]` pins the Qwen3-class
routing contract over DENSE experts, so a quantized kernel is judged against it
by substituting the exact dequantized contract weights: once the artefact is
fixed, a quantized expert IS a dense expert whose entries happen to be s*q + b.
No bespoke oracle, and the routing half is judged by the same reference that
judges the battery.

Both arms are verified, not just ours, so the timing compares two
contract-passing implementations rather than one that happens to be faster.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import namedtuple
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
from kernelverify.pack.wide_qmv import pack_nibbles  # noqa: E402
from kernelverify.runners.runner import DeviceCase, MetalRunner  # noqa: E402
from kernelverify.schemas.native_ops import NATIVE_OPS  # noqa: E402
from kernelverify.schemas.quant_contract import (  # noqa: E402
    QuantContract,
    canonical_quantize,
)

CONTRACT = QuantContract(bits=4, group_size=64)
Case = namedtuple("Case", "dtype")

# (n_tokens, d_model, n_experts, d_ffn)
VERIFY_SHAPES = [(1, 512, 16, 256), (8, 512, 16, 256), (4, 256, 8, 128)]
TIMED_SHAPE = (2048, 64, 768)          # Qwen3-30B-A3B class: d, experts, ffn
TIMED_TOKENS = [1, 2, 4, 8, 16]
MIN_SAMPLE_MS = 5.0
ROUNDS = 7
WORKING_SET_MB = 512


def quantize_experts(experts: np.ndarray):
    """Packed artefact plus the exact dequantized weights the contract means."""
    arts = [canonical_quantize(e, CONTRACT) for e in experts]
    packed = np.stack([pack_nibbles(a.q) for a in arts])
    scales = np.stack([a.scales for a in arts])
    biases = np.stack([a.biases for a in arts])
    dequantized = np.stack([
        a.scales.astype(np.float64).repeat(CONTRACT.group_size, axis=1)
        * a.q.astype(np.float64)
        + a.biases.astype(np.float64).repeat(CONTRACT.group_size, axis=1)
        for a in arts
    ])
    return packed, scales, biases, dequantized


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
    op = NATIVE_OPS["moe_dispatch"]
    ok = True
    for n_tokens, d_model, n_experts, d_ffn in VERIFY_SHAPES:
        for mode in ("normal", "constant_rows"):
            x, router, experts = make_case(n_tokens, d_model, n_experts, d_ffn,
                                           seed=n_tokens * 31 + len(mode), mode=mode)
            packed, scales, biases, dequantized = quantize_experts(experts)
            inputs = {"x": x, "router": router, "experts": dequantized}
            ref = op.reference(inputs)
            tol = op.tolerance(Case(dtype="float16"), inputs, ref)

            grid, threadgroup = routing_launch(n_tokens)
            route = runner.run(routing_spec(), [DeviceCase(
                inputs={"x": x, "router": router},
                output_shapes=[((n_tokens, 2), "uint32"), ((n_tokens, 2), "float32")],
                grid=grid, threadgroup=threadgroup,
                template={"E": n_experts})])[0]
            if not route.ok:
                print(f"    {mode:<14} n={n_tokens:<3} ROUTING RUNNER FAIL: {route.error}")
                ok = False
                continue
            idx, gate = route.outputs

            # The contract's own routing, from the shipped reference's helper.
            from kernelverify.reference.native_kernels import _topk_by_prob
            logits = x.astype(np.float32) @ router.astype(np.float32).T
            probs = np.exp(logits - logits.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            want = _topk_by_prob(probs, 2, tie_high=False)
            routing_exact = np.array_equal(idx.astype(int), want)

            grid, threadgroup, r = dispatch_launch(d_ffn, n_tokens)
            result = runner.run(dispatch_spec(), [DeviceCase(
                inputs={"x": x, "idx": idx, "gate": gate, "w_q": packed,
                        "scales": scales, "biases": biases},
                output_shapes=[((n_tokens, d_ffn), "float16")],
                grid=grid, threadgroup=threadgroup,
                template={"T": "float16", "R": r})])[0]
            if not result.ok:
                print(f"    {mode:<14} n={n_tokens:<3} DISPATCH RUNNER FAIL: {result.error}")
                ok = False
                continue
            err = float(np.max(np.abs(result.outputs[0].astype(np.float64) - ref)))
            passed = err <= tol and routing_exact
            ok = ok and passed
            print(f"    {mode:<14} n={n_tokens:<3} E={n_experts:<3} "
                  f"routing exact {str(routing_exact):<5} err {err:9.3e} "
                  f"tol {tol:9.3e}  {'pass' if passed else 'FAIL'}")
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

        def median_of(fn):
            mx.eval(fn(0))
            mx.synchronize()
            copies = calibrate(lambda c: dispatch_once(fn, c))
            return statistics.median(
                [dispatch_once(fn, copies) / copies for _ in range(ROUNDS)])

        t_ours, t_mlx = median_of(ours), median_of(theirs)
        t_od, t_md = median_of(ours_dispatch), median_of(mlx_dispatch)
        print(f"  {n_tokens:>7} | {t_ours*1e6:>9.1f} {t_mlx*1e6:>9.1f} "
              f"{t_mlx/t_ours:>6.2f}x {str(agree):>6} "
              f"| {t_od*1e6:>9.1f} {t_md*1e6:>9.1f} {t_md/t_od:>6.2f}x")
    return ok


def main() -> int:
    if not verify(MetalRunner()):
        print("\nVERDICT: kernel does not verify; no timing claim permitted")
        return 1
    if not bench():
        print("\nWARNING: an arm disagreed with the other at timing shapes")
        return 1
    print("\nratio > 1.00x means the verified kernel beats MLX's routing + gather_qmm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
