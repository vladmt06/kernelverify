"""Verify, then time, the fused quantized-KV decode against MLX's own ops.

Same order and same discipline as the other two pack gates: nothing is timed
until every case verifies through the crash-isolated runner, samples are
batched to at least MIN_SAMPLE_MS, arms are interleaved within each round,
the reference arm's own spread is the clock canary, and caches rotate over a
working set larger than any GPU cache.

The incumbent arm is the mlx-lm composition of this operator (two
`mx.quantized_matmul` calls around a precise softmax, the arrangement in
mlx_lm.models.base.quantized_scaled_dot_product_attention), adapted to the
contract's semantics: the step's own K/V enter at full precision, and the
activations run at fp32 because the contract requires fp32 scores - the
stock fp16-score composition would fail the very canary this operator
carries. Both arms read identical artefact bytes (checked against
mx.quantize) and both arms are verified through the shipped oracle before
either is timed, so the ratio compares two contract-passing implementations.

The claim under test: one launch per decode step against roughly ten
dispatched ops, on the operator every layer of every decoded token pays.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from kernelverify.extraction.surface import LiveCall  # noqa: E402
from kernelverify.pack.evidence import (  # noqa: E402
    CaseEvidence,
    GateEvidence,
    output_fingerprint,
    render_banner,
)
from kernelverify.pack.kv_attention import (  # noqa: E402
    KERNEL_NAME,
    build,
    kernel_spec,
    launch_config,
    quantize_cache,
    require_capacity,
    should_dispatch,
)
from kernelverify.pack.verify import (  # noqa: E402
    judge,
    kv_inputs,
    reference_and_tolerance,
)
from kernelverify.runners import MetalRunner, RunCase, specialize  # noqa: E402

# (B, H, T, DH) x BITS: 3 shapes x 2 widths = the 6 runner-isolated cases.
VERIFY_SHAPES = [(1, 8, 512, 128), (4, 2, 64, 64), (1, 4, 300, 128)]
VERIFY_BITS = (4, 8)

TIMED_T = [64, 128, 256, 512, 1024]
TIMED_B = [1, 2, 4, 8]
TIMED_H, TIMED_DH = 8, 128
MIN_SAMPLE_MS = 5.0
ROUNDS = 7
WORKING_SET_MB = 512

# Interleaving equalises a clock excursion across the arms; it cannot detect
# one. The reference arm's own spread is the detector, and rows that fail it
# are withheld rather than published.
MAX_CANARY_SPREAD = 1.5


def make_case(b, h, t, dh, seed, dtype="float16"):
    """Inputs in the battery's own regime: q and new_k at peak 0.6 (the
    contract's QUERY_SCALE), caches at unit scale."""
    rng = np.random.default_rng(seed)

    def peaked(shape):
        a = rng.standard_normal(shape).astype(np.float32)
        return (a * (0.6 / float(np.max(np.abs(a))))).astype(dtype)

    q, new_k = peaked((b, h, dh)), peaked((b, h, dh))
    k_cache = rng.standard_normal((h, t, dh)).astype(dtype)
    v_cache = rng.standard_normal((h, t, dh)).astype(dtype)
    new_v = rng.standard_normal((b, h, dh)).astype(dtype)
    return q, k_cache, v_cache, new_k, new_v


def mlx_arm(q, new_k, new_v, kq, vq, bits):
    """The incumbent: mlx-lm's quantized SDPA arrangement at the contract's
    semantics. All inputs are mx arrays; q/new_* are (B, H, DH) fp16, the
    cache artefacts are (H, T, ...) as mx.quantize emits them."""
    dh = q.shape[-1]
    t = kq[0].shape[1]
    scale = 1.0 / float(np.sqrt(dh))
    qs = mx.transpose(q, (1, 0, 2)).astype(mx.float32) * scale     # (H,B,DH)
    scores = mx.quantized_matmul(qs, *kq, transpose=True,
                                 group_size=64, bits=bits)         # (H,B,T)
    nk = mx.transpose(new_k, (1, 0, 2)).astype(mx.float32)
    s_new = mx.sum(qs * nk, axis=-1, keepdims=True)                # (H,B,1)
    probs = mx.softmax(mx.concatenate([scores, s_new], axis=-1),
                       axis=-1, precise=True)
    out = mx.quantized_matmul(probs[..., :t], *vq, transpose=False,
                              group_size=64, bits=bits)            # (H,B,DH)
    out = out + probs[..., t:] * mx.transpose(new_v, (1, 0, 2)).astype(mx.float32)
    return mx.transpose(out, (1, 0, 2)).astype(mx.float16)


def quantized_arms_inputs(k_cache, v_cache, bits):
    """One artefact, both arms: the packed triplets our kernel reads, and the
    same bytes reshaped the way mx.quantized_matmul wants them."""
    k_wq, k_sc, k_bi = quantize_cache(k_cache, bits)
    v_wq, v_sc, v_bi = quantize_cache(v_cache, bits)
    ours = (k_wq, k_sc, k_bi, v_wq, v_sc, v_bi)
    theirs = ((mx.array(k_wq), mx.array(k_sc), mx.array(k_bi)),
              (mx.array(v_wq), mx.array(v_sc), mx.array(v_bi)))
    return ours, theirs


def artefacts_identical(cache, bits, packed, scales, biases) -> bool:
    h, t, dh = cache.shape
    wq, sc, bi = mx.quantize(mx.array(cache.reshape(h * t, dh)),
                             group_size=64, bits=bits)
    mx.eval(wq, sc, bi)
    return (np.array_equal(packed.reshape(h * t, -1), np.array(wq))
            and np.array_equal(scales.reshape(h * t, -1), np.array(sc))
            and np.array_equal(biases.reshape(h * t, -1), np.array(bi)))


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
GATE_POLICY = (
    "pack-gate fixed-case sweep: 3 fixed (B, H, T, DH) shapes x 2 bit widths "
    "(4, 8), judged against the shipped NATIVE_OPS['kv_attention'] reference "
    "and tolerance, with the mlx-ops incumbent arm judged by the same oracle "
    "and artefact bytes checked identical to mx.quantize per case. This is "
    "NOT the 16-eval mutation-scored battery, which has not run against this "
    "operator; 2/3-bit widths have unit-test coverage only, no gate evidence."
)
SEED_PROTOCOL = (
    "np.random.default_rng(t + bits) per (shape, bits) in make_case(), q and "
    "new_k rescaled to peak 0.6 (the contract's QUERY_SCALE), caches at unit "
    "scale, in bench/pack_kv_attention.py verify()"
)


def verify(runner: MetalRunner) -> GateEvidence:
    evidence = GateEvidence(gate="pack_kv_attention", policy=GATE_POLICY,
                            seed_protocol=SEED_PROTOCOL)
    template = kernel_spec()
    # One spec per distinct (BITS, DH) - the raw door's compile-time
    # constants - with cases grouped under it, all sharing one worker
    # session as one candidate.
    grouped: dict[tuple, tuple] = {}
    for b, h, t, dh in VERIFY_SHAPES:
        for bits in VERIFY_BITS:
            require_capacity(t)  # the capacity bound is correctness (TCAP)
            q, kc, vc, nk, nv = make_case(b, h, t, dh, seed=t + bits)
            (k_wq, k_sc, k_bi, v_wq, v_sc, v_bi), theirs_q = \
                quantized_arms_inputs(kc, vc, bits)
            label = f"B={b} H={h} T={t} DH={dh} {bits}-bit"
            evidence.checks.append(CaseEvidence(
                label=f"{label} artefact identical to mx.quantize",
                passed=artefacts_identical(kc, bits, k_wq, k_sc, k_bi)))
            ref, tol = reference_and_tolerance(
                "kv_attention", kv_inputs(q, kc, vc, nk, nv, bits))
            raw_template = {"T": "half", "BITS": bits, "DH": dh}
            if (bits, dh) not in grouped:
                grouped[(bits, dh)] = (specialize(template, raw_template), [], [])
            _, cases, exp = grouped[(bits, dh)]
            grid, threadgroup = launch_config(b, h)
            spec_ev = evidence.specialization(KERNEL_NAME, "kv_attention",
                                              raw_template, threadgroup)
            cases.append(RunCase(
                inputs={"q": q, "k_wq": k_wq, "k_scales": k_sc, "k_biases": k_bi,
                        "v_wq": v_wq, "v_scales": v_sc, "v_biases": v_bi,
                        "new_k": nk, "new_v": nv},
                params={"b_rows": b, "n_heads": h, "t_cached": t},
                output_shapes=[((b, h, dh), "float16")],
                label=label))
            spec_ev.calls.append(LiveCall(
                inputs={"q": q, "k_wq": k_wq, "k_scales": k_sc, "k_biases": k_bi,
                        "v_wq": v_wq, "v_scales": v_sc, "v_biases": v_bi,
                        "new_k": nk, "new_v": nv},
                output_shapes=[((b, h, dh), "float16")],
                grid=grid, threadgroup=threadgroup,
                template=(("T", "float16"), ("BITS", bits), ("DH", dh)),
                label=label))
            # The incumbent is judged by the same oracle, so the timing
            # compares two contract-passing implementations.
            incumbent = mlx_arm(mx.array(q), mx.array(nk), mx.array(nv),
                                *theirs_q, bits)
            mx.eval(incumbent)
            exp.append((spec_ev, label, ref, tol, np.array(incumbent)))

    spec_batches = [(spec, cases) for spec, cases, _ in grouped.values()]
    expected = [e for _, _, exp in grouped.values() for e in exp]
    results = [r for batch in runner.run_candidate(spec_batches) for r in batch]
    for result, (spec_ev, label, ref, tol, incumbent) in zip(results, expected):
        if not result.ok:
            spec_ev.cases.append(CaseEvidence(
                label=label, passed=False, tol=tol,
                detail=f"runner {result.status.value}: {result.detail}"))
            continue
        v_ours = judge(result.outputs[0], ref, tol)
        spec_ev.cases.append(CaseEvidence(
            label=label, passed=v_ours.ok, err=v_ours.err, tol=v_ours.tol,
            output_sha256=output_fingerprint(result.outputs[0])))
        # The incumbent arm is a gate fairness condition, not kernel evidence.
        v_mlx = judge(incumbent, ref, tol)
        evidence.checks.append(CaseEvidence(
            label=f"{label} mlx-ops incumbent arm passes the same oracle",
            passed=v_mlx.ok, err=v_mlx.err, tol=v_mlx.tol))

    render_banner(evidence,
                  "correctness (runner-isolated, shipped kv_attention contract):")
    return evidence


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def dispatch(build_one, copies: int) -> float:
    outs = [build_one(i) for i in range(copies)]
    t0 = time.perf_counter()
    mx.eval(outs)
    mx.synchronize()
    return time.perf_counter() - t0


def calibrate_copies(sample) -> int:
    copies = 8
    while copies < 4096 and sample(copies) * 1e3 < MIN_SAMPLE_MS:
        copies *= 2
    return copies


def time_row(kernel, b, h, t, dh, bits) -> tuple:
    """(ours_us, mlx_us, agree) for one shape, or a spread rejection."""
    assert should_dispatch(t)
    q, kc, vc, nk, nv = make_case(b, h, t, dh, seed=9 + t + b)

    cache_bytes = 2 * h * t * (dh * bits / 8 + (dh // 64) * 4)
    n_sets = max(2, min(10, int(WORKING_SET_MB * 1e6 // cache_bytes) + 1))
    sets = []
    for seed in range(n_sets):
        rng = np.random.default_rng(300 + seed)
        kci = rng.standard_normal((h, t, dh)).astype(np.float16)
        vci = rng.standard_normal((h, t, dh)).astype(np.float16)
        ours_np, theirs_mx = quantized_arms_inputs(kci, vci, bits)
        sets.append(([mx.array(a) for a in ours_np], theirs_mx))
    mx.eval([a for s in sets for a in s[0]])
    mx.eval([a for s in sets for pair in s[1] for a in pair])

    qx, nkx, nvx = mx.array(q), mx.array(nk), mx.array(nv)
    mx.eval(qx, nkx, nvx)
    grid, threadgroup = launch_config(b, h)

    def ours(i):
        arrs, _ = sets[i % n_sets]
        return kernel(inputs=[qx, *arrs, nkx, nvx],
                      output_shapes=[(b, h, dh)], output_dtypes=[mx.float16],
                      grid=grid, threadgroup=threadgroup,
                      template=[("T", mx.float16), ("BITS", bits), ("DH", dh)])[0]

    def theirs(i):
        _, (kq, vq) = sets[i % n_sets]
        return mlx_arm(qx, nkx, nvx, kq, vq, bits)

    a, m = np.array(ours(0)).astype(np.float64), np.array(theirs(0)).astype(np.float64)
    agree = float(np.max(np.abs(a - m))) <= 5e-3 * max(1.0, float(np.max(np.abs(m))))

    mx.synchronize()
    copies = calibrate_copies(lambda c: dispatch(ours, c))
    a_samples, b_samples = [], []
    for _ in range(ROUNDS):
        a_samples.append(dispatch(ours, copies) / copies)
        b_samples.append(dispatch(theirs, copies) / copies)
    spread = max(b_samples) / min(b_samples)
    if spread > MAX_CANARY_SPREAD:
        return None, spread, agree
    return (statistics.median(a_samples), statistics.median(b_samples)), spread, agree


def bench(kernel) -> bool:
    ok = True
    print(f"\ntiming: interleaved A/B, {ROUNDS} rounds, batched dispatches of "
          f">= {MIN_SAMPLE_MS} ms, caches rotated over {WORKING_SET_MB} MB")

    print(f"\n  T sweep at B=1, H={TIMED_H}, DH={TIMED_DH}")
    print(f"  {'bits':>5} {'T':>5} | {'ours us':>9} {'mlx us':>9} {'ratio':>7} {'agree':>6}")
    for bits in (8, 4):
        for t in TIMED_T:
            row, spread, agree = time_row(kernel, 1, TIMED_H, t, TIMED_DH, bits)
            ok = ok and agree
            if row is None:
                print(f"  {bits:>5} {t:>5} | {'REJECTED: canary spread':>27} "
                      f"{spread:.2f}x > {MAX_CANARY_SPREAD}x")
                ok = False
                continue
            t_ours, t_mlx = row
            print(f"  {bits:>5} {t:>5} | {t_ours*1e6:>9.1f} {t_mlx*1e6:>9.1f} "
                  f"{t_mlx/t_ours:>6.2f}x {str(agree):>6}")

    print(f"\n  B sweep at T=512, H={TIMED_H}, DH={TIMED_DH}, 8-bit")
    print(f"  {'B':>5} | {'ours us':>9} {'mlx us':>9} {'ratio':>7} {'agree':>6}")
    for b in TIMED_B:
        row, spread, agree = time_row(kernel, b, TIMED_H, 512, TIMED_DH, 8)
        ok = ok and agree
        if row is None:
            print(f"  {b:>5} | {'REJECTED: canary spread':>27} "
                  f"{spread:.2f}x > {MAX_CANARY_SPREAD}x")
            ok = False
            continue
        t_ours, t_mlx = row
        print(f"  {b:>5} | {t_ours*1e6:>9.1f} {t_mlx*1e6:>9.1f} "
              f"{t_mlx/t_ours:>6.2f}x {str(agree):>6}")
    return ok


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-only", action="store_true",
                        help="run the correctness gate and stop; no timing")
    args = parser.parse_args(argv)
    if not verify(MetalRunner()).ok:
        print("\nVERDICT: kernel does not verify; no timing claim permitted")
        return 1
    if args.verify_only:
        print("\nVERDICT: verified; timing skipped (--verify-only)")
        return 0
    if not bench(build(mx)):
        print("\nSome rows were withheld: either the arms disagreed, or the "
              "reference arm's own time moved too much inside the round. "
              "Re-run on an idle machine before quoting anything.")
        return 1
    print("\nratio > 1.00x means one fused launch beats the mlx-lm-shaped op "
          "composition at the same contract; directional pending binding run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
