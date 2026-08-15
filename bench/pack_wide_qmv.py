"""Verify, then time, the wide-tile quantized matvec against MLX's own.

Order matters and is enforced: no timing is printed unless every case
verifies through the crash-isolated runner against the shipped verdict,
via kernelverify.pack.verify (the battery's own NATIVE_OPS reference and
tolerance; today that is the r_contract anchor, ensemble floor, K_QUANT).

Timing discipline, from kv-runner-e9's measurement that GPU timings near
200 us move by up to 4x with power state:

- every sample is a batched dispatch sized to at least MIN_SAMPLE_MS, so
  nothing sub-millisecond is ever timed;
- ours and MLX are sampled interleaved within a round, so a power-state
  excursion hits both arms rather than whichever ran first;
- weights rotate over a working set larger than any cache, because a
  few-MB buffer reused across iterations reports cache bandwidth;
- both arms read the SAME artefact arrays, which is checkable because the
  canonical quantizer is bit-exact against mx.quantize.

The claim under test: MLX's qmv_wide re-reads the whole weight matrix once
per five input vectors, so at M = 6..11 it pays two or three passes where
one is enough.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from kernelverify.pack.dispatch_shapes import (  # noqa: E402
    PINNED_QWEN3_4B,
    qmv_dispatch_shapes,
)
from kernelverify.extraction.surface import LiveCall  # noqa: E402
from kernelverify.pack.evidence import (  # noqa: E402
    CaseEvidence,
    GateEvidence,
    output_fingerprint,
    render_banner,
)
from kernelverify.pack.verify import (  # noqa: E402
    judge,
    qmv_inputs,
    reference_and_tolerance,
)
from kernelverify.schemas.native_ops import K_QUANT  # noqa: E402
from kernelverify.pack.wide_qmv import (  # noqa: E402
    KERNEL_NAME,
    SUPPORTED_BITS,
    build,
    kernel_spec,
    launch_config,
    pack_codes,
    should_dispatch,
)
from kernelverify.runners import MetalRunner, RunCase, specialize  # noqa: E402
from kernelverify.schemas.quant_contract import (  # noqa: E402
    QuantContract,
    canonical_quantize,
)

SHAPES = [(2560, 2560), (4096, 4096)]
VERIFY_M = [1, 4, 5, 6, 8, 11]
TIMED_M = [1, 2, 4, 5, 6, 7, 8, 10, 11]
MIN_SAMPLE_MS = 5.0
ROUNDS = 7
WORKING_SET_MB = 512

# A round is only trustworthy if the reference arm reads the same each time.
# Twice now a run has reported a kernel "win" that was the machine's clock
# moving under it: MLX's own time for one fixed shape moved 2.9x inside a
# single interleaved round while nothing about MLX changed. Interleaving alone
# does not catch that, it only makes both arms suffer it together, so the
# reference arm's own spread is checked and the ratios are withheld when it is
# too wide to support them.
MAX_CANARY_SPREAD = 1.5

# The serving block's E2E coverage (ruling D1): the qmv shapes mlx-lm
# dispatches when decoding Qwen3-4B, verified at 3 bits only (D4 cut 2-bit
# from the block) and at exactly the M values the pack routes here.
E2E_BITS = 3

# The pinned artifact lives in the MAIN checkout (bench/.models is gitignored
# and never propagates to worktrees), so the path is absolute and canonical
# across lanes, like /Users/vlad/llama.cpp.
QWEN3_3BIT_ARTIFACT = Path("/Users/vlad/kernelverify/bench/.models/qwen3-4b-3bit-g64")


def e2e_config() -> tuple[dict, str]:
    """The artifact's own config when it has landed, else the pinned
    Qwen/Qwen3-4B architecture fields, with a provenance string either way."""
    config_path = QWEN3_3BIT_ARTIFACT / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text()), f"artifact {config_path}"
    return dict(PINNED_QWEN3_4B), ("pinned Qwen/Qwen3-4B config "
                                   "(artifact not landed yet)")


# Derived once, together, so the recorded provenance always names the config
# that actually produced the shapes in use.
E2E_CONFIG, E2E_PROVENANCE = e2e_config()
E2E_SHAPES = qmv_dispatch_shapes(E2E_CONFIG)


def e2e_verify_m() -> list[int]:
    """Exactly the tile widths should_dispatch routes to this kernel over the
    block's serving range B = 1..16; gate coverage follows the boundary."""
    return [m for m in range(1, 17) if should_dispatch(m)]


def weights_for(d_out: int, d_in: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.02).astype(np.float16)


def artefact_for(d_out: int, d_in: int, seed: int, bits: int = 4):
    w = weights_for(d_out, d_in, seed)
    return w, canonical_quantize(w, QuantContract(bits=bits, group_size=64))


def artefact_bytes(d_out: int, d_in: int, bits: int) -> float:
    """Bytes one quantized weight set actually occupies: packed codes plus
    fp16 scale and bias per 64-wide group."""
    return d_out * d_in * bits / 8 + 2 * (d_out * d_in / 64) * 2


def weight_sets(d_out: int, d_in: int, bits: int) -> list:
    """Quantized weight sets totalling at least WORKING_SET_MB, to rotate
    over during timing: a few-MB buffer reused across iterations reports
    cache bandwidth, not memory bandwidth."""
    set_bytes = artefact_bytes(d_out, d_in, bits)
    n_sets = max(2, min(64, int(WORKING_SET_MB * 1e6 // set_bytes) + 1))
    sets = [mx.quantize(mx.array(weights_for(d_out, d_in, seed=100 + seed)),
                        group_size=64, bits=bits)
            for seed in range(n_sets)]
    mx.eval([a for s in sets for a in s])
    return sets


def interleaved_samples(build_a, build_b, rounds: int = ROUNDS) -> tuple:
    """Per-round times of two arms, sampled interleaved within every round.

    Interleaving is load-bearing and NOT sufficient (AGENTS.md): it equalizes
    a clock excursion across the arms but cannot detect one, so the caller
    must still gate on the reference arm's own spread. Both arms are warmed
    first; the dispatch batch is calibrated on arm A to MIN_SAMPLE_MS.
    """
    mx.eval(build_a(0), build_b(0))
    mx.synchronize()
    copies = calibrate_copies(lambda c: dispatch(build_a, c))
    a_samples, b_samples = [], []
    for _ in range(rounds):
        a_samples.append(dispatch(build_a, copies) / copies)
        b_samples.append(dispatch(build_b, copies) / copies)
    return a_samples, b_samples


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
GATE_POLICY = (
    "pack-gate fixed-case sweep: 2 microbenchmark matrix shapes at every "
    "supported bit width, plus the six Qwen3-4B E2E decode dispatch shapes "
    "at 3 bits over the routed win-zone M values, x {unit, corpus} input "
    "scales per (BITS, M) specialization, judged against the shipped "
    "NATIVE_OPS['quantized_matmul'] reference and tolerance; artefact bytes "
    "checked identical to mx.quantize per (shape, bits). This is NOT the "
    "16-eval mutation-scored battery, which has not run against this operator."
)
SEED_PROTOCOL = (
    "weights: np.random.default_rng(7) per (shape, bits); inputs: "
    "np.random.default_rng(11) per (shape, bits), drawn M-major then "
    "{unit, corpus} scale, in bench/pack_wide_qmv.py verify()"
)


def verify(runner: MetalRunner, e2e_m: list | None = None) -> GateEvidence:
    """The correctness gate. `e2e_m` widens the E2E tile-width coverage past
    the routed zone; the pricing probe passes its full sweep so that nothing
    it times is unverified."""
    e2e_m = e2e_verify_m() if e2e_m is None else list(e2e_m)
    evidence = GateEvidence(gate="pack_wide_qmv", policy=GATE_POLICY,
                            seed_protocol=SEED_PROTOCOL)
    if E2E_SHAPES:
        evidence.checks.append(CaseEvidence(
            label=f"E2E dispatch shapes derived from {E2E_PROVENANCE}",
            passed=True))
    template = kernel_spec()
    coverage = [(d_out, d_in, bits, list(VERIFY_M), "")
                for (d_out, d_in) in SHAPES for bits in SUPPORTED_BITS]
    coverage += [(s.d_out, s.d_in, E2E_BITS, e2e_m, s.name)
                 for s in E2E_SHAPES]
    for d_out, d_in, bits, verify_m, site in coverage:
        site_tag = f" ({site})" if site else ""
        w, art = artefact_for(d_out, d_in, seed=7, bits=bits)
        packed = pack_codes(art.q, bits)

        # Both arms must read identical bytes for the timing to be a fair test.
        mx_wq, mx_sc, mx_bi = mx.quantize(mx.array(w), group_size=64, bits=bits)
        mx.eval(mx_wq, mx_sc, mx_bi)
        identical = (np.array_equal(packed, np.array(mx_wq))
                     and np.array_equal(art.scales, np.array(mx_sc))
                     and np.array_equal(art.biases, np.array(mx_bi)))
        evidence.checks.append(CaseEvidence(
            label=(f"{d_out}x{d_in}{site_tag} {bits}-bit artefact identical "
                   f"to mx.quantize"),
            passed=identical))

        # One spec per tile width M (M and R are compile-time constants in the
        # raw door), all sharing one worker session as one candidate.
        rng = np.random.default_rng(11)
        spec_batches, case_refs = [], []
        for m in verify_m:
            grid, threadgroup, r = launch_config(d_out, m)
            raw_template = {"T": "half", "BITS": bits, "M": m, "R": r}
            spec = specialize(template, raw_template)
            spec_ev = evidence.specialization(KERNEL_NAME, "quantized_matmul",
                                              raw_template, threadgroup)
            cases = []
            for tag, scale in (("unit", 1.0), ("corpus", 10.0)):
                x = (rng.standard_normal((m, d_in)).astype(np.float32)
                     * scale).astype(np.float16)
                ref, tol = reference_and_tolerance(
                    "quantized_matmul", qmv_inputs(x, w, bits))
                label = f"{bits}-bit M={m} {tag} {d_out}x{d_in}{site_tag}"
                cases.append(RunCase(
                    inputs={"x": x, "w_q": packed,
                            "scales": art.scales, "biases": art.biases},
                    params={"d_in_arg": d_in, "d_out_arg": d_out,
                            "row_blocks": grid[1]},
                    output_shapes=[((m, d_out), "float16")],
                    label=label))
                spec_ev.calls.append(LiveCall(
                    inputs={"x": x, "w_q": packed,
                            "scales": art.scales, "biases": art.biases},
                    output_shapes=[((m, d_out), "float16")],
                    grid=grid, threadgroup=threadgroup,
                    template=(("T", "float16"), ("BITS", bits),
                              ("M", m), ("R", r)),
                    label=label))
                case_refs.append((spec_ev, label, ref, tol))
            spec_batches.append((spec, cases))

        results = [r for batch in runner.run_candidate(spec_batches) for r in batch]
        for result, (spec_ev, label, ref, tol) in zip(results, case_refs):
            if not result.ok:
                spec_ev.cases.append(CaseEvidence(
                    label=label, passed=False, tol=tol,
                    detail=f"runner {result.status.value}: {result.detail}"))
                continue
            v = judge(result.outputs[0], ref, tol)
            spec_ev.cases.append(CaseEvidence(
                label=label, passed=v.ok, err=v.err, tol=v.tol,
                output_sha256=output_fingerprint(result.outputs[0])))

    render_banner(evidence, "correctness (runner-isolated, Phase 0 contract, "
                            f"K = {K_QUANT:g}):")
    return evidence


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def calibrate_copies(sample) -> int:
    """Fewest copies per dispatch that still reach MIN_SAMPLE_MS."""
    copies = 8
    while copies < 4096:
        if sample(copies) * 1e3 >= MIN_SAMPLE_MS:
            return copies
        copies *= 2
    return copies


def dispatch(build_one, copies: int) -> float:
    outs = [build_one(i) for i in range(copies)]
    t0 = time.perf_counter()
    mx.eval(outs)
    mx.synchronize()
    return time.perf_counter() - t0


def bench() -> bool:
    kernel = build(mx)
    any_unstable = False
    print(f"\ntiming: interleaved A/B, {ROUNDS} rounds, batched dispatches of "
          f">= {MIN_SAMPLE_MS} ms, weights rotated over {WORKING_SET_MB} MB")

    for (d_out, d_in), bits in [(s, b) for s in SHAPES for b in SUPPORTED_BITS]:
        weight_bytes = artefact_bytes(d_out, d_in, bits)
        sets = weight_sets(d_out, d_in, bits)

        print(f"\n  {d_out} x {d_in}, {bits}-bit group 64, "
              f"{len(sets)} weight sets")
        print(f"  {'M':>3} {'R':>3} {'ours us':>9} {'mlx us':>9} {'ratio':>7} "
              f"{'ours GB/s':>10} {'mlx passes':>11}")
        unstable = False
        for m in TIMED_M:
            x = mx.random.normal(shape=(m, d_in)).astype(mx.float16)
            mx.eval(x)
            grid, threadgroup, r = launch_config(d_out, m)

            def ours(i):
                wq, sc, bi = sets[i % len(sets)]
                return kernel(inputs=[x, wq, sc, bi],
                              output_shapes=[(m, d_out)], output_dtypes=[mx.float16],
                              grid=grid, threadgroup=threadgroup,
                              template=[("T", mx.float16), ("M", m), ("R", r),
                                        ("BITS", bits)])[0]

            def theirs(i):
                wq, sc, bi = sets[i % len(sets)]
                return mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                           group_size=64, bits=bits)

            a_samples, b_samples = interleaved_samples(ours, theirs)
            t_ours = statistics.median(a_samples)
            t_mlx = statistics.median(b_samples)
            spread = max(b_samples) / min(b_samples)
            if spread > MAX_CANARY_SPREAD:
                print(f"  {m:>3} {r:>3} {'':>9} {'':>9} {'REJECTED':>7} "
                      f"canary spread {spread:.2f}x > {MAX_CANARY_SPREAD}x")
                unstable = True
                continue
            passes = -(-m // 5) if m < 12 else 0
            print(f"  {m:>3} {r:>3} {t_ours*1e6:>9.1f} {t_mlx*1e6:>9.1f} "
                  f"{t_mlx/t_ours:>6.2f}x {weight_bytes/t_ours/1e9:>10.1f} "
                  f"{passes:>11}")
        any_unstable = any_unstable or unstable
    return not any_unstable


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
    stable = bench()
    print("\nratio > 1.00x means the verified kernel beats mx.quantized_matmul")
    if not stable:
        print("some rows were rejected: the reference arm's own time moved too "
              "much inside the round, so those ratios would describe the "
              "machine rather than the kernels. Re-run on an idle machine.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
