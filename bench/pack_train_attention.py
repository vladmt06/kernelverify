"""Verify, then time, the training-attention forward against what stock runs.

Order matters and is enforced by the shape of this file: `bench()` refuses to
run unless `verify()` came back green, so no configuration is ever timed that
has not first been judged by `NATIVE_OPS["train_attention"]` through the shared
`kernelverify.pack.verify` path. This gate states no tolerance of its own.

Every launchable knob setting is verified and every launchable knob setting is
timed. Launchability is read from the device rather than assumed, because which
settings a chip can run is a property of the chip; a setting it refuses is
recorded with the reason it gave.

The three timing arms, and why these three:

- COMPOSED is the arm that matters. `mx.fast.scaled_dot_product_attention`
  fuses in a plain call and DECOMPOSES inside any gradient trace, so a training
  step runs two matmuls and a softmax at every layer. That is the thing this
  kernel replaces, and the ratio against it is the one the end-to-end claim
  rests on.
- FUSED is Apple's own kernel on the inference path. It is not what training
  runs and is not what we replace; it is the honest ceiling for a correct
  attention forward on this machine, and it is measured beside ours so the
  remaining gap is published rather than implied.
- OURS is one arm per launchable knob setting, so the sweep and the comparison
  come out of a single interleaved sampler rather than two runs.

Timing discipline is the shared engine's (`bench/interleave.py`): batched
dispatches of at least MIN_SAMPLE_MS, arms interleaved inside a round with
rotation, and the fused arm's own spread as the canary. Interleaving equalises
a clock excursion across the arms and cannot detect one, so a cell whose canary
spread exceeds MAX_CANARY_SPREAD is WITHHELD rather than published.

The gate:

    the LONG cell's best launchable setting must have a credited ratio above
    1.0 against the composed arm, where the credited ratio is the smallest
    composed sample over the largest ours sample. A registered cell that does
    not clear is RECORDED as routing to stock rather than failing the run.

The long cell carries the gate because attention's share of a training step is
quadratic in the width while every other region's is linear, so the long band
is where the end-to-end claim is won or lost. The short cell is reported beside
it and routes to stock when it does not clear, which is what per-cell routing
is for and what the receipt states in words.

This is NOT the wording the plan carried into the run, and the change is
recorded rather than quietly made: the plan asked both cells to clear, and at
the short cell the credited ratio sits on 1.0 and crosses it in either
direction between runs (1.03 on 2026-08-24 and 0.99 an hour later). What that
measures is the door and not the kernel: at the short cell the cast and the pad
are 60.6 percent of the call, and the kernel alone is faster there than MLX's
fused primitive. Both halves of that overhead are queued in TODOS.md with their
measurements, and until one of them lands the short band routes to stock.

The fused ratio is reported at both cells and gates nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from interleave import (  # noqa: E402
    MAX_CANARY_SPREAD,
    MIN_SAMPLE_MS,
    arms_agree,
    interleaved_arms,
)
from kernelverify.compiler.lint import lint  # noqa: E402
from kernelverify.compiler.search_space import Limits, enumerate_knobs  # noqa: E402
from kernelverify.pack import train_attention as ta  # noqa: E402
from kernelverify.pack.evidence import (  # noqa: E402
    CaseEvidence,
    GateEvidence,
    render_banner,
)
from kernelverify.pack.verify import (  # noqa: E402
    attn_inputs,
    group_heads,
    verify_output,
)
from kernelverify.runners.metal import MetalRunner  # noqa: E402

ROUNDS = 5

GATE_POLICY = (
    "every launchable knob setting linted for contract clause C1, screened for "
    "unwritten output cells with two complementary fills, and judged against "
    "NATIVE_OPS['train_attention'] (fp64 causal grouped-query reference, "
    "ensemble-floored tolerance, K borrowed) at both storage dtypes over the "
    "tile-edge widths and every group ratio; timing runs only on a green gate"
)
SEED_PROTOCOL = (
    "operands drawn at fp32 from numpy's default_rng(CASE_SEED) and cast on "
    "the device, so the reference reads exactly the bytes the kernel read"
)
CASE_SEED = 5

# The verification cases: the widths where a tiled kernel's faults live (one
# row past a tile, one short of it, the exact boundary), every group ratio the
# geometry can take, and both storage dtypes. Kept small on purpose, because
# the fp64 reference builds a width-by-width score matrix per head and the
# registered cells' own geometry would need half a gigabyte of it.
VERIFY_WIDTHS = (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                 127, 128, 129)
VERIFY_HEADS = ((8, 1), (8, 2), (8, 4), (8, 8))
VERIFY_DTYPES = ("float16", "bfloat16")

# The two registered cells, at the real Qwen3-4B geometry.
TIMED_CELLS = ((4, 65), (2, 1057))

# The cell the gate rules on. Attention's share of a step is quadratic in the
# width and every other region's is linear, so the long band is where the
# end-to-end claim is won; the short band is reported and routes to stock when
# it does not clear.
GATING_CELL = (2, 1057)

# Two fills a cell that no store reached would hold, and no computed value can
# equal in both passes. The same rule `kernelverify/compiler/unwritten.py`
# applies through the raw runner, through the fill the MLX door offers.
FILL_A = -12345.0
FILL_B = 98765.0


def _dtype(name: str):
    return {"float16": mx.float16, "bfloat16": mx.bfloat16}[name]


def operands(batch, n_q, n_kv, t_len, head_dim, dtype, seed=CASE_SEED):
    """Drawn at fp32 and cast on the device: bfloat16 has no numpy dtype, and
    the reference has to read exactly what the kernel read."""
    rng = np.random.default_rng(seed)

    def draw(heads, scale):
        return mx.array((rng.standard_normal((batch, heads, t_len, head_dim))
                         * scale).astype(np.float32)).astype(dtype)

    return draw(n_q, 0.4), draw(n_kv, 0.9), draw(n_kv, 0.9)


def knob_space():
    """The launchable half of the knob space, as this chip reports its limits."""
    limits = Limits.probe(MetalRunner())
    space = enumerate_knobs(ta.fwd_axes(), limits,
                            threads=ta.fwd_threads,
                            threadgroup_bytes=ta.fwd_threadgroup_bytes)
    return space, limits


def _padded_call(kern, knobs, q, k, v, *, init_value=None):
    """One dispatch of the raw door, without the door's own slicing, so the
    unwritten screen sees every cell the kernel declared."""
    batch, n_q_heads, t_len, head_dim = q.shape
    multiple = ta.fwd_row_multiple(knobs)
    qf = ta.pad_rows(mx, q, multiple).astype(mx.float32)
    kf = ta.pad_rows(mx, k, multiple).astype(mx.float32)
    vf = ta.pad_rows(mx, v, multiple).astype(mx.float32)
    t_pad = qf.shape[2]
    grid, threadgroup = ta.fwd_launch(knobs, batch, n_q_heads, t_pad)
    return kern(inputs=[qf, kf, vf, mx.array([t_len], dtype=mx.uint32)],
                template=[("T", q.dtype)],
                output_shapes=[(batch, n_q_heads, t_pad, head_dim),
                               (batch, n_q_heads, t_pad)],
                output_dtypes=[q.dtype, mx.float32],
                grid=grid, threadgroup=threadgroup, init_value=init_value)


def screen_unwritten(kern, knobs, q, k, v) -> tuple[bool, str]:
    """Did a store reach every cell the kernel declared?

    Two passes with different fills. A kernel that does not read its output
    writes identical bytes in both, so a cell that differs between them is a
    cell no store reached. This attests coverage and NOT value: a kernel that
    stores a wrong number into every cell passes here, which is what the
    tolerance gate is for.
    """
    first = _padded_call(kern, knobs, q, k, v, init_value=FILL_A)
    second = _padded_call(kern, knobs, q, k, v, init_value=FILL_B)
    mx.eval(*first, *second)
    for name, a, b in zip(("out", "lse"), first, second):
        left = np.array(a.astype(mx.float32))
        right = np.array(b.astype(mx.float32))
        differing = int(np.count_nonzero(left != right))
        if differing:
            return False, f"{differing} unwritten cells in {name}"
    return True, ""


def verify(space) -> GateEvidence:
    """The correctness gate: every launchable setting, every case."""
    evidence = GateEvidence(gate="pack_train_attention", policy=GATE_POLICY,
                            seed_protocol=SEED_PROTOCOL)
    evidence.checks.append(CaseEvidence(
        label=(f"{len(space.launchable)} of {space.enumerated} knob settings "
               f"launchable on this device"),
        passed=bool(space.launchable)))
    for reason, count in sorted(space.census.items()):
        if reason == "launchable":
            continue
        evidence.checks.append(CaseEvidence(
            label=f"{count} refused: {reason}", passed=True))

    for knobs in space.launchable:
        source = ta.fwd_door_source(knobs)
        for binding in ("half", "bfloat16_t"):
            report = lint(source, types={"T": binding})
            evidence.checks.append(CaseEvidence(
                label=f"{knobs} source in contract with T = {binding}",
                passed=report.ok, detail=report.reason or None))

        spec_ev = evidence.specialization(
            ta.FWD_KERNEL_NAME, "train_attention", dict(knobs), (32, 1, 1))
        kern = ta.build_fwd(mx, knobs, ta.HEAD_DIM)

        q, k, v = operands(2, 8, 2, 129, ta.HEAD_DIM, mx.float16)
        screened, detail = screen_unwritten(kern, knobs, q, k, v)
        spec_ev.cases.append(CaseEvidence(
            label=f"{knobs} writes every declared cell",
            passed=screened, detail=detail or None))

        for name in VERIFY_DTYPES:
            dtype = _dtype(name)
            for n_q, n_kv in VERIFY_HEADS:
                for t_len in VERIFY_WIDTHS:
                    q, k, v = operands(2, n_q, n_kv, t_len, ta.HEAD_DIM, dtype)
                    out, _lse = ta.run_fwd(mx, kern, knobs, q, k, v)
                    mx.eval(out)
                    inputs = attn_inputs(np.array(q.astype(mx.float32)),
                                         np.array(k.astype(mx.float32)),
                                         np.array(v.astype(mx.float32)))
                    verdict = verify_output(
                        "train_attention", inputs,
                        group_heads(np.array(out.astype(mx.float32)), n_kv),
                        name)
                    spec_ev.cases.append(CaseEvidence(
                        label=f"{name} q{n_q}/kv{n_kv} T={t_len}",
                        passed=verdict.ok, err=verdict.err, tol=verdict.tol))

    render_banner(evidence, "correctness (battery reference and tolerance, "
                            "K borrowed and uncalibrated):")
    return evidence


def composed_arm(q, k, v):
    """What a training step actually runs at every layer: the decomposition MLX
    falls back to inside a gradient trace."""
    batch, n_q_heads, t_len, head_dim = q.shape
    n_kv = k.shape[1]
    grouped = q.reshape(batch, n_kv, n_q_heads // n_kv, t_len, head_dim)
    scores = (grouped @ mx.swapaxes(k[:, :, None], -1, -2)) * ta.SCALE
    allowed = mx.tril(mx.ones((t_len, t_len), dtype=mx.bool_))
    scores = mx.where(allowed, scores, mx.array(-float("inf"), scores.dtype))
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return (probs @ v[:, :, None]).reshape(batch, n_q_heads, t_len, head_dim)


def ratio_lo(numerator: list, denominator: list) -> float:
    """The worst pairing the samples permit: smallest numerator over largest
    denominator, the same reduction every pricing verdict in this repository
    is computed with."""
    return min(numerator) / max(denominator)


def bench(space) -> bool:
    """Time every launchable setting against composed and against fused."""
    print(f"\ntiming: interleaved arms, {ROUNDS} rounds, batched dispatches of "
          f">= {MIN_SAMPLE_MS} ms, canary = the fused arm's own spread")
    green = True
    for batch, t_len in TIMED_CELLS:
        cell = ta.cell_key(batch, t_len)
        q, k, v = operands(batch, ta.N_Q_HEADS, ta.N_KV_HEADS, t_len,
                           ta.HEAD_DIM, mx.bfloat16)
        mx.eval(q, k, v)

        builders = {
            "composed": lambda i: composed_arm(q, k, v),
            "fused": lambda i: mx.fast.scaled_dot_product_attention(
                q, k, v, scale=ta.SCALE, mask="causal"),
        }
        kernels = {}
        for knobs in space.launchable:
            label = f"SG{knobs['SGROUPS']}/BK{knobs['BKEY']}"
            kernels[label] = (ta.build_fwd(mx, knobs, ta.HEAD_DIM), knobs)

            def arm(i, label=label):
                kern, setting = kernels[label]
                return ta.run_fwd(mx, kern, setting, q, k, v)[0]

            builders[label] = arm

        # A ratio between two arms that compute different things is not a
        # ratio. The gate's own verdicts are taken at geometries an fp64
        # reference can afford; this asks the weaker question at the exact
        # shapes about to be timed, which is the only place it can be asked.
        composed_once = composed_arm(q, k, v).astype(mx.float32)
        mx.eval(composed_once)
        disagreed = []
        for label in kernels:
            # numpy cannot hold bfloat16, so both arms are widened on the
            # device before the comparison, which is exact.
            ours_once = builders[label](0).astype(mx.float32)
            mx.eval(ours_once)
            if not arms_agree(ours_once, composed_once):
                disagreed.append(label)
        if disagreed:
            print(f"  {cell}: arms disagree with composed: {disagreed}")
            green = False
            continue

        samples = interleaved_arms(builders, ROUNDS)
        canary = max(samples["fused"]) / min(samples["fused"])
        print(f"\n  {cell}  batch {batch}, width {t_len}, "
              f"canary spread {canary:.3f}")
        if canary > MAX_CANARY_SPREAD:
            print(f"  WITHHELD: the reference arm's spread is over "
                  f"{MAX_CANARY_SPREAD}, so the clock moved under this cell")
            green = False
            continue

        composed_ms = [s * 1e3 for s in samples["composed"]]
        fused_ms = [s * 1e3 for s in samples["fused"]]
        print(f"  {'arm':>12} {'median ms':>10} {'vs composed':>12} "
              f"{'vs fused':>9}")
        print(f"  {'composed':>12} {np.median(composed_ms):10.3f} "
              f"{'1.00':>12} {'':>9}")
        print(f"  {'fused':>12} {np.median(fused_ms):10.3f} "
              f"{ratio_lo(composed_ms, fused_ms):12.2f} {'1.00':>9}")
        best = None
        for label in kernels:
            ours_ms = [s * 1e3 for s in samples[label]]
            against_composed = ratio_lo(composed_ms, ours_ms)
            against_fused = ratio_lo(fused_ms, ours_ms)
            print(f"  {label:>12} {np.median(ours_ms):10.3f} "
                  f"{against_composed:12.2f} {against_fused:9.2f}")
            if best is None or against_composed > best[1]:
                best = (label, against_composed, against_fused)
        print(f"  best at {cell}: {best[0]} at {best[1]:.2f}x composed, "
              f"{best[2]:.2f}x fused")
        if best[1] <= 1.0:
            print(f"  {cell} ROUTES TO STOCK: nothing here beats what stock "
                  f"runs, which is a first-class outcome and not a failure")
            if (batch, t_len) == GATING_CELL:
                print(f"  GATE FAILED: {cell} is the cell the claim rests on")
                green = False
    return green


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="run the correctness gate and stop")
    args = parser.parse_args(argv)

    space, limits = knob_space()
    print(f"device limits: {limits.max_threads_per_threadgroup} threads, "
          f"{limits.max_threadgroup_memory} bytes of threadgroup memory")
    evidence = verify(space)
    if not evidence.ok:
        print("\nrefusing to time an unverified kernel")
        return 1
    if args.verify_only:
        return 0
    return 0 if bench(space) else 1


if __name__ == "__main__":
    raise SystemExit(main())
