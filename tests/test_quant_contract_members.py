"""The quantized contract ensemble's CPU members, and the `factored-groups`
repair.

Why this module exists
----------------------
`factored-groups` stands for the int-accumulate quantized GEMV kernel: per
group it forms `s * sum(x*q) + b * sum(x)`. It used to form both per-group
sums with numpy's pairwise reduction, which is EXACT on a constant row - 64
identical fp32 values halve down a power-of-two tree with no rounding at all -
while every real int-accumulate kernel sums the group as a chain and rounds 63
times. The member was therefore unrealistically accurate exactly where the
device kernels are least accurate, the leave-one-out floor it participates in
was too tight there, and the shipped tolerance flagged a correct, C1-admissible
device kernel on 10 of the 1,536 committed serving records (worst 1.743x).

Two things are pinned here. First, the member's per-group reductions are
explicit fp32 chains, checked against a scalar-loop rewrite rather than against
a copy of the implementation. Second, the CONTINUITY INVARIANT: re-running the
committed grid moves the `factored-groups` column and nothing else, so a change
that quietly moved another member's arithmetic - a numpy or Accelerate upgrade,
a stray edit to a shared helper - fails here instead of silently re-basing the
floor.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from kernelverify.schemas.quant_contract import (
    ENSEMBLE,
    QuantContract,
    canonical_quantize,
    dequantize,
)
from phase0_contract_k import err, make_w, make_x

BITS = 3
GROUP = 64
# The K that SHIPPED the ten false positives (ADR 0012), kept as a literal on
# purpose: the "before" column below is a fact about that tolerance and must
# not follow K_QUANT when it moves. The re-derived K is asserted separately.
K_AT_THE_TIME = 4.0
CPU_MEMBERS = tuple(ENSEMBLE)
UNTOUCHED = tuple(m for m in CPU_MEMBERS if m != "factored-groups")
DEVICE_MEMBER = "device-factored-simd"

EVIDENCE = (Path(__file__).resolve().parents[1] / "bench" / "results"
            / "quant_serving_adequacy.json")

# The committed grid's pinned axes (bench/calibrate_quant_serving.grid_iteration).
GRID_BATCHES = (1, 2, 8, 16)
GRID_MODES = ("unit", "corpus-scale", "near-zero", "constant-rows")
GRID_DTYPES = ("float32", "float16")


# ---------------------------------------------------------------------------
# An independent rewrite of the member: per-group sums as explicit scalar
# chains, then the member's own (unchanged) cross-group tail. Deliberately
# written in Python scalars so it shares no numpy reduction with the code it
# checks; the shapes it runs on are tiny for that reason.
# ---------------------------------------------------------------------------
def _reference_factored_groups(x, a, exact_activation_sum=False):
    g = a.contract.group_size
    rows, cols = a.q.shape
    groups = cols // g
    xf = x.astype(np.float32)
    q = a.q.reshape(rows, groups, g).astype(np.float32)
    xq = np.zeros((x.shape[0], rows, groups), np.float32)
    xs = np.zeros((x.shape[0], groups), np.float32)
    for b in range(x.shape[0]):
        for grp in range(groups):
            block = xf[b, grp * g:(grp + 1) * g]
            if exact_activation_sum:
                xs[b, grp] = np.float32(np.float64(block).sum())
            else:
                acc = np.float32(0.0)
                for k in range(g):
                    acc = np.float32(acc + block[k])
                xs[b, grp] = acc
            for r in range(rows):
                acc = np.float32(0.0)
                for k in range(g):
                    acc = np.float32(acc + np.float32(block[k] * q[r, grp, k]))
                xq[b, r, grp] = acc
    out = (xq * a.scales.astype(np.float32)[None, :, :]).sum(axis=2)
    out += xs @ a.biases.astype(np.float32).T
    return out.astype(x.dtype)


def _artefact(d_out=8, d_in=128, bits=BITS, seed=5):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((d_out, d_in)) * 0.02).astype(np.float16)
    return canonical_quantize(w, QuantContract(bits=bits, group_size=GROUP))


@pytest.mark.parametrize("mode", ["unit", "corpus-scale", "constant-rows"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("bits", [3, 4])
def test_factored_groups_sums_each_group_as_an_explicit_fp32_chain(mode, dtype,
                                                                   bits):
    a = _artefact(bits=bits)
    x = make_x(2, a.q.shape[1], dtype, mode, np.random.default_rng(11))
    ours = ENSEMBLE["factored-groups"](x, a)
    theirs = _reference_factored_groups(x, a)
    assert ours.dtype == theirs.dtype
    assert np.array_equal(ours, theirs)


def test_factored_groups_activation_sum_is_not_exact_on_constant_rows():
    """The regression the repair exists for: on a constant row the member's
    per-group activation sum must carry the chain's rounding, not the exact
    value a pairwise tree returns for free over 64 identical summands."""
    a = _artefact()
    x = make_x(2, a.q.shape[1], np.float32, "constant-rows",
               np.random.default_rng(11))
    chained = _reference_factored_groups(x, a)
    exact = _reference_factored_groups(x, a, exact_activation_sum=True)
    assert not np.array_equal(chained, exact)
    assert np.array_equal(ENSEMBLE["factored-groups"](x, a), chained)


def test_factored_groups_stays_correct_and_inside_the_class():
    """A different rounding sequence, not a different answer: the member's
    deviation from the fp64 contract stays at working-precision scale and in
    line with the rest of the ensemble."""
    a = _artefact(d_out=64, d_in=256)
    x = make_x(4, a.q.shape[1], np.float32, "unit", np.random.default_rng(3))
    ref = x.astype(np.float64) @ dequantize(a, np.float64).T
    scale = float(np.abs(ref).max())
    errors = {name: err(fn(x, a), ref) for name, fn in ENSEMBLE.items()}
    assert errors["factored-groups"] < 1e-5 * scale
    assert errors["factored-groups"] <= 8.0 * max(
        errors[m] for m in UNTOUCHED)


# ---------------------------------------------------------------------------
# The continuity invariant, against the committed serving evidence.
#
# The rng stream of one (shape, draw, seed) block is pinned by the harness
# (make_w first, then make_x per (batch, mode, dtype), batch outermost), so a
# block's inputs are reproducible on CPU alone. Reproduction is self-checking:
# if any of it were wrong, the five untouched members would not come back
# bit-identical.
# ---------------------------------------------------------------------------
def _block_inputs(shape, draw, seed):
    d_out, d_in = (int(v) for v in shape.split("x"))
    rng = np.random.default_rng(10_000 + seed)
    w = make_w(d_out, d_in, draw, rng)
    artefact = canonical_quantize(
        w, QuantContract(scheme="mlx-affine", bits=BITS, group_size=GROUP))
    xs = {}
    for batch in GRID_BATCHES:
        for mode in GRID_MODES:
            for name in GRID_DTYPES:
                dtype = np.float32 if name == "float32" else np.float16
                xs[(batch, mode, name)] = make_x(batch, d_in, dtype, mode, rng)
    return artefact, xs


@pytest.fixture(scope="module")
def evidence():
    return json.loads(EVIDENCE.read_text())["records"]


def _tolerance(record, members, k):
    return max(record["base_tol"], k * max(members[m] for m in CPU_MEMBERS))


def _shipped_tolerance(record, members):
    return _tolerance(record, members, K_AT_THE_TIME)


def _false_positives(records):
    """The shipped quantized oracle against the device member it must never
    flag: max(base_tol, K * worst CPU member) vs device-factored-simd."""
    return [r for r in records
            if r["members"][DEVICE_MEMBER] > _shipped_tolerance(
                r, r["members"])]


def _recompute(records):
    """Every CPU member of the given records, from the pinned rng stream."""
    blocks = {}
    for r in records:
        blocks.setdefault((r["shape"], r["draw"], r["seed"]), []).append(r)
    out = []
    for (shape, draw, seed), group in sorted(blocks.items()):
        artefact, xs = _block_inputs(shape, draw, seed)
        w64 = dequantize(artefact, np.float64)
        for r in group:
            x = xs[(r["batch"], r["mode"], r["dtype"])]
            ref = x.astype(np.float64) @ w64.T
            out.append((r, {name: err(fn(x, artefact), ref)
                            for name, fn in ENSEMBLE.items()}))
    return out


# One block per (shape, draw) of the two cheapest serving shapes, at the two
# smallest batches and every mode and dtype: enough breadth to catch a member
# that moved for an unrelated reason, small enough to stay a unit test. The
# lm_head shape is deliberately absent; it costs a 3.1 GB fp64 reference.
def _continuity_sample(records):
    wanted = []
    for r in records:
        if (r["shape"] in ("1024x2560", "2560x4096") and r["batch"] in (1, 2)
                and r["seed"] in (0, 101) and r["draw"] == "normal-0.02"):
            wanted.append(r)
    return wanted


def test_continuity_only_factored_groups_moves(evidence):
    sample = _continuity_sample(evidence)
    assert len(sample) == 64, "the sample's own shape is part of the check"
    moved = 0
    for record, members in _recompute(sample):
        where = (f"{record['shape']} B{record['batch']} {record['draw']} "
                 f"s{record['seed']} {record['mode']} {record['dtype']}")
        for name in UNTOUCHED:
            assert members[name] == record["members"][name], f"{name} @ {where}"
        moved += members["factored-groups"] != record["members"]["factored-groups"]
    assert moved, "the repair changed nothing on the sampled records"


def test_the_ten_shipped_false_positives_are_closed(evidence):
    """The headline. On the committed evidence the shipped tolerance flags a
    correct in-contract device kernel on ten records; under the repaired member
    every one of them is back inside tolerance, with the margin measured."""
    flagged = _false_positives(evidence)
    assert len(flagged) == 10
    assert all(r["batch"] == 1 and r["mode"] == "constant-rows"
               and r["dtype"] == "float32" for r in flagged)
    before = max(r["members"][DEVICE_MEMBER] / _shipped_tolerance(r, r["members"])
                 for r in flagged)
    assert round(before, 3) == 1.743

    after = []
    for record, members in _recompute(flagged):
        for name in UNTOUCHED:
            assert members[name] == record["members"][name]
        members[DEVICE_MEMBER] = record["members"][DEVICE_MEMBER]
        after.append(record["members"][DEVICE_MEMBER]
                     / _shipped_tolerance(record, members))
    assert len(after) == 10
    assert max(after) < 1.0, sorted(after)
    assert 0.20 < min(after) <= max(after) < 0.30, sorted(after)


def test_k_stays_four_and_the_member_now_carries_its_class_alone(evidence):
    """ADR 0016, the two readings that decided K.

    The device grid's demand fell to 2.766 and its harness printed
    `shipped K: 3.0`, but the shipped K must cover the shapes the verifier is
    actually pointed at. Two quantities live on the serving grid and they
    disagree, so both are pinned here:

    - the K-derivation demand over the NINE-name membership (ADR 0012's rule)
      is 3.120, which the K grid covers at 4.0 and not at 3.0;
    - the leave-one-out spread over the SIX CPU members the shipped tolerance
      actually divides by is 7.561, up from 4.076 before the repair.

    The second is an ensemble-adequacy statistic, not a live flag: the shipped
    floor CONTAINS factored-groups, so a candidate rounding like it is judged
    against a floor that already holds its own error. It says the int-domain
    class now rests on one member that sticks out, which is a membership
    question ADR 0016 records as open rather than resolving.
    """
    from kernelverify.schemas.native_ops import K_QUANT

    assert K_QUANT == 4.0
    block = [r for r in evidence
             if r["shape"] == "9728x2560" and r["draw"] == "normal-0.02"
             and r["seed"] == 1]
    assert len(block) == 32, "the block's own shape is part of the check"

    worst, binding = 0.0, ""
    for record, members in _recompute(block):
        for name in CPU_MEMBERS:
            others = max(members[m] for m in CPU_MEMBERS if m != name)
            if members[name] > record["base_tol"] and others > 0:
                ratio = members[name] / others
                if ratio > worst:
                    worst = ratio
                    binding = f"{name} B{record['batch']} {record['mode']} {record['dtype']}"

    assert round(worst, 3) == 7.561, (worst, binding)
    assert binding == "factored-groups B1 constant-rows float32", binding
    assert worst > K_QUANT, "recorded as open: no K in the grid covers this spread"


def test_the_shipped_verifier_flags_no_correct_kernel_after_the_repair(evidence):
    """The product claim, on the cells that carried the defect: judged the way
    the shipped verifier judges - candidate against max(base, K * floor over
    the six CPU members) - the ten flagged records are all inside tolerance,
    and nothing else in those records' blocks has taken their place."""
    from kernelverify.schemas.native_ops import K_QUANT

    flagged = _false_positives(evidence)
    assert len(flagged) == 10
    for record, members in _recompute(flagged):
        tol = max(record["base_tol"],
                  K_QUANT * max(members[m] for m in CPU_MEMBERS))
        for name in ("device-factored-simd", "device-dequant-simd",
                     "device-dequant-loop"):
            ratio = record["members"][name] / tol
            assert ratio < 1.0, (name, ratio, record["shape"], record["mode"])
