"""ADR 0016's whole-grid readings, checked against a committed artifact.

Why this module exists
----------------------
ADR 0016 reports four numbers about the serving grid, and every one of them is
a reading of the 1,536 committed ADR 0013 records with ONE column recomputed -
`factored-groups`, the member the ADR repaired:

    held-out device member vs the shipped tolerance, worst   1.743  ->  0.328
    K-derivation demand over the nine-name membership        6.973  ->  3.120

The device grid's equivalent records were preserved as an artifact when its ADR
landed; the serving grid's were not, so those four numbers rested on a run
nobody could repeat without a 70-minute recomputation. `tests/test_quant_contract
_members.py` closes part of that hole by recomputing a 32-record block live, but
a block is not the grid, and the two numbers above are maxima OVER the grid: a
sample cannot pin them.

`bench/derive_repaired_member_column.py` writes the missing column to
`bench/results/quant_serving_repaired_member.json`, and this module reads it.
Everything here is then arithmetic over two committed files - no recomputation,
no rng, under a second.

The identity check is the load-bearing part
-------------------------------------------
A derived artifact is only worth its provenance. This module asserts the
COMMITTED file's header against the LIVE evidence hash and the LIVE code
identity, so an artifact that predates a change to the member, to the error
metric, or to the evaluators that produced it fails here. That is deliberately
strict: touching any of those modules obliges a regeneration (one command, in
the failure message), because an artifact that silently outlived its code is
exactly the failure this repo has already had once - a committed reinterpretation
file named code that had since moved twice, behind a check that compared the
file to itself.
"""

import json
from pathlib import Path

import pytest

from kernelverify.schemas.native_ops import K_QUANT

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "bench" / "results" / "quant_serving_adequacy.json"
ARTIFACT = ROOT / "bench" / "results" / "quant_serving_repaired_member.json"

REPAIRED = "factored-groups"
# The member the shipped tolerance must never flag: a correct, C1-admissible
# device kernel, held out of the floor it is judged against.
HELD_OUT = "device-factored-simd"
# The K that shipped the ten false positives, a literal on purpose: the "before"
# column is a fact about THAT tolerance and must not follow K_QUANT when it moves.
K_AT_THE_TIME = 4.0

REGENERATE = "  .venv/bin/python -u bench/derive_repaired_member_column.py"


@pytest.fixture(scope="module")
def evidence():
    return json.loads(EVIDENCE.read_text())["records"]


@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT.exists():
        pytest.fail(f"{ARTIFACT.name} is missing; regenerate it with\n{REGENERATE}")
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def repaired(evidence, artifact):
    """Every record's nine member errors, with the repaired column substituted."""
    rows = artifact["records"]
    assert len(rows) == len(evidence)
    out = []
    for record, row in zip(evidence, rows):
        shape, batch, draw, seed, mode, dtype, value = row
        identity = (record["shape"], record["batch"], record["draw"],
                    record["seed"], record["mode"], record["dtype"])
        assert (shape, batch, draw, seed, mode, dtype) == identity, (
            f"the artifact's rows are not the evidence's rows, in order: {row}")
        members = dict(record["members"])
        members[REPAIRED] = value
        out.append((record, members))
    return out


def _demand(record, members, names):
    """ADR 0012's K rule on one record: the worst ratio of any member's error
    to the floor of the OTHERS, over the admissible cases (a member at or under
    the dtype's base tolerance is not evidence about K)."""
    worst, binding = 0.0, ""
    for name in names:
        others = max(members[m] for m in names if m != name)
        if members[name] > record["base_tol"] and others > 0:
            ratio = members[name] / others
            if ratio > worst:
                worst = ratio
                binding = (f"{name} @ {record['shape']} B{record['batch']} "
                           f"{record['draw']} {record['mode']} {record['dtype']}")
    return worst, binding


def _shipped_tolerance(record, members, cpu_members):
    return max(record["base_tol"], K_AT_THE_TIME * max(members[m] for m in cpu_members))


def test_the_committed_artifact_names_the_evidence_and_code_it_came_from(artifact):
    """The staleness guard. The header is compared against the LIVE files, so
    an artifact that outlived the member, the error metric or the evaluators
    that produced it fails here rather than being read as current."""
    import hashlib

    # Imported from the generator, never restated: one source for what identity
    # this artifact claims. (`bench/` is on the path via tests/conftest.py.)
    from derive_repaired_member_column import EVIDENCE_SHA256, code_identity

    header = artifact["derived_from"]
    live_evidence = hashlib.sha256(EVIDENCE.read_bytes()).hexdigest()
    assert live_evidence == EVIDENCE_SHA256, (
        "the evidence file itself has changed; this column is a reading OF it")
    assert header["evidence_sha256"] == EVIDENCE_SHA256, (
        f"the committed artifact was derived from different evidence; regenerate:\n{REGENERATE}")
    assert header["code_sha256"] == code_identity(), (
        f"the committed artifact predates the code that produces it; regenerate:\n{REGENERATE}")
    assert header["member"] == REPAIRED
    assert header["adr"] == "ADR 0016"


def test_the_artifact_says_exactly_which_members_it_verified(artifact):
    """What the continuity invariant does and does NOT cover.

    The generator re-derives the five untouched CPU members on every record and
    raises if one moved, so the invariant is enforced at derive time and cannot
    be re-checked here without repeating a 70-minute run. This test therefore
    does NOT assert that the invariant held - it asserts that the artifact
    states its own scope honestly, by naming the members rather than describing
    them. The three device members need a GPU and are carried through from the
    evidence unverified; an artifact that claimed otherwise would be a hollow
    assurance of exactly the kind this file exists to replace, and an earlier
    draft of this test made precisely that mistake by grepping the artifact's
    own prose for the word 'bit-identical'.
    """
    from kernelverify.schemas.quant_contract import ENSEMBLE

    header = artifact["derived_from"]
    assert header["cpu_members_rederived"] == sorted(m for m in ENSEMBLE if m != REPAIRED)
    assert REPAIRED not in header["cpu_members_rederived"]
    assert header["device_members_carried_through_unverified"] == [
        "device-dequant-loop", "device-dequant-simd", "device-factored-simd"]
    assert set(header["cpu_members_rederived"]).isdisjoint(
        header["device_members_carried_through_unverified"])


def test_the_artifact_moves_the_repaired_column(evidence, artifact):
    moved = sum(row[6] != record["members"][REPAIRED]
                for record, row in zip(evidence, artifact["records"]))
    assert moved, "the repair changed no record: the artifact cannot be of the repair"


def test_the_held_out_device_member_worst_ratio_falls_to_0_328(evidence, repaired):
    """ADR 0016's headline, over the whole grid rather than the ten flagged
    records: judged the way the shipped verifier judges - candidate against
    max(base_tol, K * floor over the six CPU members it divides by) - the
    held-out device member's worst reading falls from a flag to a third of the
    tolerance."""
    from kernelverify.schemas.quant_contract import ENSEMBLE

    cpu = tuple(ENSEMBLE)
    assert len(cpu) == 6, cpu

    before = max(r["members"][HELD_OUT] / _shipped_tolerance(r, r["members"], cpu)
                 for r in evidence)
    after = max(members[HELD_OUT] / _shipped_tolerance(record, members, cpu)
                for record, members in repaired)

    assert round(before, 3) == 1.743, before
    assert round(after, 3) == 0.328, after
    assert before > 1.0 >= after, "before: a false positive; after: inside tolerance"


def test_the_k_derivation_demand_falls_to_3_120_and_the_grid_covers_it_at_four(repaired, evidence):
    """Why K stayed 4.0 (ADR 0016).

    The device harness re-ran after the repair and its own grid demanded 2.766,
    which the K grid covers at 3.0 - and its progress line printed exactly that.
    But the shipped K must cover the shapes the verifier is pointed at, and on
    the SERVING shapes the same rule demands 3.120, which 3.0 does not cover.
    The grid's next value is 4.0, so K did not move. This test is the reason
    that reading is not a claim: it is recomputed here from the committed
    records every run.
    """
    names = tuple(evidence[0]["members"])
    assert len(names) == 9, names

    before, before_where = max(
        (_demand(r, r["members"], names) for r in evidence), key=lambda t: t[0])
    after, after_where = max(
        (_demand(record, members, names) for record, members in repaired),
        key=lambda t: t[0])

    assert round(before, 3) == 6.973, (before, before_where)
    assert round(after, 3) == 3.120, (after, after_where)
    # Which member binds is the repair's whole story: before, a correct device
    # kernel stuck out against a floor that was too tight; after, the member
    # that stands for its arithmetic class does.
    assert before_where == (f"{HELD_OUT} @ 1024x2560 B1 normal-0.02 "
                            "constant-rows float32"), before_where
    assert after_where == (f"{REPAIRED} @ 2560x9728 B2 normal-0.02 "
                           "constant-rows float32"), after_where


def test_adr_0014s_four_residual_cells_collapse_to_about_one(evidence, repaired):
    """ADR 0014's open residual, closed - the per-shape table of ADR 0016.

    ADR 0014 recorded four shapes whose admissible demand still exceeded the
    shipped K = 4, every one bound by `device-factored-simd` on constant-rows
    float32 at batch 1. That was read at the time as a K or membership problem.
    It was neither: the floor those cells produced was too tight because
    `factored-groups` was exact on constant rows. Under the repair all four sit
    near 1, which is what a member that is one legitimate rounding order among
    several should read.

    The other two shapes are pinned too, though ADR 0014 never named them: they
    sat at 3.405 and 2.972, under K and therefore unremarked, and pinning them
    is what makes "four cells" a measured count rather than a chosen one.

    WHAT THE ADR's "under the repair" COLUMN IS, exactly: the SAME cell re-read,
    not a fresh maximum over the shape. That distinction is load-bearing and the
    ADR's prose does not make it, so both readings are pinned here. Re-maximising
    over the shape after the repair gives higher numbers - 1.942 at 1024x2560,
    where `factored-groups` at batch 16 now binds - because the repair made that
    member LESS accurate by design, so it rises as the device member falls.
    Every one of those maxima is still far under K, which is the claim that
    matters; "roughly 1" is true of the four residual cells and not of the grid.
    """
    names = tuple(evidence[0]["members"])

    # Each shape's binding (record, member) BEFORE the repair.
    binding = {}
    for i, r in enumerate(evidence):
        for name in names:
            others = max(r["members"][m] for m in names if m != name)
            if r["members"][name] > r["base_tol"] and others > 0:
                ratio = r["members"][name] / others
                if ratio > binding.get(r["shape"], (0.0,))[0]:
                    binding[r["shape"]] = (ratio, i, name)

    before = {"1024x2560": 6.973, "9728x2560": 4.947, "4096x2560": 4.563,
              "2560x4096": 4.001, "2560x9728": 3.405, "151936x2560": 2.972}
    assert {s: round(v[0], 3) for s, v in binding.items()} == before
    assert sum(v[0] > K_QUANT for v in binding.values()) == 4, (
        "ADR 0014 named four cells above K; the count is measured, not chosen")

    # ADR 0016's column: those same cells, re-read under the repaired member.
    reread = {"1024x2560": 1.022, "9728x2560": 1.052,
              "4096x2560": 0.941, "2560x4096": 0.915}
    for shape, expected in reread.items():
        _, i, name = binding[shape]
        record, members = repaired[i]
        assert record["mode"] == "constant-rows" and record["batch"] == 1
        assert name == HELD_OUT, "all four were bound by the held-out device member"
        others = max(members[m] for m in names if m != name)
        assert round(members[name] / others, 3) == expected, (shape, members[name] / others)

    # The stronger reading the ADR does not report: re-maximised per shape.
    maxima = {}
    for record, members in repaired:
        d, _ = _demand(record, members, names)
        maxima[record["shape"]] = max(maxima.get(record["shape"], 0.0), d)
    assert {s: round(v, 3) for s, v in maxima.items()} == {
        "1024x2560": 1.942, "151936x2560": 1.804, "2560x4096": 1.464,
        "2560x9728": 3.120, "4096x2560": 2.220, "9728x2560": 2.328}
    assert max(maxima.values()) < K_QUANT, maxima


def test_the_heldout_half_of_k_demand_is_the_known_fp16_artifact_not_a_demand(evidence):
    """Why the two numbers above use the MEMBER half of `k_demand` only.

    The shipped estimator (`calibrate_quant_device.k_demand`) has a second half:
    each held-out implementation against the floor of the FULL membership. Run
    over the serving records that half reads 195, and it is not a K demand - it
    is ADR 0014's fp16 floor collapse. At fp16 activations every member rounds
    its output to the storage dtype, so the floor becomes the output's own
    rounding, the shipped tolerance's floor term goes inert (base_tol wins in
    all 768 fp16 records) and the ratio explodes while the actual tolerance
    overshoot is about 12x.

    This is pinned rather than merely commented because the trap is live: anyone
    recomputing "the K demand" over these records with the shipped estimator
    gets 195, and without this test would read the 3.120 above as wrong.
    """
    names = tuple(evidence[0]["members"])
    heldout, where = 0.0, ""
    for r in evidence:
        full = max(r["members"][m] for m in names)
        for name, value in r["heldout"].items():
            if value > r["base_tol"] and full > 0 and value / full > heldout:
                heldout, where = value / full, f"{name} {r['dtype']}"

    assert round(heldout, 3) == 195.198, (heldout, where)
    assert where.endswith("float16"), where
    fp16 = [r for r in evidence if r["dtype"] == "float16"]
    assert len(fp16) == 768
    assert all(r["base_tol"] >= K_AT_THE_TIME * max(r["members"][m] for m in names)
               for r in fp16), "the floor term is inert at fp16: base_tol wins everywhere"


def test_three_point_one_two_zero_is_what_holds_k_at_four(repaired, evidence):
    """The device grid alone would have licensed K = 3.0; the serving grid does
    not. Pinning the cover() lookup on both readings is what makes that a
    decision rather than a coincidence."""
    from calibrate_quant_bits import K_GRID

    names = tuple(evidence[0]["members"])
    demand = max(_demand(record, members, names)[0] for record, members in repaired)

    covering = [k for k in K_GRID if k >= demand]
    assert covering[0] == 4.0, (demand, K_GRID)
    assert 3.0 < demand <= 4.0, demand
    assert K_QUANT == covering[0], (
        "the shipped K is the smallest grid value covering the serving demand")
    # The device grid's own post-repair demand, for contrast: it licenses 3.0.
    assert min(k for k in K_GRID if k >= 2.766) == 3.0
