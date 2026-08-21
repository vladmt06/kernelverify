"""The Day 1 profile harness, everywhere it can be pinned without a device.

Almost all of it can be. The harness's job is to gather numbers and refuse
when they cannot mean what they would be read to mean, and both halves of that
are arithmetic over records. What genuinely needs the GPU - that a dial fires
inside a real step, that an arm's graph is the one its seams produced, that
the marked pass computes what the plain pass computes - lives in
tests/test_profile_stock_live.py.

The tests that matter most here are the refusals. A harness that gathered a
plausible wrong number and published it would look exactly like a harness that
worked, so each refusal is asserted together with the fact that nothing was
published.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import profile_instrument as pi  # noqa: E402
import profile_knobs as pk  # noqa: E402
import profile_rules as rules  # noqa: E402
import profile_stock as ps  # noqa: E402
from decode_rules import RunInvalid  # noqa: E402
from harness_runner import PreconditionFailed  # noqa: E402
from memory_guard import EXIT_NO_DEVICE, EXIT_PRECONDITION  # noqa: E402

DEPTH, ADAPTED = 36, 16

# One resolution floor, one stock step and one intercept, so every fake below
# is read against the same registered numbers rather than against whatever
# each test happened to pick.
R = 0.01
STOCK_MS = 100.0
INTERCEPT_MS = 60.0
SLOPES = {"L": 20.0, "Q": 15.0, "P3": 10.0, "A": 8.0}


def _context(cell="B", width="short", **over):
    fields = dict(batch=4, batch_width=rules.WIDTHS[width]["batch_width"],
                  model="qwen3-4b-4bit-g64", adapted=ADAPTED,
                  supervised=100, supervised_of=256, batch_sha256="batch-sha")
    fields.update(over)
    return ps.cell_context(cell, width, **fields)


def _arm_time(spec, *, slopes, residues, stock_ms, intercept_ms,
              scaffold_slopes):
    if spec.role == pk.STOCK:
        return stock_ms
    if spec.role == pk.KNOB:
        return intercept_ms + slopes[spec.candidate] * spec.nominal_phi
    if spec.role == pk.SCAFFOLD:
        return stock_ms + (scaffold_slopes.get(spec.candidate, 0.0)
                           * spec.nominal_phi)
    return intercept_ms - residues.get(spec.candidate, 0.0)


def _width(width="short", *, cell="B", slopes=None, residues=None,
           scaffold_slopes=None, stock_ms=STOCK_MS,
           intercept_ms=INTERCEPT_MS, rounds=ps.ROUNDS, jitter=None,
           context=None, actual=None, peak_gb=12.0):
    """One width's raw arms, in the shape the child records them.

    Every arm is a clean line by default, so a test that wants to fail one
    gate moves one number and nothing else, which is what makes a failure
    attributable to the gate the test names.
    """
    slopes = dict(SLOPES if slopes is None else slopes)
    residues = dict(residues or {})
    scaffold_slopes = dict(scaffold_slopes or {})
    jitter = dict(jitter or {})
    actual = dict(actual or {})
    context = _context(cell, width) if context is None else context
    arms = {}
    for spec in ps.arm_manifest(width):
        base = _arm_time(spec, slopes=slopes, residues=residues,
                         stock_ms=stock_ms, intercept_ms=intercept_ms,
                         scaffold_slopes=scaffold_slopes)
        samples = [base] * rounds
        if spec.label in jitter:
            samples = list(jitter[spec.label])
        arms[spec.label] = {
            "candidate": spec.candidate, "role": spec.role,
            "nominal_phi": spec.nominal_phi,
            "actual_phi": actual.get(spec.label, spec.nominal_phi),
            "sites": None, "context": context, "samples_ms": samples,
        }
    return {"width": width, "context": context, "arms": arms,
            "peak_gb": peak_gb}


def _structural(counts=None, *, identity_ok=True, foreign=()):
    observed = counts if counts is not None else {
        "attn-core": {pi.FORWARD: DEPTH, pi.BACKWARD: ADAPTED},
        "head-matmul": {pi.FORWARD: 1, pi.BACKWARD: 1},
        "cross-entropy": {pi.FORWARD: 1, pi.BACKWARD: 1},
        "qmm": {pi.FORWARD: 7 * DEPTH, pi.BACKWARD: 7 * ADAPTED - 3},
    }
    return {"gradients_compared": 56, "loss_equal": identity_ok,
            "gradients_differing": [] if identity_ok else ["layers.0.q"],
            "counts": observed, "foreign_on_removal": list(foreign),
            "completeness": ps.completeness(
                observed,
                ps.expected_counts(DEPTH, ADAPTED, sorted(pi.REGIONS)))}


def _cell(cell="B", *, widths=None, structural=None, **over):
    widths = ({name: _width(name, cell=cell) for name in ps.widths_for(cell)}
              if widths is None else widths)
    record = {
        "cell": cell, "depth": DEPTH, "adapted": ADAPTED, "widths": widths,
        "structural": ({name: _structural() for name in widths}
                       if structural is None else structural),
        "stock": {"certified": 0}, "rounds": ps.ROUNDS,
    }
    record.update(over)
    record["readings"] = {
        name: ps.width_reading(one, resolution_floor_ms=R)
        for name, one in sorted(widths.items())}
    return record


def _record(**over):
    record = {
        "closing_idle": {"idle": True},
        "cells": {"B": _cell("B")},
    }
    record.update(over)
    return record


def _plan(**over):
    plan = {
        "schema_version": 2,
        "bands": {name: f"bench/.data/{rules.WIDTHS[name]['data']}"
                  for name in rules.WIDTH_ORDER},
        "seed": 7, "optimizer": "adam", "optimizer_config": {},
        "learning_rate": 1e-5, "memory_ceiling_gb": 20.0,
        "wall_cap_seconds": 3600.0, "cells": ["B"],
        "resolution": {"floors_ms": {"short": R, "long": R},
                       "addendum_sha256": "addendum-sha"},
    }
    plan.update(over)
    return plan


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------
def test_amendment_eight_s_arm_counts_are_what_the_manifest_produces():
    """26 at the short width and 35 at the long, leaf by leaf.

    Amendment 8 corrected Amendment 7's 27 by finding candidate P3 missing
    from it, so the count is the check that the correction is what runs.
    """
    assert len(ps.arm_manifest("short")) == 26
    assert len(ps.arm_manifest("long")) == 35


def test_candidate_a_is_at_the_long_width_and_nowhere_else():
    short = {arm.candidate for arm in ps.arm_manifest("short")}
    long = {arm.candidate for arm in ps.arm_manifest("long")}
    assert "A" not in short
    assert "A" in long
    assert long - short == {"A"}


def test_a_rewrite_carries_an_ablation_and_a_retune_does_not():
    """Clause 18 credits a retune with its slope alone, and `KnobReading`
    refuses a retune that measured a residue nothing may consume, so the
    manifest must not build an arm the reducer will reject."""
    roles = {}
    for arm in ps.arm_manifest("long"):
        roles.setdefault(arm.candidate, set()).add(arm.role)
    assert pk.ABLATION in roles["L"] and pk.ABLATION in roles["A"]
    assert pk.ABLATION not in roles["Q"] and pk.ABLATION not in roles["P3"]
    for candidate in ("L", "Q", "P3", "A"):
        assert (pk.kind_of(candidate) == pk.REWRITE) == (
            pk.ABLATION in roles[candidate])


def test_every_candidate_places_four_knob_and_four_scaffold_arms():
    counted = {}
    for arm in ps.arm_manifest("long"):
        counted.setdefault((arm.candidate, arm.role), 0)
        counted[(arm.candidate, arm.role)] += 1
    for candidate in ("L", "Q", "P3", "A"):
        assert counted[(candidate, pk.KNOB)] == 4
        assert counted[(candidate, pk.SCAFFOLD)] == 4


def test_the_scaffold_is_placed_at_the_knob_s_own_settings():
    """A scaffold measured at different settings prices a different arm, and
    the reducer refuses that outright, so the manifest has to agree first."""
    for width in rules.WIDTH_ORDER:
        by_role = {}
        for arm in ps.arm_manifest(width):
            if arm.role in (pk.KNOB, pk.SCAFFOLD):
                by_role.setdefault((arm.candidate, arm.role), set()).add(
                    arm.nominal_phi)
        for candidate in ps.CANDIDATES_AT_WIDTH[width]:
            assert (by_role[(candidate, pk.KNOB)]
                    == by_role[(candidate, pk.SCAFFOLD)] == set(pk.PHIS))


def test_the_partition_regions_add_no_arms_of_their_own():
    """P1 reuses candidate L's dial and P2 reuses candidate A's; only P3 has
    a ladder, because only P3 measures something no candidate measures."""
    candidates = {arm.candidate for arm in ps.arm_manifest("long")}
    assert set(ps.PARTITION.values()) - candidates == set()
    assert "P1" not in candidates and "P2" not in candidates


def test_an_arm_label_is_never_repeated():
    for width in rules.WIDTH_ORDER:
        labels = [arm.label for arm in ps.arm_manifest(width)]
        assert len(set(labels)) == len(labels)


def test_an_unregistered_width_is_refused_rather_than_defaulted():
    with pytest.raises(RunInvalid, match="registered width"):
        ps.arm_manifest("medium")


def test_only_the_cell_that_decides_is_measured_at_both_widths():
    assert ps.widths_for(rules.PRIMARY_CELL) == rules.WIDTH_ORDER
    for cell in set(rules.CELLS) - {rules.PRIMARY_CELL}:
        assert ps.widths_for(cell) == ("short",)


def test_an_unregistered_cell_is_refused_rather_than_defaulted():
    with pytest.raises(RunInvalid, match="not a registered cell"):
        ps.widths_for("Z")


# ---------------------------------------------------------------------------
# The context
# ---------------------------------------------------------------------------
def test_the_context_carries_the_token_count_the_matmuls_actually_run_at():
    """`default_loss` trains on `batch[:, :-1]`, so a batch 65 tokens wide
    runs a 64-token step and a floor measured at 65 would be a floor for a
    shape the step never produces."""
    context = _context(width="short")
    assert context["batch_width"] == 65
    assert context["operation_width"] == 64
    assert context["tokens"] == 4 * 64


def test_a_batch_that_padded_to_another_width_refuses():
    with pytest.raises(RunInvalid, match="not the width the profile"):
        _context(width="short", batch_width=97)


def test_the_supervised_fraction_is_recorded_and_not_left_to_be_derived():
    """It IS candidate L's floor, so a floor measured against a different
    supervised count is a floor for a different problem."""
    context = _context(supervised=128, supervised_of=256)
    assert context["supervised_fraction"] == 0.5


def test_the_two_widths_differ_in_width_and_agree_about_the_model():
    short, long = _context(width="short"), _context(width="long")
    assert short["operation_width"] != long["operation_width"]
    for field in ("cell", "model", "adapted", "batch"):
        assert short[field] == long[field]


# ---------------------------------------------------------------------------
# The counts, which the boundary instrument still supplies
# ---------------------------------------------------------------------------
def test_the_backward_expectation_is_not_the_forward_one():
    counts = ps.expected_counts(DEPTH, ADAPTED, ["attn-core", "qmm"])
    assert counts["attn-core"] == {pi.FORWARD: DEPTH, pi.BACKWARD: ADAPTED}
    assert counts["qmm"] == {pi.FORWARD: 7 * DEPTH,
                             pi.BACKWARD: 7 * ADAPTED - 3}


def test_the_step_regions_fire_once_per_step_in_each_direction():
    counts = ps.expected_counts(DEPTH, ADAPTED,
                                ["head-matmul", "cross-entropy"])
    assert all(one == {pi.FORWARD: 1, pi.BACKWARD: 1}
               for one in counts.values())


@pytest.mark.parametrize("depth,adapted", [(0, 1), (36, 0), (36, 37), (-1, 1)])
def test_an_arrangement_mlx_lm_cannot_produce_is_refused(depth, adapted):
    with pytest.raises(RunInvalid):
        ps.expected_counts(depth, adapted, ["attn-core"])


def test_a_region_with_no_registered_count_is_refused_not_skipped():
    with pytest.raises(RunInvalid, match="no expected count"):
        ps.expected_counts(DEPTH, ADAPTED, ["attn-core", "invented"])


def test_a_seam_that_never_fired_is_caught_rather_than_read_as_zero():
    """Without this check the region is absent from the log entirely, its
    time lands in the remainder, and it looks exactly like an operation that
    costs nothing."""
    expected = ps.expected_counts(DEPTH, ADAPTED, ["attn-core", "qmm"])
    report = ps.completeness({"qmm": expected["qmm"]}, expected)
    assert report["ok"] is False
    assert {one["region"] for one in report["mismatches"]} == {"attn-core"}
    assert all(one["observed"] is None for one in report["mismatches"])


def test_a_count_that_is_merely_short_is_caught_too():
    expected = ps.expected_counts(DEPTH, ADAPTED, ["attn-core"])
    report = ps.completeness(
        {"attn-core": {pi.FORWARD: DEPTH, pi.BACKWARD: ADAPTED - 1}}, expected)
    assert report["ok"] is False
    assert report["mismatches"][0]["direction"] == pi.BACKWARD


def test_counts_that_match_the_arrangement_pass():
    expected = ps.expected_counts(DEPTH, ADAPTED, sorted(pi.REGIONS))
    assert ps.completeness(
        {region: dict(counts) for region, counts in expected.items()},
        expected)["ok"] is True


# ---------------------------------------------------------------------------
# The width reducer
# ---------------------------------------------------------------------------
def test_a_clean_ladder_produces_a_reading_for_every_candidate():
    reading = ps.width_reading(_width("long"), resolution_floor_ms=R)
    assert set(reading["entries"]) == set(ps.CANDIDATES_AT_WIDTH["long"])
    for candidate, entry in reading["entries"].items():
        assert entry["type"] == "reading", entry.get("blockers")
        assert entry["share"] == pytest.approx(SLOPES[candidate] / STOCK_MS)


def test_the_share_is_the_slope_over_stock_and_carries_the_residue():
    """Clause 18 credits a rewrite with its slope PLUS its residue, so the
    share and the partition's own reading of the same dial are two different
    numbers and the record has to show both."""
    reading = ps.width_reading(
        _width("short", residues={"L": 5.0}), resolution_floor_ms=R)
    entry = reading["entries"]["L"]
    assert entry["credited_residue_ms"] == pytest.approx(5.0)
    assert entry["share"] == pytest.approx((20.0 + 5.0) / STOCK_MS)
    assert entry["scaling_share"] == pytest.approx(20.0 / STOCK_MS)


def test_a_residue_below_the_floor_is_recorded_as_zero_and_kept_raw():
    reading = ps.width_reading(
        _width("short", residues={"L": R / 2}), resolution_floor_ms=R)
    entry = reading["entries"]["L"]
    assert entry["raw_residue_ms"] == pytest.approx(R / 2)
    assert entry["credited_residue_ms"] == 0.0
    assert entry["residue_is_a_fault"] is False


def test_a_residue_negative_by_more_than_the_floor_is_a_fault():
    """Clause 33: it says the step ran SLOWER with the operation removed than
    the fit predicts without its scaling part, which cannot be true, so no
    rule carries it and no demotion excuses it."""
    reading = ps.width_reading(
        _width("short", residues={"L": -5.0}), resolution_floor_ms=R)
    entry = reading["entries"]["L"]
    assert entry["type"] == "missing_share"
    assert entry["evidence"]["residue_is_a_fault"] is True


def test_a_candidate_whose_gates_fail_is_typed_absent_and_never_a_placeholder():
    """An absence is a measurement that legitimately does not exist and the
    rules carry it; it is not a zero, and it is not a null anything could
    divide by."""
    reading = ps.width_reading(
        _width("short", slopes=dict(SLOPES, Q=-15.0)), resolution_floor_ms=R)
    entry = reading["entries"]["Q"]
    assert entry["type"] == "missing_share"
    assert "share" not in entry
    assert any("not positive" in one for one in entry["blockers"])
    assert entry["evidence"]["registered_slope_ms"] < 0


def test_a_scaffold_that_does_not_move_is_clamped_and_its_slope_ignored():
    reading = ps.width_reading(_width("short"), resolution_floor_ms=R)
    entry = reading["entries"]["Q"]
    assert entry["scaffold_case"] == "clamped"
    assert entry["effective_scaffold_slope_ms"] == 0.0


def test_a_scaffold_that_moves_past_the_limit_blocks_its_candidate():
    """Clause 21's limit is on the scaffold's own SLOPE, not merely on its
    offset: a scaffold whose cost changes with the dial enters the fitted
    slope where an offset check could not see it."""
    reading = ps.width_reading(
        _width("short", scaffold_slopes={"Q": 5.0}), resolution_floor_ms=R)
    entry = reading["entries"]["Q"]
    assert entry["type"] == "missing_share"
    assert any("scaffold" in one for one in entry["blockers"])


def test_every_gate_records_the_margin_it_held_by():
    """A reader who can only see the verdict cannot tell a gate that barely
    held from one that held by a factor of ten."""
    reading = ps.width_reading(_width("short"), resolution_floor_ms=R)
    margins = reading["entries"]["L"]["margins_in_R"]
    assert margins["pooled_excursion"] > 0
    assert margins["median_excursion"] > 0
    assert margins["scaffold_offset"] > 0


def test_an_arm_the_manifest_does_not_name_is_a_hole_and_refuses():
    measured = _width("short")
    measured["arms"]["invented"] = measured["arms"]["stock"]
    with pytest.raises(RunInvalid, match="hole is not an absence"):
        ps.width_reading(measured, resolution_floor_ms=R)


def test_an_arm_the_recording_is_missing_refuses_rather_than_fits_around_it():
    measured = _width("short")
    del measured["arms"]["L:knob:0.50"]
    with pytest.raises(RunInvalid, match="missing"):
        ps.width_reading(measured, resolution_floor_ms=R)


def test_two_arms_in_one_fit_that_measured_different_workloads_refuse():
    measured = _width("short")
    measured["arms"]["Q:knob:0.50"] = dict(
        measured["arms"]["Q:knob:0.50"],
        context=_context(width="short", supervised=1))
    with pytest.raises(RunInvalid, match="different workloads"):
        ps.width_reading(measured, resolution_floor_ms=R)


def test_a_resolution_floor_of_zero_refuses_the_whole_width():
    """Clause 26 refuses it because a floor of zero claims the machine can
    resolve any difference at all, and every gate is a multiple of it."""
    with pytest.raises(RunInvalid, match="positive"):
        ps.width_reading(_width("short"), resolution_floor_ms=0.0)


def test_every_entry_says_whether_anything_certified_reads_it():
    reading = ps.width_reading(_width("long"), resolution_floor_ms=R)
    assert reading["entries"]["L"]["certified"] is True
    assert reading["entries"]["Q"]["certified"] is True
    assert reading["entries"]["A"]["certified"] is False
    assert reading["entries"]["P3"]["certified"] is False


def test_a_reported_candidate_carries_a_spread_labelled_as_not_a_bound():
    """Clause 34 gives a reported quantity an interval, and candidate A is in
    neither the pilot nor the fresh set, so there is no calibrated interval
    for it to carry and the label is what keeps the difference visible."""
    entry = ps.width_reading(_width("long"),
                             resolution_floor_ms=R)["entries"]["A"]
    assert set(entry["observed_spread_pct"]) == {
        arm.label for arm in ps.arm_manifest("long") if arm.candidate == "A"}
    assert "not an interval at any registered rate" in \
        entry["observed_spread_is_not_a_bound"]
    assert "observed_spread_pct" not in ps.width_reading(
        _width("long"), resolution_floor_ms=R)["entries"]["L"]


def test_arms_that_ran_a_different_number_of_times_refuse():
    """Arms whose rounds differ are not the repeats the spread gate reads, and
    which arm the reducer happened to count first must not decide it."""
    measured = _width("short")
    measured["arms"]["Q:knob:0.50"]["samples_ms"] = [60.0, 60.0]
    with pytest.raises(RunInvalid, match="rounds across its arms"):
        ps.width_reading(measured, resolution_floor_ms=R)


def test_every_arm_gets_its_own_spread_and_not_one_reference_arm_s():
    """With a ladder there is no reference arm: every arm is a whole step and
    any of them can be the one the machine moved under."""
    measured = _width("short", jitter={"Q:knob:0.50": [60, 200, 60, 60, 60]})
    reading = ps.width_reading(measured, resolution_floor_ms=R)
    assert set(reading["arms"]) == set(measured["arms"])
    assert reading["arms"]["Q:knob:0.50"]["rounds_eligible"] is False
    assert reading["arms"]["stock"]["rounds_eligible"] is True


# ---------------------------------------------------------------------------
# Clause 22's partition
# ---------------------------------------------------------------------------
def test_the_partition_reads_the_scaling_share_and_not_the_candidate_share():
    """`b/T`, with no residue term, because a partition region is a region and
    not a candidate. A large residue would otherwise carry a subtotal past one
    while every fit gate stayed clean."""
    reading = ps.width_reading(
        _width("long", residues={"L": 30.0}), resolution_floor_ms=R)
    assert reading["partition"]["measured"]["P1"] == pytest.approx(0.20)
    assert reading["entries"]["L"]["share"] == pytest.approx(0.50)


def test_all_three_regions_present_and_under_one_passes_with_a_remainder():
    reading = ps.width_reading(_width("long"), resolution_floor_ms=R)
    partition = reading["partition"]
    assert partition["outcome"] == "PASS"
    assert partition["subtotal"] == pytest.approx((20 + 8 + 10) / STOCK_MS)
    assert partition["remainder_P4"] == pytest.approx(1 - partition["subtotal"])


def test_a_width_without_candidate_a_leaves_the_test_not_run_and_not_passed():
    """P2 is measured by candidate A's dial, so the short width has no P2 by
    design; recording a pass there would convert an absence into evidence."""
    reading = ps.width_reading(_width("short"), resolution_floor_ms=R)
    assert reading["partition"]["outcome"] == "NOT RUN"
    assert "P2" in reading["partition"]["absent"]
    assert reading["partition"]["remainder_P4"] is None


def test_the_measured_regions_alone_above_one_reject_even_with_one_absent():
    """Every share is non-negative, so a missing addend can only raise the
    sum, and a subtotal already above one proves the full sum is above one
    whatever the missing region turns out to be."""
    reading = ps.width_reading(
        _width("short", slopes=dict(SLOPES, L=60.0, P3=50.0)),
        resolution_floor_ms=R)
    partition = reading["partition"]
    assert partition["outcome"] == "REJECT"
    assert partition["subtotal"] == pytest.approx(1.10)
    assert "P2" in partition["absent"]


def test_a_region_whose_own_reading_failed_is_absent_with_its_reason():
    reading = ps.width_reading(
        _width("long", slopes=dict(SLOPES, P3=-10.0)), resolution_floor_ms=R)
    assert reading["partition"]["outcome"] == "NOT RUN"
    assert reading["partition"]["absent"]["P3"] == "fit_gates_failed"


# ---------------------------------------------------------------------------
# The rule-facing matrix
# ---------------------------------------------------------------------------
def _readings(**over):
    widths = {name: _width(name) for name in rules.WIDTH_ORDER}
    widths.update(over)
    return {name: ps.width_reading(one, resolution_floor_ms=R)
            for name, one in widths.items()}


def test_the_matrix_is_every_candidate_at_every_width_in_registered_order():
    matrix = ps.profile_matrix(_readings())
    assert matrix["candidate_order"] == list(rules.CANDIDATES)
    assert matrix["width_order"] == list(rules.WIDTH_ORDER)
    assert set(matrix["entries"]) == set(rules.CANDIDATES)
    for column in matrix["entries"].values():
        assert set(column) == set(rules.WIDTH_ORDER)


def test_a_candidate_a_width_does_not_measure_is_typed_and_never_omitted():
    """Complete by construction: an omission and a null are both things a
    later rule could read as a zero."""
    entry = ps.profile_matrix(_readings())["entries"]["A"]["short"]
    assert entry["type"] == "missing_share"
    assert entry["reason_code"] == "not_measured_at_this_width"
    assert entry["certified"] is False


def test_candidate_a_absent_at_both_widths_is_still_a_complete_matrix():
    """REVERSED from Amendment 6, where a candidate A with no valid width
    reached INCOMPLETE. Under clause 37 it is never incompleteness."""
    matrix = ps.profile_matrix(_readings(
        long=_width("long", slopes=dict(SLOPES, A=-1.0))))
    assert matrix["entries"]["A"]["long"]["type"] == "missing_share"
    assert matrix["entries"]["L"]["long"]["type"] == "reading"


def test_a_width_the_deciding_cell_never_measured_refuses_the_matrix():
    with pytest.raises(RunInvalid, match="highest minimum gain"):
        ps.profile_matrix({"short": ps.width_reading(_width("short"),
                                                     resolution_floor_ms=R)})


def test_two_columns_that_measured_two_arrangements_refuse():
    """The comparison exists to isolate width, so a batch or an adapter count
    that moved between the columns would be reported as width."""
    readings = _readings(long=_width(
        "long", context=_context("B", "long", batch=8, adapted=ADAPTED)))
    with pytest.raises(RunInvalid, match="two arrangements"):
        ps.profile_matrix(readings)


def test_the_columns_are_allowed_to_differ_in_what_width_moves():
    matrix = ps.profile_matrix(_readings())
    short = matrix["context_by_width"]["short"]
    long = matrix["context_by_width"]["long"]
    assert short["operation_width"] != long["operation_width"]
    assert short["band"] != long["band"]


def test_a_context_field_nobody_classified_refuses_rather_than_passing():
    """A field added later has to be named invariant or variant, because the
    check is what stops a second thing moving with width unnoticed."""
    readings = _readings()
    readings["long"]["context"] = dict(readings["long"]["context"],
                                       invented=1)
    with pytest.raises(RunInvalid, match="neither"):
        ps.profile_matrix(readings)


def test_the_matrix_carries_readings_and_typed_absences_and_nothing_else():
    readings = _readings()
    readings["long"]["entries"]["Q"] = {"share": 0.5}
    with pytest.raises(RunInvalid, match="typed"):
        ps.profile_matrix(readings)


# ---------------------------------------------------------------------------
# The blockers, gate by gate against what they replaced
# ---------------------------------------------------------------------------
def test_a_clean_recording_has_no_blockers():
    assert ps.binding_blockers(_record()) == []


def test_a_machine_that_went_busy_at_the_closing_gate_blocks():
    blockers = ps.binding_blockers(_record(closing_idle={"idle": False}))
    assert any("not idle" in one for one in blockers)


def test_an_arm_whose_rounds_describe_the_machine_blocks():
    cell = _cell("B", widths={
        "short": _width("short", jitter={"stock": [100, 400, 100, 100, 100]}),
        "long": _width("long")})
    blockers = ps.binding_blockers(_record(cells={"B": cell}))
    assert any("arm stock" in one and "spread" in one for one in blockers)


def test_a_certified_candidate_that_could_not_be_read_blocks():
    cell = _cell("B", widths={
        "short": _width("short", slopes=dict(SLOPES, Q=-15.0)),
        "long": _width("long")})
    blockers = ps.binding_blockers(_record(cells={"B": cell}))
    assert any("candidate Q" in one for one in blockers)


@pytest.mark.parametrize("candidate", ["A", "P3"])
def test_a_reported_candidate_that_could_not_be_read_does_not_block(candidate):
    """Amendment 7 ruling 2 makes candidate A reported and never a gate, and
    candidate P3 is read only by a fault check that takes no margin, so their
    absence costs a report a row and a check a width."""
    slopes = dict(SLOPES)
    slopes[candidate] = -1.0
    cell = _cell("B", widths={"short": _width("short", slopes=slopes),
                              "long": _width("long", slopes=slopes)})
    blockers = ps.binding_blockers(_record(cells={"B": cell}))
    assert blockers == []
    entry = cell["readings"]["long"]["entries"][candidate]
    assert entry["type"] == "missing_share"


def test_a_residue_fault_blocks_even_for_a_candidate_that_gates_nothing():
    """Clause 33 puts it in the fault class rather than among the typed
    absences, and the fault class has no candidates in it."""
    cell = _cell("B", widths={
        "short": _width("short"),
        "long": _width("long", residues={"A": -5.0})})
    blockers = ps.binding_blockers(_record(cells={"B": cell}))
    assert any("candidate A" in one and "residue" in one for one in blockers)


def test_a_partition_reject_blocks_the_whole_profile():
    cell = _cell("B", widths={
        "short": _width("short", slopes=dict(SLOPES, L=60.0, P3=50.0)),
        "long": _width("long")})
    blockers = ps.binding_blockers(_record(cells={"B": cell}))
    assert any("clause 22 REJECTS" in one for one in blockers)


def test_a_certified_candidate_measured_at_only_one_width_blocks():
    """The selection takes the highest MINIMUM gain across the two widths, so
    a candidate measured at one width has no minimum to take."""
    cell = _cell("B", widths={"short": _width("short")})
    blockers = ps.binding_blockers(_record(cells={"B": cell}))
    assert any("candidate L was not measured" in one for one in blockers)
    assert any("candidate Q was not measured" in one for one in blockers)


def test_a_seam_that_never_fired_blocks():
    cell = _cell("B", structural={
        "short": _structural(),
        "long": _structural({"qmm": {pi.FORWARD: 7 * DEPTH,
                                     pi.BACKWARD: 7 * ADAPTED - 3}})})
    blockers = ps.binding_blockers(_record(cells={"B": cell}))
    assert any("region counts differ" in one for one in blockers)


def test_a_marked_pass_that_changed_the_computation_blocks():
    cell = _cell("B", structural={"short": _structural(),
                                  "long": _structural(identity_ok=False)})
    blockers = ps.binding_blockers(_record(cells={"B": cell}))
    assert any("do not compute the same thing" in one for one in blockers)


def test_the_structural_pass_is_read_at_every_width_and_not_only_the_first():
    """The widths run different batches through the same seams, so a fault at
    the second one would otherwise be recorded and never read."""
    for broken in ("short", "long"):
        cell = _cell("B", structural={
            name: _structural(identity_ok=name != broken)
            for name in ("short", "long")})
        blockers = ps.binding_blockers(_record(cells={"B": cell}))
        assert any(f"{broken}: the marked pass" in one for one in blockers)


def test_a_seam_replaced_under_the_structural_pass_blocks():
    cell = _cell("B", structural={"short": _structural(),
                                  "long": _structural(foreign=["attn-core"])})
    blockers = ps.binding_blockers(_record(cells={"B": cell}))
    assert any("was replaced while the structural pass ran" in one
               for one in blockers)


def test_no_blocker_mentions_a_gate_amendment_five_removed():
    """The compile transfer, the reconciliation and the instrument-cost band
    all had a written reason to go, and a blocker naming one would mean a
    gate came back without its clause."""
    text = ps.binding_blockers.__doc__
    for gate in ("compile transfer", "reconciliation", "instrument band"):
        assert gate in text
    cell = _cell("B", widths={
        "short": _width("short", slopes=dict(SLOPES, Q=-15.0)),
        "long": _width("long")})
    blockers = " ".join(ps.binding_blockers(_record(cells={"B": cell})))
    assert "compile" not in blockers and "instrument" not in blockers


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
def test_a_plan_may_not_name_a_kernel_for_a_profile_of_stock():
    for banned in ("kept_candidates", "measurement_module"):
        with pytest.raises(RunInvalid, match="installs nothing"):
            ps.validate_plan(_plan(**{banned: ["abc"]}))


def test_the_fixed_plan_says_in_writing_that_nothing_was_installed():
    fixed = ps.validate_plan(_plan())
    assert fixed["kept_candidates"] == []
    assert fixed["measurement_module"] == ps.STOCK_MEASUREMENT_MODULE
    assert fixed["widths"] == {"B": ["short", "long"]}


def test_schema_one_is_refused_rather_than_read_as_schema_two():
    with pytest.raises(RunInvalid, match="unknown plan schema"):
        ps.validate_plan(_plan(schema_version=1))


def test_every_required_key_is_required():
    for key in ("bands", "seed", "cells", "resolution", "memory_ceiling_gb"):
        plan = _plan()
        del plan[key]
        with pytest.raises(RunInvalid, match="missing"):
            ps.validate_plan(plan)


@pytest.mark.parametrize("floors", [
    {"short": R}, {"short": R, "long": 0.0}, {"short": R, "long": -1.0},
    {"short": R, "long": "small"},
])
def test_a_plan_without_a_positive_floor_at_every_width_is_refused(floors):
    with pytest.raises(RunInvalid, match="resolution floor"):
        ps.validate_plan(_plan(resolution={"floors_ms": floors,
                                           "addendum_sha256": "x"}))


def test_a_floor_with_no_addendum_behind_it_is_refused():
    """A reading whose floor cannot be traced to the measurement that set it
    is a reading whose gates nobody can check."""
    with pytest.raises(RunInvalid, match="no addendum"):
        ps.validate_plan(_plan(resolution={"floors_ms": {"short": R,
                                                         "long": R}}))


def test_a_cell_that_needs_one_band_may_not_pin_two():
    with pytest.raises(RunInvalid, match="need exactly"):
        ps.validate_plan(_plan(cells=["C"]))


def test_a_band_pointed_at_the_wrong_slice_is_refused():
    """Two datasets have two supervised fractions, which IS candidate L's
    floor, so a width comparison drawn from the wrong slice would carry a
    second variable and report it as width."""
    plan = _plan()
    plan["bands"]["long"] = "bench/.data/ultrachat-64"
    with pytest.raises(RunInvalid, match="registered at band"):
        ps.validate_plan(plan)


@pytest.mark.parametrize("over", [{"cells": []}, {"cells": ["Z"]},
                                  {"rounds": 0}])
def test_a_plan_that_cannot_produce_a_reading_is_refused(over):
    with pytest.raises(RunInvalid):
        ps.validate_plan(_plan(**over))


# ---------------------------------------------------------------------------
# The recording
# ---------------------------------------------------------------------------
def test_a_refused_run_says_so_in_the_file_name():
    import datetime

    day = datetime.date(2026, 8, 21)
    bound = ps.recording_path("/tmp", day, kind="binding", binding=True)
    refused = ps.recording_path("/tmp", day, kind="binding", binding=False)
    assert bound.name.endswith("2026-08-21.json")
    assert refused.name.endswith("REFUSED.json")


def test_the_name_follows_the_verdict_and_not_the_idle_gate_alone():
    """A run that measured everything cleanly on a quiet machine and then
    failed a fit gate used to land under the name of one that bound."""
    import datetime
    import inspect

    signature = inspect.signature(ps.recording_path)
    assert "closing_idle" not in signature.parameters
    assert "binding" in signature.parameters
    assert "REFUSED" in ps.recording_path(
        "/tmp", datetime.date(2026, 8, 21), kind="binding",
        binding=False).name


def test_a_recording_is_never_overwritten(tmp_path):
    runtime = ps.SystemRuntime()
    path = tmp_path / "rec.json"
    runtime.write_record(path, {"a": 1})
    with pytest.raises(PreconditionFailed, match="never overwritten"):
        runtime.write_record(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


# ---------------------------------------------------------------------------
# The parent loop
# ---------------------------------------------------------------------------
class _Runtime:
    def __init__(self, cells=("B",)):
        self.events = []
        self.lock_result = (True, "acquired")
        self.idle_results = [{"idle": True, "blockers": []},
                             {"idle": True, "blockers": []}]
        self.cells = cells
        self.writes = []

    def acquire_lock(self):
        self.events.append("lock")
        return self.lock_result

    def release_lock(self):
        self.events.append("release")

    def fingerprint(self):
        return {"cores": 12, "chip": "Apple M3 Pro"}

    def idle_check(self, cores):
        return self.idle_results.pop(0)

    def require_memory(self, budget, cell):
        self.events.append(f"memory:{cell}")

    def preflight(self, plan):
        self.events.append("preflight")
        return {"plan_sha256": "plan-sha"}

    def run_cell(self, plan, provenance, cell):
        self.events.append(f"run:{cell}")
        record = _cell(cell)
        del record["readings"]
        return record

    def write_record(self, path, record):
        self.writes.append((Path(path), record))

    def today(self):
        return __import__("datetime").date(2026, 8, 21)


def test_a_clean_run_binds_and_writes_schema_two():
    runtime = _Runtime()
    assert ps.run_profile(_plan(), runtime) == 0
    (path, record), = runtime.writes
    assert record["schema_version"] == 2
    assert record["units"]["time"] == "ms"
    assert record["binding"] is True
    assert "REFUSED" not in path.name
    assert record["profile_matrix"]["width_order"] == list(rules.WIDTH_ORDER)
    assert set(record["profile_matrix"]["entries"]) == set(rules.CANDIDATES)


def test_the_readings_are_derived_by_the_parent_and_not_by_the_child():
    """The child measures and the parent reduces, so a child cannot publish a
    number no rule ever saw the samples behind."""
    runtime = _Runtime()
    ps.run_profile(_plan(), runtime)
    (_path, record), = runtime.writes
    readings = record["cells"]["B"]["readings"]
    assert set(readings) == {"short", "long"}
    assert readings["long"]["entries"]["Q"]["type"] == "reading"


def test_a_run_that_went_busy_at_the_closing_gate_records_and_does_not_bind():
    runtime = _Runtime()
    runtime.idle_results = [{"idle": True}, {"idle": False, "blockers": ["x"]}]
    assert ps.run_profile(_plan(), runtime) == 1
    (path, record), = runtime.writes
    assert record["binding"] is False
    assert "REFUSED" in path.name


def test_a_busy_machine_at_the_opening_gate_measures_nothing():
    from memory_guard import EXIT_NOT_IDLE

    runtime = _Runtime()
    runtime.idle_results = [{"idle": False, "blockers": ["busy"]}]
    assert ps.run_profile(_plan(), runtime) == EXIT_NOT_IDLE
    assert runtime.writes == []
    assert "release" in runtime.events


def test_an_invalid_plan_never_reaches_the_lock():
    runtime = _Runtime()
    assert ps.run_profile(_plan(schema_version=1), runtime) == EXIT_PRECONDITION
    assert runtime.events == []
    assert runtime.writes == []


def test_the_lock_is_released_even_when_a_cell_raises():
    runtime = _Runtime()

    def boom(plan, provenance, cell):
        raise PreconditionFailed("cell died")

    runtime.run_cell = boom
    assert ps.run_profile(_plan(), runtime) == EXIT_PRECONDITION
    assert runtime.events[-1] == "release"
    assert runtime.writes == []


def test_there_is_no_calibrate_mode_left_to_call():
    """Its band priced the boundary instrument, and Amendment 5 clause 1 took
    the instrument out of the timing path, so the mode has no referent."""
    import inspect

    assert "calibrate" not in inspect.signature(ps.run_profile).parameters
    with pytest.raises(SystemExit):
        ps.main(["--plan", "/nonexistent", "--calibrate"])


# ---------------------------------------------------------------------------
# The ruling, which step 8 owns
# ---------------------------------------------------------------------------
def test_no_ruling_is_computed_from_a_schema_two_recording_yet():
    """A `--decide` running the old arithmetic over a schema-2 recording would
    produce a ruling with the right shape and the wrong meaning."""
    with pytest.raises(RunInvalid, match="Amendment 7"):
        ps.decide({"binding": True}, {"binding": True})


def test_the_decide_entry_point_refuses_rather_than_crashes(tmp_path):
    profile = tmp_path / "p.json"
    sweep = tmp_path / "s.json"
    profile.write_text(json.dumps({"binding": True}))
    sweep.write_text(json.dumps({"binding": True}))
    assert ps.run_decide(profile, sweep, tmp_path / "out.json") \
        == EXIT_PRECONDITION
    assert not (tmp_path / "out.json").exists()


# ---------------------------------------------------------------------------
# The child's refusals, which are all a caller ever sees of it
# ---------------------------------------------------------------------------
def test_the_parent_really_can_spawn_this_file_as_its_child(tmp_path,
                                                            monkeypatch):
    """A real subprocess, end to end, with the device switched off.

    Everything else about the child is tested by calling `child_main`
    directly, which proves nothing about the wiring around it: whether the
    entrypoint resolves, whether the argv is the one the child's parser
    accepts, whether the task file survives the round trip, and whether the
    exit code comes back as the numbered refusal the detached runner reads.
    """
    from harness_runner import ChildRefusal, spawn_child

    monkeypatch.setenv("KV_FORCE_NO_METAL", "1")
    with pytest.raises(ChildRefusal) as raised:
        spawn_child(
            {"kind": "profile", "cell": "cell B", "cell_name": "B",
             "plan": ps.validate_plan(_plan()), "provenance": {},
             "budget_gb": 8.0},
            tmp_path, wall_cap_s=120.0,
            child_entrypoint=Path(ps.__file__).resolve())
    assert raised.value.returncode == EXIT_NO_DEVICE
    assert ps.child_exit_for_parent(raised.value.returncode) == EXIT_NO_DEVICE
    assert list(tmp_path.iterdir()) == []


def test_the_child_refuses_when_the_device_is_switched_off(tmp_path,
                                                           monkeypatch):
    monkeypatch.setenv("KV_FORCE_NO_METAL", "1")
    assert ps.child_main(tmp_path / "task.json",
                         tmp_path / "out.json") == EXIT_NO_DEVICE


def test_the_child_refuses_a_kind_it_does_not_implement(tmp_path, monkeypatch):
    monkeypatch.delenv("KV_FORCE_NO_METAL", raising=False)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"kind": "train", "budget_gb": 8.0,
                                "parent_pid": 1}))
    out = tmp_path / "out.json"
    monkeypatch.setattr(ps, "_ChildGuard", lambda *a, **k: type(
        "G", (), {"check": lambda self, cell: 0.0})())
    monkeypatch.setattr(ps, "stack_record", lambda: {})
    monkeypatch.setattr(ps, "_check_child_inputs", lambda task, stack: None)
    assert ps.child_main(task, out) == EXIT_PRECONDITION
    assert not out.exists()
