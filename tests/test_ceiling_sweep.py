"""The ceiling sweep, everywhere it can be pinned without a device.

The arithmetic is clause 8's and it has a wrong version that looks right: an
average of per-direction ratios rather than a ratio of summed costs. Clause 8
names the wrong one explicitly and supplies the case where the two land on
opposite sides of the threshold, so that case is a test here rather than a
sentence in a document.

What needs the GPU - that a dense cotangent measures a different operation
from a `sum()`-driven one, that the stock arm dispatches a quantized matmul
and the dense arm does not - lives in tests/test_ceiling_sweep_live.py.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import ceiling_sweep as cs  # noqa: E402
import profile_knobs as pk  # noqa: E402
import profile_rules as rules  # noqa: E402
import profile_stock as ps  # noqa: E402
from decode_rules import RunInvalid  # noqa: E402
from harness_runner import PreconditionFailed  # noqa: E402


def _samples(*, stock_f, stock_b, dense_f, dense_b, rounds=5):
    return {
        f"{cs.STOCK}:{cs.FORWARD}": [stock_f] * rounds,
        f"{cs.STOCK}:{cs.BACKWARD}": [stock_b] * rounds,
        f"{cs.DENSE}:{cs.FORWARD}": [dense_f] * rounds,
        f"{cs.DENSE}:{cs.BACKWARD}": [dense_b] * rounds,
    }


def _counts(forward=36, backward=16):
    return {cs.FORWARD: forward, cs.BACKWARD: backward}


def _all_arms(value=1.0):
    return {arm.label: [value] * 5 for arm in cs.kill_manifest()}


# --- the manifest ----------------------------------------------------------
def test_the_kill_bench_is_forty_eight_arms():
    """Amendment 10 clause 44 corrected an earlier draft's 24.

    The draft counted shapes, directions and widths and dropped the
    implementation, which is the one dimension a ceiling comparison cannot do
    without: a ceiling needs the thing AND the thing it is a ceiling for.
    """
    manifest = cs.kill_manifest()
    assert len(manifest) == 48
    assert len(rules.SHAPES) * len(rules.WIDTH_ORDER) * 2 * 2 == 48
    assert {arm.implementation for arm in manifest} == set(cs.IMPLEMENTATIONS)
    assert {arm.direction for arm in manifest} == set(cs.DIRECTIONS)


def test_every_registered_shape_is_at_both_registered_widths():
    """Clause 8 requires a shape within 1.10 at BOTH widths before it counts,
    so a shape measured at one has nothing to be shown at both."""
    seen = {}
    for arm in cs.kill_manifest():
        seen.setdefault(arm.shape, set()).add(arm.width)
    assert sorted(seen) == sorted(rules.SHAPES)
    for shape, widths in seen.items():
        assert widths == set(rules.WIDTH_ORDER), shape


@pytest.mark.parametrize("field, value", [
    ("shape", "S9"), ("width", "medium"),
    ("implementation", "approximate"), ("direction", "sideways"),
])
def test_an_arm_outside_the_registered_axes_refuses(field, value):
    kwargs = dict(label="x", shape="S1", width="short",
                  implementation=cs.STOCK, direction=cs.FORWARD)
    kwargs[field] = value
    with pytest.raises(RunInvalid):
        cs.KillArm(**kwargs)


# --- clause 8's reduction --------------------------------------------------
def test_the_ratio_is_of_summed_costs_and_not_an_average_of_ratios():
    """Clause 8's own worked case, which is why it names the wrong one.

    On equal counts with stock costs 100 and 20 against floor costs 100 and
    10, the average of per-direction ratios is 1.50 and the ratio of sums is
    1.09. They land on opposite sides of the 1.10 threshold, so only one of
    them answers "is stock close to the ceiling in the time it actually
    spends".
    """
    ratios = cs.per_round_shape_ratios(
        _samples(stock_f=100.0, stock_b=20.0, dense_f=100.0, dense_b=10.0),
        _counts(forward=1, backward=1))
    assert ratios[0] == pytest.approx(120.0 / 110.0)
    assert ratios[0] == pytest.approx(1.0909, abs=1e-4)
    average_of_ratios = (100.0 / 100.0 + 20.0 / 10.0) / 2
    assert average_of_ratios == pytest.approx(1.50)
    assert (ratios[0] <= rules.KILL_RATIO) != (
        average_of_ratios <= rules.KILL_RATIO)


def test_the_counts_weight_the_directions_and_are_not_assumed_equal():
    """Under LoRA a projection runs `depth` times forward and fewer than
    `adapted` times backward, so one count cannot describe both."""
    even = cs.per_round_shape_ratios(
        _samples(stock_f=10.0, stock_b=100.0, dense_f=10.0, dense_b=10.0),
        _counts(forward=1, backward=1))[0]
    forward_heavy = cs.per_round_shape_ratios(
        _samples(stock_f=10.0, stock_b=100.0, dense_f=10.0, dense_b=10.0),
        _counts(forward=36, backward=1))[0]
    assert even == pytest.approx(110.0 / 20.0)
    assert forward_heavy == pytest.approx(460.0 / 370.0)
    assert forward_heavy < even


def test_the_sums_are_taken_within_a_round_and_not_across_rounds():
    """Clause 8 says WITHIN, and the two disagree.

    Summing across rounds first would pair a stock forward from one round with
    a dense backward from another, which is the combination clause 31 rejects
    for the uncertainty box and the same fault in a different place.
    """
    measured = {
        f"{cs.STOCK}:{cs.FORWARD}": [10.0, 20.0],
        f"{cs.STOCK}:{cs.BACKWARD}": [20.0, 10.0],
        f"{cs.DENSE}:{cs.FORWARD}": [10.0, 10.0],
        f"{cs.DENSE}:{cs.BACKWARD}": [10.0, 10.0],
    }
    ratios = cs.per_round_shape_ratios(measured, _counts(1, 1))
    assert ratios == [pytest.approx(1.5), pytest.approx(1.5)]


def test_the_ratio_the_kill_rule_reads_is_the_largest_round():
    """The largest is the one furthest from the ceiling and therefore the
    least likely to kill, which is the conservative direction for a rule whose
    effect is to remove a candidate from contention."""
    measured = {
        f"{cs.STOCK}:{cs.FORWARD}": [10.0, 30.0, 20.0],
        f"{cs.STOCK}:{cs.BACKWARD}": [7.0, 7.0, 7.0],
        f"{cs.DENSE}:{cs.FORWARD}": [10.0, 10.0, 10.0],
        f"{cs.DENSE}:{cs.BACKWARD}": [7.0, 7.0, 7.0],
    }
    entry = cs.shape_at_width(measured, _counts(1, 0))
    assert entry.ratio == pytest.approx(3.0)
    assert entry.numerator_high == pytest.approx(30.0)
    assert entry.denominator_low == pytest.approx(10.0)


def test_arms_that_ran_a_different_number_of_rounds_refuse():
    measured = _samples(stock_f=1.0, stock_b=1.0, dense_f=1.0, dense_b=1.0)
    measured[f"{cs.DENSE}:{cs.BACKWARD}"] = [1.0] * 4
    with pytest.raises(RunInvalid, match="WITHIN"):
        cs.per_round_shape_ratios(measured, _counts())


def test_a_missing_implementation_refuses_rather_than_halving_the_sum():
    measured = _samples(stock_f=1.0, stock_b=1.0, dense_f=1.0, dense_b=1.0)
    del measured[f"{cs.DENSE}:{cs.FORWARD}"]
    with pytest.raises(RunInvalid, match="absent"):
        cs.per_round_shape_ratios(measured, _counts())


def test_a_shape_that_ran_in_neither_direction_refuses():
    """A shape counted at zero contributes nothing to a ratio of summed costs,
    so its ratio would be 0/0 or an artefact of whichever arm ran."""
    with pytest.raises(RunInvalid, match="zero times"):
        cs.per_round_shape_ratios(
            _samples(stock_f=1.0, stock_b=1.0, dense_f=1.0, dense_b=1.0),
            _counts(forward=0, backward=0))


def test_a_round_carrying_a_non_positive_sample_is_dropped():
    """Amendment 11 clause 48. Nothing here subtracts, so a sample at or below
    zero is a clock excursion rather than a small cost. The ROUND goes, not
    the sample, because a ratio is a sum taken WITHIN a round."""
    measured = _samples(stock_f=2.0, stock_b=2.0, dense_f=1.0, dense_b=1.0,
                        rounds=4)
    measured[f"{cs.DENSE}:{cs.FORWARD}"] = [1.0, 0.0, 1.0, 1.0]
    ratios = cs.per_round_shape_ratios(measured, _counts(1, 1))
    assert len(ratios) == 3
    assert ratios == [pytest.approx(2.0)] * 3
    assert ratios.dropped_rounds == (1,)


def test_a_dropped_round_is_dropped_from_the_corners_too():
    """The corners the ratio was taken over must come from the rounds the
    ratio was taken over, or a reader checking the pairing checks a round that
    never entered it."""
    measured = _samples(stock_f=2.0, stock_b=2.0, dense_f=1.0, dense_b=1.0,
                        rounds=3)
    measured[f"{cs.STOCK}:{cs.FORWARD}"] = [2.0, 99.0, 2.0]
    measured[f"{cs.DENSE}:{cs.FORWARD}"] = [1.0, -1.0, 1.0]
    entry = cs.shape_at_width(measured, _counts(1, 1))
    assert entry.numerator_high == pytest.approx(4.0)
    assert entry.ratio == pytest.approx(2.0)


def test_a_shape_whose_every_round_was_dropped_refuses():
    measured = _samples(stock_f=1.0, stock_b=1.0, dense_f=0.0, dense_b=0.0)
    with pytest.raises(RunInvalid, match="no ratio at all"):
        cs.per_round_shape_ratios(measured, _counts())


def test_a_sample_that_is_not_finite_refuses():
    measured = _samples(stock_f=float("inf"), stock_b=1.0,
                        dense_f=1.0, dense_b=1.0)
    with pytest.raises(RunInvalid, match="needs a number"):
        cs.per_round_shape_ratios(measured, _counts())


# --- the readings the kill rule consumes -----------------------------------
def test_the_readings_are_what_the_committed_kill_rule_reads():
    """The point of this test is the handoff: the sweep produces exactly what
    `profile_rules.kill_q` consumes, and neither invents the other's shape."""
    counts = {shape: _counts() for shape in rules.SHAPES}
    readings = cs.kill_readings(_all_arms(), counts)
    assert sorted(readings) == sorted(rules.SHAPES)
    ruling = rules.kill_q(readings)
    assert ruling["certified"] is False
    assert ruling["killed"] is True


def _with_headroom(samples, shape, width, factor=4.0):
    for direction in cs.DIRECTIONS:
        samples[f"{width}:{shape}:{cs.STOCK}:{direction}"] = [factor] * 5
    return samples


def test_headroom_at_one_width_takes_that_shape_out_of_the_at_ceiling_set():
    """Clause 8's two-width reduction: a shape counts only where stock is
    within 1.10 at BOTH widths, so headroom at either width takes it out."""
    counts = {shape: _counts() for shape in rules.SHAPES}
    readings = cs.kill_readings(
        _with_headroom(_all_arms(), "S5", "long"), counts)
    ruling = rules.kill_q(readings)
    assert "S5" not in ruling["at_ceiling"]
    assert sorted(ruling["at_ceiling"]) == ["S1", "S2", "S3", "S4", "S6"]


def test_one_shape_with_headroom_still_kills_because_the_rule_is_all_but_one():
    """The exact boundary, and it is not the intuitive one.

    Five of six IS all-but-one, so a single shape with headroom does not save
    candidate Q. This is the case Amendment 7 clause 36's concession is about:
    the per-shape evidence can say "one shape has real headroom" while the
    rule still reports killed, and after the demotion no terminal reads it.
    """
    counts = {shape: _counts() for shape in rules.SHAPES}
    readings = cs.kill_readings(
        _with_headroom(_all_arms(), "S5", "long"), counts)
    ruling = rules.kill_q(readings)
    assert ruling["needed"] == rules.KILL_SHAPES == len(rules.SHAPES) - 1
    assert ruling["killed"] is True


def test_two_shapes_with_headroom_keep_the_candidate_alive():
    counts = {shape: _counts() for shape in rules.SHAPES}
    samples = _with_headroom(_all_arms(), "S5", "long")
    samples = _with_headroom(samples, "S3", "short")
    ruling = rules.kill_q(cs.kill_readings(samples, counts))
    assert sorted(ruling["at_ceiling"]) == ["S1", "S2", "S4", "S6"]
    assert ruling["killed"] is False


def test_a_hole_in_the_recorded_arms_refuses():
    samples = _all_arms()
    del samples["short:S1:stock:forward"]
    with pytest.raises(RunInvalid, match="hole is not an absence"):
        cs.kill_readings(samples, {shape: _counts() for shape in rules.SHAPES})


def test_an_arm_nobody_registered_refuses():
    samples = _all_arms()
    samples["short:S7:stock:forward"] = [1.0] * 5
    with pytest.raises(RunInvalid, match="manifest does not name"):
        cs.kill_readings(samples, {shape: _counts() for shape in rules.SHAPES})


def test_a_shape_with_no_call_counts_refuses():
    counts = {shape: _counts() for shape in rules.SHAPES}
    del counts["S3"]
    with pytest.raises(RunInvalid, match="no call counts"):
        cs.kill_readings(_all_arms(), counts)


# --- clause 3 supplies the counts ------------------------------------------
def _structural(**over):
    tally = {"qmm": {}, "head-matmul": {}}
    for name, (out_dims, in_dims) in rules.SHAPES.items():
        region = "head-matmul" if name == "S5" else "qmm"
        tally[region][f"{out_dims}x{in_dims}"] = {
            cs.FORWARD: 36, cs.BACKWARD: 16}
    record = {"shape_counts": tally}
    record.update(over)
    return record


def test_the_counts_come_from_the_structural_pass_and_not_from_geometry():
    """Clause 8 says "with the call counts supplied by the structural pass of
    clause 3", and a share and a ratio that disagree about how often an
    operation ran are two readings of different workloads."""
    counts = cs.shape_counts(_structural())
    assert sorted(counts) == sorted(rules.SHAPES)
    assert counts["S5"] == {cs.FORWARD: 36, cs.BACKWARD: 16}


def test_a_shape_the_structural_pass_never_saw_refuses():
    structural = _structural()
    out_dims, in_dims = rules.SHAPES["S6"]
    del structural["shape_counts"]["qmm"][f"{out_dims}x{in_dims}"]
    with pytest.raises(RunInvalid, match="recorded none"):
        cs.shape_counts(structural)


def test_counts_for_one_shape_are_summed_across_the_regions_that_ran_it():
    """The tied head is a quantized matmul and section 3.3 puts it in two
    regions, so a shape that appears under more than one region is one shape
    with one pair of counts and not two."""
    structural = _structural()
    out_dims, in_dims = rules.SHAPES["S5"]
    structural["shape_counts"]["qmm"][f"{out_dims}x{in_dims}"] = {
        cs.FORWARD: 1, cs.BACKWARD: 1}
    counts = cs.shape_counts(structural)
    assert counts["S5"] == {cs.FORWARD: 37, cs.BACKWARD: 17}


# --- what a reported quantity carries --------------------------------------
def test_the_kill_bench_carries_an_observed_spread_and_says_it_is_not_a_bound():
    """Amendment 10 clause 44 retires clause 36's per-shape and per-direction
    resolution contexts, so there is no `R` here and no demand. What clause 37
    registers in their place is this, with the label attached to the number."""
    spread = cs.observed_spread({"short:S1:stock:forward": [10.0, 11.0, 10.5]})
    assert spread["certified"] is False
    assert cs.CERTIFIED is False
    assert spread["observed_spread_pct"]["short:S1:stock:forward"] > 0
    assert "not an interval at any registered rate" in (
        spread["observed_spread_is_not_a_bound"])


# --- candidate L's bench, Amendment 12 -------------------------------------
def test_the_bench_is_nine_arms_with_a_reference_and_no_ablation():
    """Amendment 10 clause 46's count, read as Amendment 12 clause 49 reads it.

    Four knob, four scaffold, one reference. No ablation, because clause 19
    registers candidate L's `c_floor` as zero and `KnobReading` refuses a
    candidate that measured a residue nothing may consume.
    """
    for width in rules.WIDTH_ORDER:
        manifest = cs.loss_manifest(width)
        assert len(manifest) == 9
        roles = [arm.role for arm in manifest]
        assert roles.count(pk.KNOB) == 4
        assert roles.count(pk.SCAFFOLD) == 4
        assert roles.count(pk.REFERENCE) == 1
        assert pk.ABLATION not in roles


def test_the_bench_takes_a_reference_arm_and_not_a_stock_one():
    """Clause 49 voids clause 46's word "stock" for that arm.

    The floor runs on the supervised rows and stock on all of them, so an
    offset between the two contains the row-count difference, and the
    row-count difference IS the floor.
    """
    roles = {arm.role for arm in cs.loss_manifest("short")}
    assert pk.REFERENCE in roles
    assert pk.STOCK not in roles


def test_a_reference_arm_carrying_a_dial_setting_refuses():
    with pytest.raises(RunInvalid, match="defined by not having one"):
        cs.LossArm(label="x", width="short", role=pk.REFERENCE,
                   nominal_phi=1.0)


def test_a_knob_arm_without_a_dial_setting_refuses():
    with pytest.raises(RunInvalid, match="carries one"):
        cs.LossArm(label="x", width="short", role=pk.KNOB)


def test_an_ablation_arm_on_this_bench_refuses():
    """Clause 19 registers `c_floor` as zero, so a residue arm here would
    measure something no rule may read."""
    with pytest.raises(RunInvalid, match="no ablation"):
        cs.LossArm(label="x", width="short", role=pk.ABLATION)


def test_the_two_ladders_place_the_same_actual_sizes():
    """Clause 51: the pairing clause 9 asks for cannot hold across two
    contexts, and what it was protecting is the SETTINGS, which must."""
    vocab = 151936
    bench = cs.vocabulary_ladder(vocab)
    assert list(bench) == list(pk.PHIS)
    assert bench[1.0] == vocab
    cs.same_settings(bench, cs.vocabulary_ladder(vocab))


def test_a_bench_ladder_at_other_sizes_than_the_step_refuses():
    """A ratio taken across two dials is not the quantity clause 9 defines,
    however carefully each half was measured."""
    bench = cs.vocabulary_ladder(151936)
    elsewhere = cs.vocabulary_ladder(151936 // 2)
    with pytest.raises(RunInvalid, match="these are two"):
        cs.same_settings(bench, elsewhere)


def test_the_supervised_count_comes_from_the_profile_s_own_context():
    """Section 4.2's floor is on the SUPERVISED rows and clause 20 pins which
    batch those are, so the count is read and never assumed."""
    assert cs.supervised_rows({"supervised": 3675, "supervised_of": 4224}) == 3675


def test_a_context_with_no_supervised_count_refuses():
    with pytest.raises(RunInvalid, match="no supervised count"):
        cs.supervised_rows({"supervised_of": 4224})


def test_more_supervised_rows_than_the_batch_holds_refuses():
    with pytest.raises(RunInvalid, match="more rows than the batch"):
        cs.supervised_rows({"supervised": 500, "supervised_of": 256})


def test_a_supervised_count_of_zero_refuses():
    with pytest.raises(RunInvalid, match="not a measurement"):
        cs.supervised_rows({"supervised": 0, "supervised_of": 256})


def test_the_bench_reduces_through_the_same_code_the_step_uses():
    """The handoff, and the point of it: candidate L's bench and candidate Q's
    floor take clause 5's offset against their own reference arm through ONE
    reduction, which landed with candidate Q's floor and is reused here."""
    samples, roles = {}, []
    for arm in cs.loss_manifest("short"):
        if arm.role == pk.KNOB:
            value = 30.0 + 60.0 * arm.nominal_phi
        elif arm.role == pk.SCAFFOLD:
            value = 90.1
        else:
            value = 90.0
        samples[arm.label] = [value] * pk.ROUNDS
        roles.append(pk.ArmRole(label=arm.label, candidate="L", role=arm.role,
                                phi=arm.nominal_phi))
    reading = pk.reduce_width(samples, tuple(roles), resolution_floor=0.05,
                              rounds=pk.ROUNDS, family=pk.FLOOR)["L"]
    assert reading.reference_median == pytest.approx(90.0)
    assert reading.scaffold_offset == pytest.approx(0.1)
    assert reading.pooled.slope == pytest.approx(60.0)


def test_a_floor_reading_carries_no_share_and_says_why():
    """Clause 19 puts candidate L's SHARE in the step and its `d` on a bench,
    so the bench has no step total for a share to be a fraction of."""
    reading = pk.KnobReading(
        candidate="L", kind=pk.REWRITE, stock_median=None,
        per_round=(pk.Fit(60.0, 30.0, 1.0, 0.0),),
        pooled=pk.Fit(60.0, 30.0, 1.0, 0.0),
        scaffold=pk.Fit(0.0, 0.0, 1.0, 0.0), scaffold_offset=0.1,
        family=pk.FLOOR, resolution_floor=0.05, span=0.75,
        scaffold_range=0.0)
    assert reading.residue == 0.0
    assert reading.raw_residue == 0.0
    assert reading.attributed == pytest.approx(60.0)
    with pytest.raises(RunInvalid, match="no share"):
        reading.share


def test_a_floor_reading_that_measured_a_residue_refuses():
    """Clause 19 registers candidate L's `c_floor` as ZERO, so an ablated arm
    on this bench measures something no rule may read."""
    with pytest.raises(RunInvalid, match="registers its `c_floor` as zero"):
        pk.KnobReading(
            candidate="L", kind=pk.REWRITE, stock_median=None,
            per_round=(pk.Fit(60.0, 30.0, 1.0, 0.0),),
            pooled=pk.Fit(60.0, 30.0, 1.0, 0.0),
            scaffold=pk.Fit(0.0, 0.0, 1.0, 0.0), scaffold_offset=0.1,
            family=pk.FLOOR, ablated_median=5.0, resolution_floor=0.05,
            span=0.75, scaffold_range=0.0)


def test_a_stock_less_width_whose_family_has_no_reference_refuses():
    """Without a stock arm and without a reference arm, clause 5's scaffold
    offset has no baseline at all and would be silently taken against nothing."""
    samples, roles = {}, []
    for arm in cs.loss_manifest("short"):
        if arm.role == pk.REFERENCE:
            continue
        samples[arm.label] = [50.0] * pk.ROUNDS
        roles.append(pk.ArmRole(label=arm.label, candidate="L", role=arm.role,
                                phi=arm.nominal_phi))
    with pytest.raises(RunInvalid, match="no baseline at all"):
        pk.reduce_width(samples, tuple(roles), resolution_floor=0.05,
                        rounds=pk.ROUNDS, family=pk.FLOOR)


# --- the recording, and what it refuses ------------------------------------
def _plan(**over):
    plan = {"schema_version": 1,
            "resolution": {"floors_ms": {"short": 0.1, "long": 1.0},
                           "addendum_sha256": "addendum-sha"},
            "model": {"hidden": 2560, "vocab": 151936}}
    plan.update(over)
    return plan


def _loss_samples(width, *, slope=60.0, intercept=30.0, scaffold=90.1,
                  reference=90.0, rounds=None):
    rounds = pk.ROUNDS if rounds is None else rounds
    samples, roles = {}, []
    for arm in cs.loss_manifest(width):
        if arm.role == pk.KNOB:
            value = intercept + slope * arm.nominal_phi
        elif arm.role == pk.SCAFFOLD:
            value = scaffold
        else:
            value = reference
        samples[arm.label] = [value] * rounds
        roles.append(pk.ArmRole(label=arm.label, candidate="L", role=arm.role,
                                phi=arm.nominal_phi))
    return samples, tuple(roles)


def _sweep_record(**over):
    counts = {shape: _counts() for shape in rules.SHAPES}
    loss_samples, loss_roles = {}, {}
    for width in rules.WIDTH_ORDER:
        loss_samples[width], loss_roles[width] = _loss_samples(width)
    kwargs = dict(
        counts=counts, contexts=_contexts(),
        resolution_floors_ms={"short": 0.1, "long": 1.0},
        fingerprint={"cores": 12}, idle_before={"idle": True},
        idle_after={"idle": True})
    kwargs.update(over)
    return cs.sweep_recording(_all_arms(), loss_samples, loss_roles, **kwargs)


def _contexts():
    return {
        "short": {"width": "short", "supervised": 91,
                  "supervised_of": 256},
        "long": {"width": "long", "supervised": 3699,
                 "supervised_of": 4224},
    }


def _fake_sweep_runtime(monkeypatch, observed):
    import machine_state

    class Lock:
        def __init__(self, owner):
            observed["lock_owner"] = owner

        def acquire(self):
            observed["lock_acquired"] = True
            return True, "acquired"

        def release(self):
            observed["lock_released"] = True

    monkeypatch.setattr(machine_state, "MeasurementLock", Lock)
    monkeypatch.setattr(machine_state, "fingerprint",
                        lambda: {"cores": 12})
    monkeypatch.setattr(machine_state, "idle_check",
                        lambda _cores: {"idle": True})
    monkeypatch.setattr(cs, "run_kill_bench", lambda **_kwargs: {})

    def run_loss(context, *, width, **_kwargs):
        observed.setdefault("loss", []).append(
            (width, context["width"], context["supervised"]))
        return {}, ()

    def recording(*_args, **kwargs):
        observed["recording"] = kwargs
        return {"kind": "ceiling-sweep"}

    monkeypatch.setattr(cs, "run_loss_bench", run_loss)
    monkeypatch.setattr(cs, "sweep_recording", recording)
    monkeypatch.setattr(cs, "sweep_blockers", lambda _record: [])


def test_run_sweep_refuses_a_missing_registered_width_before_timing(
        tmp_path, monkeypatch):
    timed = []
    monkeypatch.setattr(
        cs, "run_kill_bench", lambda **_kwargs: timed.append(True))

    contexts = _contexts()
    del contexts["long"]
    with pytest.raises(RunInvalid, match="long"):
        cs.run_sweep(_plan(), contexts, structural=_structural(),
                     results_dir=tmp_path)
    assert timed == []


def test_run_sweep_refuses_a_context_filed_under_the_wrong_width_before_timing(
        tmp_path, monkeypatch):
    timed = []
    monkeypatch.setattr(
        cs, "run_kill_bench", lambda **_kwargs: timed.append(True))

    contexts = _contexts()
    contexts["long"]["width"] = "short"
    with pytest.raises(RunInvalid, match="long.*short"):
        cs.run_sweep(_plan(), contexts, structural=_structural(),
                     results_dir=tmp_path)
    assert timed == []


def test_each_loss_bench_uses_its_width_s_own_supervised_count(
        tmp_path, monkeypatch):
    observed = {}
    _fake_sweep_runtime(monkeypatch, observed)

    cs.run_sweep(_plan(), _contexts(), structural=_structural(),
                 results_dir=tmp_path)

    assert observed["loss"] == [
        ("short", "short", 91),
        ("long", "long", 3699),
    ]


def test_run_sweep_feeds_shape_counts_the_structural_tally(
        tmp_path, monkeypatch):
    observed = {}
    _fake_sweep_runtime(monkeypatch, observed)

    cs.run_sweep(_plan(), _contexts(), structural=_structural(),
                 results_dir=tmp_path)

    assert observed["recording"]["counts"]["S5"] == {
        cs.FORWARD: 36, cs.BACKWARD: 16}
    assert observed["lock_owner"] == "ceiling_sweep"
    assert observed["lock_acquired"] is True
    assert observed["lock_released"] is True


def _profile_recording():
    contexts = _contexts()
    return {
        "schema_version": 2,
        "cells": {
            rules.PRIMARY_CELL: {
                "widths": {width: {"context": dict(contexts[width])}
                           for width in rules.WIDTH_ORDER},
                # Amendment 17 clause 67 and Amendment 18 clause 71: the
                # pass runs at the short width, so the tally lives there and
                # every other width carries clause 68's typed absence.
                "structural": {
                    width: (dict(_structural(), taken=True)
                            if width == ps.STRUCTURAL_WIDTH
                            else {"taken": False,
                                  "reason_code": ps.STRUCTURAL_NOT_TAKEN})
                    for width in rules.WIDTH_ORDER},
            },
        },
    }


def test_the_tally_is_read_where_clause_67_runs_the_pass():
    """Amendment 18 clause 71. Before it, this reader took a tally from EVERY
    width and required them to agree; clause 67 leaves only one width with a
    pass, so the agreement check is gone with its evidence and the tally is
    read where the pass runs."""
    contexts, structural = cs._profile_inputs(_profile_recording())
    assert tuple(contexts) == rules.WIDTH_ORDER
    assert structural["taken"] is True
    assert cs.shape_counts(structural)


@pytest.mark.parametrize("width,entry,needle", [
    (ps.STRUCTURAL_WIDTH,
     {"taken": False, "reason_code": ps.STRUCTURAL_NOT_TAKEN},
     "clause 67 takes the structural pass at width"),
    ("long", None, "which clause 67 does not run it at"),
    ("long", {"taken": False, "reason_code": "the machine was busy"},
     "not the absence clause 67 registers"),
    ("long", {"reason_code": ps.STRUCTURAL_NOT_TAKEN},
     "rather than a boolean"),
])
def test_the_sweep_refuses_a_structural_record_the_clauses_forbid(
        width, entry, needle):
    """The same typed union the profile's own reader validates. A tally is the
    kill rule's weighting, so a forged one silently changes what the ceiling
    was compared over."""
    record = _profile_recording()
    record["cells"][rules.PRIMARY_CELL]["structural"][width] = (
        dict(_structural(), taken=True) if entry is None else entry)
    with pytest.raises(RunInvalid) as caught:
        cs._profile_inputs(record)
    assert needle in str(caught.value)


def test_a_missing_structural_entry_refuses_the_sweep():
    record = _profile_recording()
    del record["cells"][rules.PRIMARY_CELL]["structural"]["long"]
    with pytest.raises(RunInvalid, match="INCOMPLETE"):
        cs._profile_inputs(record)


def test_the_entry_point_refuses_an_unknown_profile_recording_schema(
        tmp_path, monkeypatch, capsys):
    plan_path = tmp_path / "plan.json"
    profile_path = tmp_path / "profile.json"
    plan_path.write_text(json.dumps(_plan()))
    profile = _profile_recording()
    profile["schema_version"] = 1
    profile_path.write_text(json.dumps(profile))
    timed = []
    monkeypatch.setattr(
        cs, "run_sweep", lambda *_args, **_kwargs: timed.append(True))

    assert cs.main(["--plan", str(plan_path),
                    "--profile", str(profile_path)]) == ps.EXIT_PRECONDITION
    assert "profile recording is schema 1" in capsys.readouterr().out
    assert timed == []


@pytest.mark.parametrize("missing, expected", [
    ("cell", "no deciding cell"),
    ("contexts", "no contexts"),
    ("structural", "no structural record"),
])
def test_the_entry_point_names_each_missing_profile_input(
        missing, expected, tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    profile_path = tmp_path / "profile.json"
    plan_path.write_text(json.dumps(_plan()))
    profile = _profile_recording()
    if missing == "cell":
        profile["cells"].clear()
    elif missing == "contexts":
        del profile["cells"][rules.PRIMARY_CELL]["widths"]
    else:
        del profile["cells"][rules.PRIMARY_CELL]["structural"]
    profile_path.write_text(json.dumps(profile))

    assert cs.main(["--plan", str(plan_path),
                    "--profile", str(profile_path)]) == ps.EXIT_PRECONDITION
    assert expected in capsys.readouterr().out


def test_the_entry_point_joins_the_deciding_cell_s_recorded_inputs(
        tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.json"
    profile_path = tmp_path / "profile.json"
    plan_path.write_text(json.dumps(_plan()))
    profile = _profile_recording()
    profile_path.write_text(json.dumps(profile))
    observed = {}

    def run(plan, contexts, *, structural, results_dir, pass_index,
            **_kwargs):
        observed.update(plan=plan, contexts=contexts,
                        structural=structural, results_dir=results_dir,
                        pass_index=pass_index)
        return {"kind": "ceiling-sweep"}, []

    monkeypatch.setattr(cs, "run_sweep", run)
    assert cs.main(["--plan", str(plan_path),
                    "--profile", str(profile_path),
                    "--pass-index", "2"]) == 0
    assert observed["contexts"] == _contexts()
    assert observed["structural"] == profile["cells"][
        rules.PRIMARY_CELL]["structural"][rules.WIDTH_ORDER[0]]
    assert observed["pass_index"] == 2


def test_the_entry_point_refuses_a_forbidden_long_width_structural_pass(
        tmp_path, monkeypatch, capsys):
    """REPLACES test_the_entry_point_refuses_widths_with_different_structural_tallies.

    That test drove the cross-width agreement check, which Amendment 18 clause
    71 removes because clause 67 leaves only one width with a pass to agree
    with. What the entry point refuses now is a recording that carries a pass
    at a width the registered arrangement does not run one at, which is the
    same recording the old test would have been handed."""
    plan_path = tmp_path / "plan.json"
    profile_path = tmp_path / "profile.json"
    plan_path.write_text(json.dumps(_plan()))
    profile = _profile_recording()
    profile["cells"][rules.PRIMARY_CELL]["structural"]["long"] = dict(
        _structural(), taken=True)
    profile_path.write_text(json.dumps(profile))
    timed = []
    monkeypatch.setattr(
        cs, "run_sweep", lambda *_args, **_kwargs: timed.append(True))

    assert cs.main(["--plan", str(plan_path),
                    "--profile", str(profile_path)]) == ps.EXIT_PRECONDITION
    assert "clause 67 does not run it at" in capsys.readouterr().out
    assert timed == []


def test_the_entry_point_reports_the_path_the_sweep_actually_wrote(
        tmp_path, monkeypatch, capsys):
    import datetime

    plan_path = tmp_path / "plan.json"
    profile_path = tmp_path / "profile.json"
    plan_path.write_text(json.dumps(_plan()))
    profile_path.write_text(json.dumps(_profile_recording()))
    observed = {}
    _fake_sweep_runtime(monkeypatch, observed)
    monkeypatch.setattr(ps, "RESULTS_DIR", tmp_path)

    class Midnight:
        days = iter((datetime.date(2026, 8, 21),
                     datetime.date(2026, 8, 22)))

        @classmethod
        def today(cls):
            return next(cls.days)

    monkeypatch.setattr(cs, "date", Midnight)
    assert cs.main(["--plan", str(plan_path),
                    "--profile", str(profile_path),
                    "--pass-index", "2"]) == 0
    output = capsys.readouterr().out
    assert "ceiling-sweep-2026-08-21-pass2.json" in output
    assert "ceiling-sweep-2026-08-22-pass2.json" not in output
    assert (tmp_path / "ceiling-sweep-2026-08-21-pass2.json").exists()


def test_the_entry_point_reuses_the_profile_s_lock_refusal_exit(
        tmp_path, monkeypatch, capsys):
    plan_path = tmp_path / "plan.json"
    profile_path = tmp_path / "profile.json"
    plan_path.write_text(json.dumps(_plan()))
    profile_path.write_text(json.dumps(_profile_recording()))

    def locked(*_args, **_kwargs):
        raise cs.LockHeld("held by another measurement")

    monkeypatch.setattr(cs, "run_sweep", locked)
    assert cs.main(["--plan", str(plan_path),
                    "--profile", str(profile_path)]) == ps.EXIT_LOCK_HELD
    assert "held by another measurement" in capsys.readouterr().out


def test_no_pass_index_keeps_the_legacy_sweep_names_byte_identical():
    import datetime

    day = datetime.date(2026, 8, 21)
    assert cs.sweep_path(
        "/tmp", day, refused=False,
        pass_index=None).name == "ceiling-sweep-2026-08-21.json"
    assert cs.sweep_path(
        "/tmp", day, refused=True,
        pass_index=None).name == "ceiling-sweep-2026-08-21.REFUSED.json"


@pytest.mark.parametrize("pass_index", [0, -1, True, "2"])
def test_a_sweep_path_refuses_an_invalid_pass_index(pass_index):
    import datetime

    with pytest.raises(RunInvalid, match="positive integer"):
        cs.sweep_path(
            "/tmp", datetime.date(2026, 8, 21), refused=True,
            pass_index=pass_index)


def test_three_sweep_passes_have_three_distinct_recording_paths():
    import datetime

    day = datetime.date(2026, 8, 21)
    paths = [cs.sweep_path(
        "/tmp", day, refused=True, pass_index=index)
             for index in (1, 2, 3)]
    assert [path.name for path in paths] == [
        "ceiling-sweep-2026-08-21-pass1.REFUSED.json",
        "ceiling-sweep-2026-08-21-pass2.REFUSED.json",
        "ceiling-sweep-2026-08-21-pass3.REFUSED.json",
    ]
    assert len(set(paths)) == 3


@pytest.mark.parametrize("pass_index", [0, -1, True, "2"])
def test_run_sweep_refuses_an_invalid_pass_before_timing(
        pass_index, tmp_path, monkeypatch):
    timed = []
    monkeypatch.setattr(
        cs, "run_kill_bench", lambda **_kwargs: timed.append(True))
    with pytest.raises(RunInvalid, match="positive integer"):
        cs.run_sweep(
            _plan(), _contexts(), structural=_structural(),
            results_dir=tmp_path, pass_index=pass_index)
    assert timed == []


def test_the_same_sweep_pass_index_is_still_never_overwritten(
        tmp_path, monkeypatch):
    observed = {}
    _fake_sweep_runtime(monkeypatch, observed)
    kwargs = dict(structural=_structural(), results_dir=tmp_path,
                  pass_index=2)

    cs.run_sweep(_plan(), _contexts(), **kwargs)
    with pytest.raises(PreconditionFailed, match="never overwritten"):
        cs.run_sweep(_plan(), _contexts(), **kwargs)


def test_the_recording_carries_both_benches_and_no_verdict():
    """The sweep measures and the rules rule, and the two live in different
    files so a harness cannot quietly become the thing that decides."""
    record = _sweep_record()
    assert record["schema_version"] == 2
    assert record["kind"] == "ceiling-sweep"
    assert sorted(record["kill"]["per_shape"]) == sorted(rules.SHAPES)
    assert sorted(record["loss_bench"]["readings"]) == sorted(rules.WIDTH_ORDER)

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for value in node:
                yield from keys(value)

    # KEYS and not a substring search: the words appear in the prose that
    # labels a reported quantity, and prose explaining that a verdict reaches
    # no terminal is the opposite of carrying one.
    named = set(keys(record))
    for word in ("killed", "verdict", "selected", "terminal", "at_ceiling"):
        assert word not in named, f"the recording carries a {word!r} field"


def test_the_recording_keeps_the_context_for_each_measured_width():
    record = _sweep_record()
    assert record["contexts"] == _contexts()


def test_the_kill_half_is_labelled_reported_and_carries_no_certified_bound():
    """Amendment 7 clause 36 demoted it, and clause 37 registers what a
    reported quantity carries instead of a calibrated interval."""
    record = _sweep_record()
    assert record["kill"]["certified"] is False
    assert "not an interval at any registered rate" in (
        record["kill"]["observed_spread_is_not_a_bound"])


def test_candidate_a_is_recorded_as_having_no_floor_and_why():
    """Clause 27 gives it none, and an absence with no reason beside it reads
    as a measurement that failed."""
    record = _sweep_record()
    assert record["candidate_a"]["floor"] is None
    assert "ratio of 1.0" in record["candidate_a"]["reason"]


def test_candidate_l_s_reading_carries_clause_nineteen_s_two_facts():
    """Both are reported wherever candidate L's gain appears, and a reader of
    this file is one of those places."""
    record = _sweep_record()
    for width in rules.WIDTH_ORDER:
        reading = record["loss_bench"]["readings"][width]
        assert reading["c_floor_ms"] == 0.0
        assert reading["family"] == pk.FLOOR
        assert "in isolation" in reading["bench_exception"]
        assert reading["floor_slope_ms"] == pytest.approx(60.0)


def test_a_width_with_no_resolution_floor_refuses_the_recording():
    """Clause 21 forbids reusing another context's `R`, and clause 33 makes a
    missing context a refusal rather than a floor of zero."""
    with pytest.raises(RunInvalid, match="forbids"):
        _sweep_record(resolution_floors_ms={"short": 0.1})


def test_a_busy_machine_at_the_close_blocks():
    record = _sweep_record(idle_after={"idle": False})
    assert any("went busy" in reason for reason in cs.sweep_blockers(record))


def test_a_bench_missing_a_width_blocks():
    """The selection takes the highest minimum saving across both widths, so a
    floor at one width has no minimum to take."""
    record = _sweep_record()
    del record["loss_bench"]["readings"]["long"]
    assert any("did not run at" in reason
               for reason in cs.sweep_blockers(record))


def test_a_clean_sweep_has_no_blockers():
    assert cs.sweep_blockers(_sweep_record()) == []


def test_the_kill_half_contributes_no_blockers_because_it_is_reported():
    """A reported quantity reaches no terminal, so there is nothing here for a
    gate to protect. The concession is clause 36's and it is deliberate."""
    record = _sweep_record()
    for direction in cs.DIRECTIONS:
        record["kill"]["arms"][f"long:S5:{cs.STOCK}:{direction}"] = [
            1.0, 500.0, 1.0, 500.0, 1.0]
    assert cs.sweep_blockers(record) == []


@pytest.mark.parametrize("mutate, expected", [
    ({"schema_version": 2}, "schema 1"),
    ({"resolution": {"addendum_sha256": "x"}}, "no `resolution.floors_ms`"),
    ({"resolution": {"floors_ms": {"short": 0.1, "long": 0.0},
                     "addendum_sha256": "x"}}, "resolve any difference"),
    ({"resolution": {"floors_ms": {"short": 0.1, "long": 1.0}}},
     "names no addendum"),
    ({"model": {"hidden": 0, "vocab": 151936}}, "pinned dimensions"),
])
def test_a_sweep_plan_missing_what_it_needs_refuses(mutate, expected):
    with pytest.raises(RunInvalid, match=expected):
        cs.validate_sweep_plan(_plan(**mutate))


def test_a_sweep_plan_that_carries_everything_is_frozen():
    frozen = cs.validate_sweep_plan(_plan())
    assert frozen["floors_ms"] == {"short": 0.1, "long": 1.0}
    assert frozen["vocab"] == 151936


def test_an_exploratory_sweep_plan_has_no_floors_to_choose():
    frozen = cs.validate_sweep_plan(
        _plan(resolution={"exploratory": True}))
    assert frozen["exploratory"] is True
    assert frozen["floors_ms"] == {
        width: ps.EXPLORATORY_FLOOR_MS for width in rules.WIDTH_ORDER}

    for smuggled in ({"floors_ms": {"short": 0.1, "long": 1.0}},
                      {"addendum_sha256": "addendum-sha"}):
        with pytest.raises(RunInvalid, match="could have chosen them"):
            cs.validate_sweep_plan(_plan(
                resolution=dict({"exploratory": True}, **smuggled)))


def test_an_exploratory_sweep_recording_can_never_bind():
    record = _sweep_record(exploratory=True)
    assert record["kind"] == "exploratory"
    blockers = cs.sweep_blockers(record)
    assert any("unusable for binding by construction" in one
               for one in blockers)

    record["idle_after"] = {"idle": True}
    for reading in record["loss_bench"]["readings"].values():
        reading["blockers"] = []
    assert any("unusable for binding by construction" in one
               for one in cs.sweep_blockers(record))


def test_run_sweep_threads_the_exploratory_kind_without_an_addendum(
        tmp_path, monkeypatch):
    observed = {}
    _fake_sweep_runtime(monkeypatch, observed)

    record, _blockers = cs.run_sweep(
        _plan(resolution={"exploratory": True}), _contexts(),
        structural=_structural(), results_dir=tmp_path)

    assert observed["recording"]["exploratory"] is True
    assert "addendum_sha256" not in record


def test_the_benches_interleave_their_arms_instead_of_running_each_to_completion():
    """The fault this arrangement exists to prevent, pinned without a device.

    Timing one arm to completion before starting the next separates two arm
    medians by however long every arm between them took, so their difference
    carries that much machine drift and gets WORSE as rounds are added.
    Measured 2026-08-21 at the pinned 4B dimensions: tripling candidate L's
    rounds from 5 to 15 moved its scaffold offset spread from 0.999 to 6.621
    ms at the short width and from 48.026 to 109.666 at the long. Round noise
    shrinks with rounds; this grew, which is the signature of drift.
    """
    order = []
    calls = {name: (lambda name=name: order.append(name) or 0.0)
             for name in ("a", "b", "c")}
    samples = cs._timed_rounds(calls, warmups=2, rounds=4)

    assert {k: len(v) for k, v in samples.items()} == {"a": 4, "b": 4, "c": 4}
    timed = order[len(calls) * 2:]
    assert timed == ["a", "b", "c", "b", "c", "a", "c", "a", "b",
                     "a", "b", "c"], (
        "every arm runs inside every round and the starting arm rotates, so "
        "no arm always takes the coolest slot")
    runs = [timed[i:i + 4] for i in range(0, len(timed), 4)]
    assert not any(len(set(run)) == 1 for run in runs), (
        "an arm timed to completion before the next one starts is the "
        "arrangement that made more rounds read worse than fewer")


def test_a_round_over_no_arms_is_refused():
    with pytest.raises(RunInvalid, match="times nothing"):
        cs._timed_rounds({}, warmups=1, rounds=1)


def test_a_sweep_that_opens_on_a_busy_machine_refuses_before_timing(
        tmp_path, monkeypatch):
    """The profile refuses at its opening gate and the sweep recorded and read on.

    `run_profile` returns EXIT_NOT_IDLE before it measures anything, so no
    profile recording can carry a busy opening. The sweep sampled the same
    gate, stored the answer and never looked at it, so a sweep whose arms ran
    against a competing process landed as a usable recording.
    """
    import machine_state

    observed = {}
    _fake_sweep_runtime(monkeypatch, observed)
    monkeypatch.setattr(
        machine_state, "idle_check",
        lambda _cores: {"idle": False,
                        "blockers": ["Xcode at 400% CPU competing for the GPU"]})

    with pytest.raises(cs.NotIdle, match="competing"):
        cs.run_sweep(_plan(), _contexts(), structural=_structural(),
                     results_dir=tmp_path)
    assert "loss" not in observed
    assert "recording" not in observed
    assert observed["lock_released"] is True
