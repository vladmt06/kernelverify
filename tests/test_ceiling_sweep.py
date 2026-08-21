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

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import ceiling_sweep as cs  # noqa: E402
import profile_rules as rules  # noqa: E402
from decode_rules import RunInvalid  # noqa: E402


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
