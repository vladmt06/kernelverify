"""The knob instrument, everywhere it can be pinned without a device.

Almost all of it can be. A knob's job is to turn a set of timings into one
credited number and to refuse when that number cannot mean what it would be
read to mean, and both halves are arithmetic over samples.

What genuinely needs the GPU - that a dial changes the step, that an arm at
full setting is bit-identical to stock, that a compiled arm keeps its own
graph, that a trace during timing is caught - lives in
tests/test_profile_knobs_live.py.

The tests that matter most here are the refusals and the two counterexamples
that an outside review produced against earlier drafts of the registered text.
A knob that produced a plausible wrong number would look exactly like a knob
that worked.
"""

import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import profile_knobs as pk  # noqa: E402
from decode_rules import RunInvalid  # noqa: E402


# --- the fit ---------------------------------------------------------------


def test_a_known_line_comes_back_exactly():
    phis = [1.0, 0.75, 0.5, 0.25]
    times = [3.0 + 8.0 * phi for phi in phis]
    got = pk.fit(phis, times)
    assert got.slope == pytest.approx(8.0)
    assert got.intercept == pytest.approx(3.0)
    assert got.r_squared == pytest.approx(1.0)
    assert got.max_residual == pytest.approx(0.0, abs=1e-12)
    assert got.is_a_dial


def test_a_known_line_with_known_noise_recovers_its_slope():
    """Noise of a known size moves the slope by less than the noise itself.

    The residual is NOT the injected noise: least squares refits the line to
    the perturbed points, so alternating noise of 0.01 leaves residuals of
    0.012. What matters is that both stay the same order as the noise and
    that the recovered slope is still the one that was put in.
    """
    phis = [1.0, 0.75, 0.5, 0.25]
    noise = [0.01, -0.01, 0.01, -0.01]
    times = [3.0 + 8.0 * phi + n for phi, n in zip(phis, noise)]
    got = pk.fit(phis, times)
    assert got.slope == pytest.approx(8.0, abs=0.1)
    assert got.r_squared > 0.999
    assert abs(got.max_residual) < 3 * max(abs(n) for n in noise)
    assert abs(got.max_residual) < pk.RESIDUAL_LIMIT * abs(got.slope)


def test_a_flat_line_is_not_a_dial():
    got = pk.fit([1.0, 0.75, 0.5, 0.25], [5.0, 5.0, 5.0, 5.0])
    assert got.slope == pytest.approx(0.0)
    assert not got.is_a_dial


def test_a_line_that_slopes_the_wrong_way_is_not_a_dial():
    phis = [1.0, 0.75, 0.5, 0.25]
    got = pk.fit(phis, [3.0 - 2.0 * phi for phi in phis])
    assert got.slope < 0
    assert not got.is_a_dial


def test_the_fit_refuses_what_it_cannot_fit():
    with pytest.raises(RunInvalid, match="no line to fit"):
        pk.fit([1.0, 0.5], [1.0, 2.0, 3.0])
    with pytest.raises(RunInvalid, match="at least three settings"):
        pk.fit([1.0, 0.5], [1.0, 2.0])
    with pytest.raises(RunInvalid, match="same value"):
        pk.fit([0.5, 0.5, 0.5], [1.0, 2.0, 3.0])


# --- the reading -----------------------------------------------------------


def _fits(slopes):
    return tuple(pk.Fit(slope=s, intercept=0.0, r_squared=1.0,
                        max_residual=0.0) for s in slopes)


def _reading(kind=pk.RETUNE, *, slopes=(8.0, 8.1, 7.9, 8.0, 8.05),
             intercept=12.0, stock=20.0, ablated=None, scaffold_slope=0.1,
             offset=0.02, floor=0.05, r_squared=1.0, max_residual=0.0,
             pooled_slope=None, span=0.75, scaffold_range=0.0,
             scaffold_r_squared=1.0, scaffold_residual=0.0):
    """One reading, defaulting to a clean one.

    `scaffold_range` defaults BELOW the resolution floor, so clause 26 clamps
    the scaffold to zero and its slope limit is not binding; a test that means
    to exercise that limit sets a range at or above the floor and puts the
    scaffold in the fitted case.
    """
    return pk.KnobReading(
        candidate="Q" if kind == pk.RETUNE else "L",
        kind=kind,
        stock_median=stock,
        per_round=_fits(slopes),
        pooled=pk.Fit(
            slope=statistics.median(slopes) if pooled_slope is None
            else pooled_slope,
            intercept=intercept, r_squared=r_squared,
            max_residual=max_residual),
        scaffold=pk.Fit(slope=scaffold_slope, intercept=0.0,
                        r_squared=scaffold_r_squared,
                        max_residual=scaffold_residual),
        scaffold_offset=offset,
        ablated_median=ablated,
        resolution_floor=floor,
        span=span,
        scaffold_range=scaffold_range,
    )


def test_a_retune_is_credited_with_its_slope_alone():
    reading = _reading(pk.RETUNE)
    assert reading.residue == 0.0
    assert reading.attributed == pytest.approx(reading.slope)
    assert reading.share == pytest.approx(reading.slope / 20.0)
    assert reading.blockers() == []


def test_a_rewrite_is_credited_with_its_slope_plus_its_residue():
    reading = _reading(pk.REWRITE, intercept=12.0, ablated=9.0)
    assert reading.residue == pytest.approx(3.0)
    assert reading.attributed == pytest.approx(reading.slope + 3.0)
    assert reading.blockers() == []


def test_a_residue_below_the_resolution_floor_is_recorded_as_zero():
    reading = _reading(pk.REWRITE, intercept=12.0, ablated=11.99, floor=0.05)
    assert reading.residue == 0.0


def test_a_residue_above_the_resolution_floor_is_kept():
    reading = _reading(pk.REWRITE, intercept=12.0, ablated=11.90, floor=0.05)
    assert reading.residue == pytest.approx(0.10)


def test_a_rewrite_without_an_ablated_arm_refuses():
    with pytest.raises(RunInvalid, match="no ablated arm"):
        _reading(pk.REWRITE, ablated=None)


def test_a_retune_with_an_ablated_arm_refuses():
    with pytest.raises(RunInvalid, match="nothing may use"):
        _reading(pk.RETUNE, ablated=9.0)


def test_an_unknown_kind_refuses():
    with pytest.raises(RunInvalid, match="neither of the two kinds"):
        pk.KnobReading(candidate="Q", kind="rewrite-ish", stock_median=20.0,
                       per_round=_fits([8.0]), pooled=_fits([8.0])[0],
                       scaffold=_fits([0.1])[0], scaffold_offset=0.0)


def test_the_slope_is_the_median_of_the_per_round_slopes():
    reading = _reading(slopes=(1.0, 2.0, 3.0, 4.0, 100.0))
    assert reading.slope == pytest.approx(3.0)


# --- the gates -------------------------------------------------------------


def test_a_dial_that_does_not_move_the_step_is_refused():
    problems = _reading(slopes=(0.0,) * 5).blockers()
    assert any("not positive" in p for p in problems)


def test_a_poor_fit_is_refused():
    problems = _reading(r_squared=0.95).blockers()
    assert any("R-squared" in p for p in problems)


def test_a_large_residual_is_refused():
    problems = _reading(max_residual=4.0).blockers()
    assert any("largest residual" in p for p in problems)


def test_a_scaffold_whose_own_cost_moves_with_the_dial_is_refused():
    problems = _reading(scaffold_slope=2.0, scaffold_range=1.5).blockers()
    assert any("scaffold's own slope" in p for p in problems)


def test_a_scaffold_offset_past_three_resolution_floors_is_refused():
    problems = _reading(offset=0.20, floor=0.05).blockers()
    assert any("scaffold offset" in p for p in problems)


def test_a_missing_resolution_floor_is_refused_rather_than_assumed():
    problems = _reading(floor=None).blockers()
    assert any("no resolution floor" in p for p in problems)


def test_a_share_that_is_not_a_fraction_is_refused():
    problems = _reading(stock=4.0).blockers()
    assert any("not a fraction" in p for p in problems)


def test_a_fit_can_pass_while_its_scaffold_fails():
    reading = _reading(r_squared=1.0, max_residual=0.0, scaffold_slope=5.0,
                       scaffold_range=3.75)
    problems = reading.blockers()
    assert any("scaffold" in p for p in problems)
    assert not any("R-squared" in p for p in problems)


def test_a_scaffold_can_pass_while_its_fit_fails():
    reading = _reading(r_squared=0.5, scaffold_slope=0.01, offset=0.0)
    problems = reading.blockers()
    assert any("R-squared" in p for p in problems)
    assert not any("scaffold" in p for p in problems)


# --- clause 26's readability and clause 33's residue fault ------------------


def test_a_residue_more_negative_than_the_floor_is_a_fault_not_a_clamp():
    """Clause 33 settles clause 18 against clause 27's absence machinery.

    It says the step ran SLOWER with the operation removed than the fit
    predicts without its scaling part, which cannot be true of any step, so
    it refuses the profile rather than typing that candidate absent.
    """
    reading = _reading(pk.REWRITE, intercept=12.0, ablated=12.20, floor=0.05)
    assert reading.raw_residue == pytest.approx(-0.20)
    assert reading.residue_is_a_fault
    assert any("negative by more than" in p for p in reading.blockers())


def test_a_residue_negative_but_inside_the_floor_is_zero_and_no_fault():
    reading = _reading(pk.REWRITE, intercept=12.0, ablated=12.02, floor=0.05)
    assert reading.raw_residue == pytest.approx(-0.02)
    assert reading.residue == 0.0
    assert not reading.residue_is_a_fault


def test_a_retune_never_has_a_residue_fault():
    assert not _reading(pk.RETUNE).residue_is_a_fault


def test_a_median_slope_that_is_not_positive_blocks_where_the_pooled_one_is():
    """Clause 26's sign precondition, on the estimator the share consumes.

    An excursion condition written on an absolute value cannot see a sign, so
    without this a dial whose registered slope is negative clears `10R` on
    magnitude and is admitted by a rule committed text already refuses.
    """
    reading = _reading(slopes=(-60.0, -20.0, -20.0, 52.0, 20.0),
                       pooled_slope=20.0)
    assert reading.slope == pytest.approx(-20.0)
    assert reading.pooled.is_a_dial
    problems = reading.blockers()
    assert any("median of the per-round slopes" in p for p in problems)
    assert not any("the fitted slope is" in p for p in problems)


def test_the_excursion_is_demanded_of_both_estimators():
    """Clause 26's reproduction: collinear medians, disagreeing estimators.

    Five exact per-round lines whose arm medians are collinear give a pooled
    slope of 20 and a median slope of 0.05, so at `R = 0.5` the pooled
    excursion clears `10R` while the median's is below `R` itself. The share
    is built from the median, so a condition on the pooled fit alone would
    certify a number no rule consumes.
    """
    reading = _reading(slopes=(0.05,) * 5, pooled_slope=20.0, span=0.75,
                       floor=0.5, offset=0.0)
    problems = reading.blockers()
    assert any("median excursion" in p for p in problems)
    assert not any("pooled excursion" in p for p in problems)


def test_a_scaffold_that_moves_less_than_the_floor_is_clamped_to_zero():
    reading = _reading(scaffold_slope=2.0, scaffold_range=0.004, floor=0.05)
    assert reading.scaffold_case == "clamped"
    assert reading.effective_scaffold_slope == 0.0
    assert not any("scaffold's own slope" in p for p in reading.blockers())


def test_a_scaffold_that_moved_and_whose_line_describes_it_keeps_its_slope():
    reading = _reading(scaffold_slope=2.0, scaffold_range=1.5, floor=0.05)
    assert reading.scaffold_case == "fitted"
    assert reading.effective_scaffold_slope == pytest.approx(2.0)
    assert any("scaffold's own slope" in p for p in reading.blockers())


def test_a_scaffold_that_moved_without_its_line_describing_it_is_unreadable():
    """Clause 26's third case, and the reading of it this code registers.

    Scaffold medians of 100, 130, 110 and 105.5 at `R = 1` have a range of 30
    and a fitted excursion of 1.05, so the excursion condition HOLDS while
    the fit describes nothing: its coefficient of determination is 0.0012 and
    its largest residual is over eighteen times `R`.

    Clause 26 words its third case as "neither of the other two conditions
    holds", which read literally would leave this state uncovered while the
    same clause calls the three cases exhaustive. The only consistent reading
    is that the third case takes everything at or above `R` the second does
    not, and this pins it.
    """
    reading = _reading(scaffold_slope=1.4, scaffold_range=30.0, floor=1.0,
                       offset=0.0, scaffold_r_squared=0.0012,
                       scaffold_residual=18.0)
    assert reading.scaffold_case == "unreadable"
    assert any("does not describe that movement" in p
               for p in reading.blockers())


def test_a_scaffold_range_that_was_never_recorded_blocks_rather_than_passes():
    reading = _reading(scaffold_range=None)
    assert reading.scaffold_case == "unjudged"
    assert any("three cases cannot be told apart" in p
               for p in reading.blockers())


def test_a_span_that_was_never_recorded_blocks_rather_than_passes():
    problems = _reading(span=None).blockers()
    assert any("actual span was" in p for p in problems)


# --- the credited ratio ----------------------------------------------------


def test_the_ratio_takes_the_worst_pairing_the_samples_permit():
    stock = _reading(pk.RETUNE, slopes=(10.0, 12.0, 14.0))
    floor = _reading(pk.RETUNE, slopes=(4.0, 5.0, 6.0))
    assert pk.ratio_lo(stock, floor) == pytest.approx(10.0 / 6.0)


def test_the_ratio_refuses_to_pair_two_different_kinds():
    stock = _reading(pk.REWRITE, ablated=9.0)
    floor = _reading(pk.RETUNE)
    with pytest.raises(RunInvalid, match="not the same kind"):
        pk.ratio_lo(stock, floor)


def test_the_ratio_refuses_a_floor_that_reduces_to_nothing():
    stock = _reading(pk.RETUNE, slopes=(10.0, 12.0))
    floor = _reading(pk.RETUNE, slopes=(0.0, 0.0))
    with pytest.raises(RunInvalid, match="zero or less"):
        pk.ratio_lo(stock, floor)


# --- the kill rule's reducer, and the counterexample that produced it -------


def test_a_shape_ratio_is_a_ratio_of_sums_not_an_average_of_ratios():
    """The counterexample an outside review used against an earlier draft.

    Equal counts, stock costs 100 and 20, floor costs 100 and 10. The average
    of the per-direction ratios is 1.50 and the ratio of summed costs is 1.09,
    so the two land on opposite sides of the kill rule's 1.10 threshold.
    """
    counts = {"forward": 1, "backward": 1}
    stock = {"forward": 100.0, "backward": 20.0}
    floor = {"forward": 100.0, "backward": 10.0}

    average_of_ratios = statistics.mean([100.0 / 100.0, 20.0 / 10.0])
    assert average_of_ratios == pytest.approx(1.50)

    got = pk.shape_ratio(stock, floor, counts)
    assert got == pytest.approx(120.0 / 110.0)
    assert got < 1.10 < average_of_ratios


def test_a_shape_ratio_weights_by_the_measured_call_counts():
    counts = {"forward": 36, "backward": 16}
    stock = {"forward": 2.0, "backward": 2.0}
    floor = {"forward": 1.0, "backward": 4.0}
    assert pk.shape_ratio(stock, floor, counts) == pytest.approx(
        (36 * 2.0 + 16 * 2.0) / (36 * 1.0 + 16 * 4.0))


def test_a_shape_ratio_refuses_a_direction_it_was_never_given():
    with pytest.raises(RunInvalid, match="never measured"):
        pk.shape_ratio({"forward": 1.0}, {"forward": 1.0},
                       {"forward": 1, "backward": 1})


def test_a_shape_ratio_refuses_a_floor_summing_to_nothing():
    with pytest.raises(RunInvalid, match="zero or less"):
        pk.shape_ratio({"forward": 1.0}, {"forward": 0.0}, {"forward": 1})


# --- the attention dial's registered criterion ------------------------------


# The registry's own order, not a copy of it, so the names in these tests can
# never drift from the names the dials are built under.
COMPLETENESS = pk.ATTENTION_DIALS
MOST, MIDDLE, LEAST = COMPLETENESS
ALL_ELIGIBLE = {name: 4 for name in COMPLETENESS}


def test_the_cheapest_dial_wins():
    ruling = pk.choose_dial({LEAST: 0.02, MIDDLE: 0.30, MOST: 0.50},
                            COMPLETENESS, ALL_ELIGIBLE)
    assert ruling["chosen"] == LEAST
    assert ruling["by_completeness"] is False


def test_a_tie_on_price_goes_to_the_more_complete_dial():
    ruling = pk.choose_dial({LEAST: 0.020, MIDDLE: 0.025, MOST: 0.028},
                            COMPLETENESS, ALL_ELIGIBLE)
    assert ruling["chosen"] == MOST
    assert ruling["by_completeness"] is True
    assert ruling["tied"] == list(COMPLETENESS)


def test_a_price_gap_wider_than_the_tie_band_is_not_a_tie():
    ruling = pk.choose_dial({LEAST: 0.02, MOST: 0.05}, COMPLETENESS,
                            ALL_ELIGIBLE)
    assert ruling["chosen"] == LEAST


def test_the_cheapest_dial_loses_if_it_cannot_place_three_settings():
    """Clause 15's eligibility test, which outranks the price entirely."""
    ruling = pk.choose_dial(
        {LEAST: 0.02, MIDDLE: 0.30, MOST: 0.50}, COMPLETENESS,
        {LEAST: 2, MIDDLE: 4, MOST: 4})
    assert ruling["chosen"] == MIDDLE
    assert ruling["refused"] == {LEAST: 2}


def test_every_dial_below_the_setting_floor_refuses_outright():
    with pytest.raises(RunInvalid, match="fewer than 3 distinct settings"):
        pk.choose_dial({LEAST: 0.02, MOST: 0.01}, COMPLETENESS,
                       {LEAST: 1, MOST: 2})


def test_a_dial_with_no_completeness_rank_refuses():
    with pytest.raises(RunInvalid, match="no registered completeness rank"):
        pk.choose_dial({"invented-dial": 0.01}, COMPLETENESS, {"invented-dial": 4})


def test_a_dial_with_no_setting_count_refuses():
    with pytest.raises(RunInvalid, match="no count of realisable settings"):
        pk.choose_dial({LEAST: 0.01}, COMPLETENESS, {})


def test_choosing_among_no_dials_refuses():
    with pytest.raises(RunInvalid, match="no dial was priced"):
        pk.choose_dial({}, COMPLETENESS, ALL_ELIGIBLE)


# --- the attention ladder, without a model ---------------------------------


def test_attention_sees_one_token_fewer_than_the_batch_carries():
    """mlx-lm's own loss trains on `batch[:, :-1]`, so the widths differ by one."""
    assert pk.attention_width(97) == 96
    assert pk.attention_width(769) == 768


def test_two_dial_positions_that_round_to_one_size_are_one_setting():
    """A repeated point raises R-squared and says nothing about linearity."""
    ladder = {1.00: {"kept": 8}, 0.75: {"kept": 6}, 0.50: {"kept": 4},
              0.25: {"kept": 2}}
    assert pk.realisable_settings(ladder) == (1.00, 0.75, 0.50, 0.25)

    collapsed = {1.00: {"kept": 2}, 0.75: {"kept": 2}, 0.50: {"kept": 1},
                 0.25: {"kept": 1}}
    assert pk.realisable_settings(collapsed) == (1.00, 0.50)
    assert len(pk.realisable_settings(collapsed)) < pk.MIN_DIAL_SETTINGS


# --- the two reductions over rounds ----------------------------------------


PHIS = {"phi=1.00": 1.0, "phi=0.75": 0.75, "phi=0.50": 0.5, "phi=0.25": 0.25}


def _samples(slopes, intercept=3.0):
    return {label: [intercept + slope * phi for slope in slopes]
            for label, phi in PHIS.items()}


def test_one_fit_per_round_across_that_rounds_settings():
    fits = pk.per_round_fits(_samples([8.0, 9.0, 10.0]), PHIS, rounds=3)
    assert [round(f.slope, 6) for f in fits] == [8.0, 9.0, 10.0]


def test_the_pooled_fit_runs_over_the_arm_medians():
    pooled = pk.pooled_fit(_samples([8.0, 9.0, 100.0]), PHIS)
    assert pooled.slope == pytest.approx(9.0)
    assert pooled.intercept == pytest.approx(3.0)


def test_the_registered_ladder_is_four_settings_descending():
    assert pk.PHIS == (1.00, 0.75, 0.50, 0.25)
    assert pk.WARMUPS == 3
    assert pk.ROUNDS == 5
