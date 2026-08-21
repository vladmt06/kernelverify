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
import profile_rules as rules  # noqa: E402
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


# --- the width reducer ------------------------------------------------------


def _width(*, knob_phis=(1.0, 0.75, 0.5, 0.25), scaffold_phis=None,
           slope=14.0, intercept=50.0, stock=64.0, scaffold_times=None,
           ablation=None, rounds=5, candidate="Q", reference=None):
    """One width's raw samples and the manifest that says what each label was."""
    scaffold_phis = knob_phis if scaffold_phis is None else scaffold_phis
    samples, roles = {}, [pk.ArmRole("stock", "stock", pk.STOCK)]
    samples["stock"] = [stock] * rounds
    for phi in knob_phis:
        label = f"{candidate}@{phi}"
        samples[label] = [intercept + slope * phi] * rounds
        roles.append(pk.ArmRole(label, candidate, pk.KNOB, phi))
    for index, phi in enumerate(scaffold_phis):
        label = f"{candidate}~{phi}"
        time = (scaffold_times[index] if scaffold_times is not None
                else stock + 0.01)
        samples[label] = [time] * rounds
        roles.append(pk.ArmRole(label, candidate, pk.SCAFFOLD, phi))
    if ablation is not None:
        samples[f"{candidate}!"] = [ablation] * rounds
        roles.append(pk.ArmRole(f"{candidate}!", candidate, pk.ABLATION))
    if reference is not None:
        samples[f"{candidate}="] = [reference] * rounds
        roles.append(pk.ArmRole(f"{candidate}=", candidate, pk.REFERENCE))
    return samples, roles


def test_a_width_reduces_to_one_reading_per_candidate():
    samples, roles = _width()
    readings = pk.reduce_width(samples, roles, resolution_floor=0.05)
    assert sorted(readings) == ["Q"]
    reading = readings["Q"]
    assert reading.kind == pk.RETUNE
    assert reading.stock_median == pytest.approx(64.0)
    assert reading.pooled.slope == pytest.approx(14.0)
    assert reading.span == pytest.approx(0.75)
    assert reading.blockers() == []


def test_the_scaffold_range_is_taken_over_arm_medians_not_raw_rounds():
    """Clause 26 says which reduction, and the two disagree.

    Four arm medians all equal to 100 can sit on rounds spanning 99 to 101,
    giving a range of 0 against a range of 2, and only the first says whether
    the ARM moved.
    """
    samples, roles = _width(scaffold_times=[100.0] * 4)
    for role in roles:
        if role.role == pk.SCAFFOLD:
            samples[role.label] = [99.0, 101.0, 100.0, 99.0, 101.0]
    reading = pk.reduce_width(samples, roles, resolution_floor=0.05)["Q"]
    assert reading.scaffold_range == pytest.approx(0.0)
    assert reading.scaffold_case == "clamped"


def test_the_span_is_the_actual_realisable_one_and_it_changes_the_verdict():
    """Clause 33's obligation, on its own numbers.

    A ladder returns the fractions it was ASKED for, and a quantized operand
    cannot always place them. At a slope of 14 the nominal span of 0.75 gives
    an excursion of 10.5, which clears a demand of 10, while the realisable
    span of 0.667 gives 9.33, which does not.
    """
    samples, roles = _width(knob_phis=(1.0, 0.75, 0.5, 1.0 / 3.0), slope=14.0)
    reading = pk.reduce_width(samples, roles, resolution_floor=1.0)["Q"]
    assert reading.span == pytest.approx(2.0 / 3.0)
    assert reading.pooled.slope * 0.75 == pytest.approx(10.5)
    assert reading.pooled.slope * reading.span == pytest.approx(9.333, abs=1e-3)
    assert any("excursion" in p for p in reading.blockers())


def test_a_manifest_naming_an_arm_with_no_samples_refuses():
    samples, roles = _width()
    del samples["Q@0.5"]
    with pytest.raises(RunInvalid, match="hole and not an absence"):
        pk.reduce_width(samples, roles, resolution_floor=0.05)


def test_samples_no_arm_in_the_manifest_claims_refuse():
    samples, roles = _width()
    samples["mystery"] = [1.0] * 5
    with pytest.raises(RunInvalid, match="does not name"):
        pk.reduce_width(samples, roles, resolution_floor=0.05)


def test_two_arms_under_one_label_refuse():
    samples, roles = _width()
    roles.append(pk.ArmRole("Q@0.5", "Q", pk.KNOB, 0.5))
    with pytest.raises(RunInvalid, match="silently overwrote"):
        pk.reduce_width(samples, roles, resolution_floor=0.05)


def test_an_arm_short_of_a_round_refuses():
    samples, roles = _width()
    samples["Q@0.5"] = samples["Q@0.5"][:-1]
    with pytest.raises(RunInvalid, match="against 5 rounds"):
        pk.reduce_width(samples, roles, resolution_floor=0.05)


def test_settings_that_placed_the_same_size_do_not_count_twice():
    """A ladder asks for four fractions; a quantized operand may place two.

    The labels stay distinct because the arms really did run, and what
    collapses is the ACTUAL fraction each one placed, so a four-arm ladder
    can be a two-point line and no count of labels would say so.
    """
    samples, roles = {}, [pk.ArmRole("stock", "stock", pk.STOCK)]
    samples["stock"] = [64.0] * 5
    nominal_to_actual = {1.0: 1.0, 0.75: 1.0, 0.5: 0.5, 0.25: 0.5}
    for nominal, actual in nominal_to_actual.items():
        for role, prefix, time in ((pk.KNOB, "Q@", 50.0 + 14.0 * actual),
                                   (pk.SCAFFOLD, "Q~", 64.01)):
            label = f"{prefix}{nominal}"
            samples[label] = [time] * 5
            roles.append(pk.ArmRole(label, "Q", role, actual))
    with pytest.raises(RunInvalid, match="round to one size are one setting"):
        pk.reduce_width(samples, roles, resolution_floor=0.05)


def test_a_scaffold_placed_at_other_settings_than_the_knob_refuses():
    samples, roles = _width(scaffold_phis=(1.0, 0.75, 0.5, 0.125))
    with pytest.raises(RunInvalid, match="prices a different arm"):
        pk.reduce_width(samples, roles, resolution_floor=0.05)


def test_a_width_without_exactly_one_stock_arm_refuses():
    samples, roles = _width()
    roles = [r for r in roles if r.role != pk.STOCK]
    del samples["stock"]
    with pytest.raises(RunInvalid, match="exactly one stock arm"):
        pk.reduce_width(samples, roles, resolution_floor=0.05)


def test_a_resolution_floor_of_zero_refuses_the_reduction():
    samples, roles = _width()
    with pytest.raises(RunInvalid, match="needs a positive one"):
        pk.reduce_width(samples, roles, resolution_floor=0.0)


def test_a_rewrite_carries_its_ablated_arm_and_a_retune_may_not():
    samples, roles = _width(candidate="L", ablation=49.5)
    reading = pk.reduce_width(samples, roles, resolution_floor=0.05)["L"]
    assert reading.kind == pk.REWRITE
    assert reading.residue == pytest.approx(0.5)
    samples, roles = _width(candidate="Q", ablation=49.5)
    with pytest.raises(RunInvalid, match="nothing may use"):
        pk.reduce_width(samples, roles, resolution_floor=0.05)


def test_a_candidate_in_no_registry_refuses_rather_than_defaulting_its_kind():
    samples, roles = _width(candidate="Z")
    with pytest.raises(RunInvalid, match="in no knob registry"):
        pk.reduce_width(samples, roles, resolution_floor=0.05)


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


def test_a_family_with_no_reference_arm_takes_its_offset_against_stock():
    """Clause 5 as written, and it is what every candidate but the floor uses."""
    samples, roles = _width(stock=64.0, scaffold_times=[64.1] * 4)
    reading = pk.reduce_width(samples, roles, resolution_floor=0.05)["Q"]
    assert reading.reference_median is None
    assert reading.scaffold_offset == pytest.approx(0.1)


def test_a_family_with_its_own_reference_arm_takes_its_offset_against_that():
    """Amendment 10 clause 45.

    The failure this prevents is not hypothetical: candidate Q's floor runs a
    DENSE matmul where stock runs a quantized one, so `scaffold minus stock`
    is dominated by the two implementations' own speed difference. Here the
    dense family runs at 40 against stock's 64 and its scaffold costs it 0.1,
    so against stock the offset reads -23.9 and refuses at clause 21's 3R,
    while against its own reference it reads the 0.1 the scaffold actually is
    and passes.
    """
    samples, roles = _width(stock=64.0, scaffold_times=[40.1] * 4,
                            reference=40.0, slope=6.0, intercept=30.0)
    reading = pk.reduce_width(samples, roles, resolution_floor=0.05)["Q"]
    assert reading.reference_median == pytest.approx(40.0)
    assert reading.scaffold_offset == pytest.approx(0.1)
    assert reading.blockers() == []

    limit = pk.SCAFFOLD_OFFSET_R_MULTIPLE * 0.05
    assert abs(reading.scaffold_offset) <= limit
    assert abs(40.1 - 64.0) > limit


def test_two_reference_arms_for_one_family_refuse():
    """Clause 5 takes ONE no-dial baseline, and two would leave the reducer
    picking which implementation the offset was measured against."""
    samples, roles = _width(reference=40.0)
    samples["Q=2"] = [41.0] * 5
    roles.append(pk.ArmRole("Q=2", "Q", pk.REFERENCE))
    with pytest.raises(RunInvalid, match="reference arms"):
        pk.reduce_width(samples, roles, resolution_floor=0.05)


def test_a_reference_arm_needs_no_dial_setting():
    """It is a no-dial baseline, so demanding a phi of it would be demanding
    the one thing it is defined by not having."""
    role = pk.ArmRole("Q=", "Q", pk.REFERENCE)
    assert role.phi is None


def test_the_floor_is_a_retune_on_candidate_q_s_own_regions():
    """Clause 18 gives candidate Q `F = d` with no `c_floor`, so its floor
    carries no ablated arm, and clause 19 puts it at the same seams."""
    floor = pk.KNOBS["Qfloor"]
    assert floor.kind == pk.RETUNE
    assert floor.ablate is None
    assert floor.reference is not None
    assert floor.regions == pk.KNOBS["Q"].regions


def test_only_the_floor_family_carries_a_reference_builder():
    """Clause 5 as written takes the offset against stock, and Amendment 10
    changes that for one family and not for the rest."""
    for name, knob in sorted(pk.KNOBS.items()):
        assert (knob.reference is not None) == (name == "Qfloor")
