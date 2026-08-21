"""The Day 1 profile's decision arithmetic, checked against its own document.

These rules were written before the profile runs, which is the whole of their
value: a rule that can still be adjusted after seeing the numbers is not a
rule. So the tests here pin the shape of each decision rather than a value the
run happens to produce, and every constant is checked against the section of
docs/research/2026-08-19-metalrunner-sprint1-prereg.md that fixed it.
"""

import statistics

import pytest

import profile_rules
from decode_rules import RunInvalid
from profile_rules import (
    CANDIDATES,
    CELLS,
    COMPILE_RATIO_BAND,
    GAIN_FLOOR,
    KILL_RATIO,
    KILL_SHAPES,
    PRIMARY_CELL,
    RECONCILE_PCT,
    SHAPES,
    Reading,
    collapse_ratio_lo,
    compile_transfer,
    gain,
    kill_q,
    ratio_lo,
    reconciles,
    select_first_operation,
)


def _reading(candidate, share, ratio, footprint=0):
    return Reading(candidate=candidate, share=share, ratio_lo=ratio,
                   footprint_delta=footprint)


# ---------------------------------------------------------------------------
# The registered constants, against section 3.2, 4.3 and 5
# ---------------------------------------------------------------------------
def test_the_cells_and_the_one_that_decides():
    assert set(CELLS) == {"A", "B", "C", "D"}
    assert CELLS["A"] == {"batch": 1, "supervision": "masked"}
    assert CELLS["B"] == {"batch": 4, "supervision": "masked"}
    assert CELLS["C"] == {"batch": 1, "supervision": "all-tokens"}
    assert CELLS["D"] == {"batch": 4, "supervision": "all-tokens"}
    assert PRIMARY_CELL == "B", "the case R18's arithmetic was budgeted against"


def test_the_five_shapes_are_the_model_geometry():
    assert SHAPES == {"S1": (4096, 2560), "S2": (2560, 4096),
                      "S3": (9728, 2560), "S4": (2560, 9728),
                      "S5": (151936, 2560)}
    assert len(SHAPES) == KILL_SHAPES + 1, (
        "the kill rule needs 4 of 5; a sixth shape changes what it means")


# ---------------------------------------------------------------------------
# The gain formula, section 4.1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("share,ratio,expected", [
    (0.0, 2.0, 1.0),      # accelerating nothing gains nothing
    (1.0, 2.0, 2.0),      # accelerating everything gains the ratio
    (0.5, 2.0, 4 / 3),    # half the step, twice as fast
    (0.5, 1.0, 1.0),      # a ratio of one is no speedup at any share
])
def test_gain_is_amdahls_relation(share, ratio, expected):
    assert gain(share, ratio) == pytest.approx(expected)


def test_gain_below_one_when_the_replacement_is_slower():
    """A floor above stock's own cost is a real reading, not an error: it says
    the candidate has nothing to win, and the rule must be able to say so."""
    assert gain(0.5, 0.5) < 1.0


@pytest.mark.parametrize("share,ratio", [
    (-0.1, 2.0), (1.1, 2.0),   # not a fraction of the step
    (0.5, 0.0), (0.5, -1.0),   # not a ratio
])
def test_gain_refuses_inputs_the_formula_does_not_describe(share, ratio):
    with pytest.raises(RunInvalid):
        gain(share, ratio)


# ---------------------------------------------------------------------------
# The credited ratio, section 4.2
# ---------------------------------------------------------------------------
def test_the_credited_ratio_is_the_worst_pairing_the_samples_permit():
    """Not the median, and not the best: the smallest numerator over the
    largest denominator, so a candidate is credited with the least its own
    measurements support."""
    assert ratio_lo([10.0, 11.0, 12.0], [4.0, 5.0, 6.0]) == pytest.approx(10 / 6)


def test_a_floor_of_zero_cannot_bound_a_ratio():
    with pytest.raises(RunInvalid):
        ratio_lo([1.0], [0.0])
    with pytest.raises(RunInvalid):
        ratio_lo([], [1.0])


# ---------------------------------------------------------------------------
# The two guards, section 3.3 and Amendment 4
# ---------------------------------------------------------------------------
def test_a_decomposition_that_accounts_for_the_step_reconciles():
    got = reconciles({"loss": 30.0, "attention": 50.0}, remainder=20.0,
                     step_total=100.0)
    assert got["ok"] and got["gap_pct"] == pytest.approx(0.0)


def test_a_decomposition_that_misses_the_step_by_more_than_two_percent_fails():
    got = reconciles({"loss": 30.0}, remainder=20.0, step_total=100.0)
    assert not got["ok"] and got["gap_pct"] == pytest.approx(50.0)
    assert got["limit_pct"] == RECONCILE_PCT


def test_the_reconciliation_boundary_is_inclusive():
    """A gap exactly at the limit reconciles; only past it is rejected."""
    assert reconciles({"a": 98.0}, remainder=0.0, step_total=100.0)["ok"]
    assert not reconciles({"a": 97.9}, remainder=0.0, step_total=100.0)["ok"]


def test_the_compile_ratio_is_reported_and_never_applied():
    got = compile_transfer(uncompiled_total=105.0, compiled_total=100.0)
    assert got["ok"] and got["ratio"] == pytest.approx(1.05)
    assert got["band"] == COMPILE_RATIO_BAND


@pytest.mark.parametrize("uncompiled", [85.0, 130.0])
def test_a_step_that_moved_too_far_under_compile_rejects_the_profile(uncompiled):
    assert not compile_transfer(uncompiled, 100.0)["ok"]


# ---------------------------------------------------------------------------
# The kill rule, section 5
# ---------------------------------------------------------------------------
def _ceilings(**overrides):
    ratios = {name: 2.0 for name in SHAPES}
    ratios.update(overrides)
    return ratios


def test_q_dies_when_stock_is_already_at_the_ceiling_almost_everywhere():
    got = kill_q(_ceilings(S1=1.02, S2=1.05, S3=1.09, S4=1.10))
    assert got["killed"] and got["at_ceiling"] == ["S1", "S2", "S3", "S4"]


def test_three_shapes_at_the_ceiling_is_not_enough():
    assert not kill_q(_ceilings(S1=1.02, S2=1.05, S3=1.09))["killed"]


def test_the_kill_line_is_inclusive_and_a_hair_past_it_is_headroom():
    assert kill_q(_ceilings(S1=1.1, S2=1.1, S3=1.1, S4=1.1))["killed"]
    assert not kill_q(_ceilings(S1=1.1, S2=1.1, S3=1.1, S4=1.1001))["killed"]


def test_the_kill_rule_reads_every_shape_or_refuses():
    with pytest.raises(RunInvalid):
        kill_q({"S1": 1.0, "S2": 1.0})


# ---------------------------------------------------------------------------
# The selection rule, section 4.3
# ---------------------------------------------------------------------------
def test_the_largest_gain_selects():
    got = select_first_operation([
        _reading("L", share=0.30, ratio=2.0),
        _reading("A", share=0.10, ratio=2.0),
    ])
    assert got["selected"] == "L" and got["verdict"] == "SELECTED"
    assert [row["candidate"] for row in got["ranked"]] == ["L", "A"]


def test_a_tie_goes_to_the_smaller_footprint_delta():
    """Within two points of gain is a tie, and the first tie-break is the
    measured peak-footprint delta, not the larger number."""
    got = select_first_operation([
        _reading("L", share=0.3000, ratio=2.0, footprint=900),
        _reading("A", share=0.2999, ratio=2.0, footprint=100),
    ])
    assert got["selected"] == "A", "the hair-larger gain does not win a tie"
    assert got["tied_with"] == ["L"]


def test_a_remaining_tie_goes_to_the_earlier_row_in_the_table():
    got = select_first_operation([
        _reading("Q", share=0.30, ratio=2.0, footprint=100),
        _reading("A", share=0.30, ratio=2.0, footprint=100),
    ])
    assert got["selected"] == "A", f"table order is {CANDIDATES}"


def test_a_weak_profile_picks_nothing_and_keeps_the_top_two():
    """Section 4.3's registered branch: below R10's floor the sprint records
    the reading rather than making a hopeful pick."""
    got = select_first_operation([
        _reading("L", share=0.05, ratio=1.5),
        _reading("A", share=0.04, ratio=1.5),
        _reading("Q", share=0.01, ratio=1.2),
    ])
    assert got["selected"] is None
    assert got["verdict"] == "NO SINGLE OPERATION REACHES THE FLOOR"
    assert got["keep"] == ["L", "A"], "the top two, in gain order"
    assert all(row["gain"] < GAIN_FLOOR for row in got["ranked"])


def test_the_floor_boundary_selects_rather_than_refuses():
    """gain == 1.10 is not below the floor. f and r chosen so the formula
    lands exactly on it: 1/(1 - f(1 - 1/r)) = 1.1 at f = 1, r = 1.1."""
    got = select_first_operation([_reading("L", share=1.0, ratio=GAIN_FLOOR)])
    assert got["ranked"][0]["gain"] == pytest.approx(GAIN_FLOOR)
    assert got["selected"] == "L"


def test_the_footprint_tie_break_cannot_select_below_the_floor():
    """Amendment 5 clause 24: the shipping floor applies to the SELECTED
    candidate, not to the largest gain. Reproduced before the fix: a leader at
    gain 1.105 pulls a 1.095 candidate into the tie band, and the smaller
    footprint then hands SELECTED to a candidate below 1.10."""
    got = select_first_operation([
        _reading("L", share=2 * (1 - 1 / 1.105), ratio=2.0, footprint=900),
        _reading("A", share=2 * (1 - 1 / 1.095), ratio=2.0, footprint=100),
    ])
    assert got["selected"] == "L", "a below-floor candidate must never ship"
    assert got["tied_with"] == [], "below the floor is not tied, it is out"
    assert [row["candidate"] for row in got["ranked"]] == ["L", "A"], \
        "the below-floor candidate stays in the ranked evidence"


def test_table_order_cannot_select_below_the_floor_either():
    """The same hole through the other tie-break: an unmeasured footprint
    sends the band to table order, and table order reads L before Q."""
    got = select_first_operation([
        _reading("Q", share=2 * (1 - 1 / 1.105), ratio=2.0, footprint=100),
        _reading("L", share=2 * (1 - 1 / 1.095), ratio=2.0, footprint=None),
    ])
    assert got["selected"] == "Q", "a below-floor candidate must never ship"
    assert got["tied_with"] == []
    assert [row["candidate"] for row in got["ranked"]] == ["Q", "L"]


def test_the_rule_refuses_a_candidate_it_never_registered():
    with pytest.raises(RunInvalid):
        select_first_operation([_reading("Z", share=0.3, ratio=2.0)])


def test_the_rule_refuses_an_empty_field():
    with pytest.raises(RunInvalid):
        select_first_operation([])


def test_the_rule_refuses_two_readings_of_one_candidate():
    """A candidate has one share at the primary cell and one credited ratio.
    Given two, the rule would silently rank the better of them and report a
    table naming the same candidate twice, which reads as a comparison."""
    with pytest.raises(RunInvalid, match="more than one reading"):
        select_first_operation([_reading("L", share=0.5, ratio=2.0),
                                _reading("L", share=0.1, ratio=1.1)])


def test_one_reading_each_is_still_accepted():
    ruling = select_first_operation([_reading("L", share=0.5, ratio=2.0),
                                     _reading("A", share=0.1, ratio=1.1)])
    assert ruling["selected"] == "L"


# ---------------------------------------------------------------------------
# Collapsing many shapes into one credited ratio (Amendment 5)
# ---------------------------------------------------------------------------
def _side(count, numerator, denominator):
    return {"count": count, "numerator": numerator, "denominator": denominator}


_NONE = {"count": 0, "numerator": [], "denominator": []}


def _shape(forward=None, backward=None):
    """Both directions always named, because the rule requires it.

    A direction a shape never ran in is written as a count of zero. Leaving it
    out means something different - that it was not measured - and the rule
    refuses that rather than weighting it as an empty measurement.
    """
    return {"forward": forward if forward is not None else dict(_NONE),
            "backward": backward if backward is not None else dict(_NONE)}


def test_the_collapse_weights_each_shape_by_how_often_it_runs():
    """A shape that runs 252 times per step and one that runs once are not
    equal evidence about the operation as a whole."""
    collapsed = collapse_ratio_lo({
        "S1": _shape(forward=_side(252, [2.0], [1.0])),
        "S5": _shape(forward=_side(1, [10.0], [10.0])),
    })
    assert collapsed["ratio_lo"] == pytest.approx((252 * 2.0 + 10.0)
                                                  / (252 * 1.0 + 10.0))
    assert collapsed["ratio_lo"] > 1.9


def test_forward_and_backward_are_weighted_by_their_own_counts():
    """Measured inside a real LoRA step: the quantized projections ran 196
    times forward and 25 times backward, because the blocks below the first
    adapted one have no backward. One count cannot describe both, and using
    the forward count for backward work is what stops the result being a
    bound on anything."""
    collapsed = collapse_ratio_lo({
        "S1": _shape(forward=_side(196, [2.0], [1.0]),
                     backward=_side(25, [4.0], [1.0])),
    })
    expected = (196 * 2.0 + 25 * 4.0) / (196 * 1.0 + 25 * 1.0)
    assert collapsed["ratio_lo"] == pytest.approx(expected)


def test_a_direction_with_no_calls_contributes_nothing():
    """A region absent from the backward is the normal case under LoRA, not
    an error, and it must not drag the ratio toward anything."""
    collapsed = collapse_ratio_lo({
        "S1": _shape(forward=_side(10, [2.0], [1.0]),
                     backward=_side(0, [], [])),
    })
    assert collapsed["ratio_lo"] == pytest.approx(2.0)


def test_the_collapse_keeps_the_worst_pairing_at_the_aggregate():
    """Smallest numerator over largest denominator, shape by shape, so no
    shape's optimistic sample can be paired with another's pessimistic one."""
    collapsed = collapse_ratio_lo({
        "S1": _shape(forward=_side(1, [3.0, 4.0, 5.0], [1.0, 2.0])),
        "S2": _shape(forward=_side(1, [9.0, 10.0], [4.0, 5.0])),
    })
    assert collapsed["ratio_lo"] == pytest.approx((3.0 + 9.0) / (2.0 + 5.0))


def test_a_single_shape_and_direction_collapses_to_the_plain_ratio():
    """The collapse must not disagree with `ratio_lo` where both apply."""
    numerator, denominator = [3.0, 4.0], [1.0, 2.0]
    collapsed = collapse_ratio_lo(
        {"S1": _shape(forward=_side(1, numerator, denominator))})
    assert collapsed["ratio_lo"] == pytest.approx(ratio_lo(numerator,
                                                           denominator))


def test_the_collapse_reports_what_each_shape_and_direction_contributed():
    """Which shape carries the ratio is the first thing a reader asks, and
    the answer decides where a kernel would actually be aimed."""
    collapsed = collapse_ratio_lo({
        "S1": _shape(forward=_side(2, [3.0], [1.0]),
                     backward=_side(1, [5.0], [1.0])),
    })
    assert collapsed["shapes"]["S1"]["forward"] == {
        "count": 2, "numerator": 6.0, "denominator": 2.0}
    assert collapsed["shapes"]["S1"]["backward"]["count"] == 1


@pytest.mark.parametrize("count", [-1, 1.5, True, None])
def test_a_count_that_is_not_a_count_refuses(count):
    with pytest.raises(RunInvalid, match="not a count"):
        collapse_ratio_lo({"S1": _shape(forward=_side(count, [1.0], [1.0]))})


@pytest.mark.parametrize("numerator,denominator", [([], [1.0]), ([1.0], [])])
def test_a_direction_measured_on_one_side_only_refuses(numerator, denominator):
    with pytest.raises(RunInvalid, match="one side"):
        collapse_ratio_lo(
            {"S1": _shape(forward=_side(1, numerator, denominator))})


def test_a_direction_with_a_zero_floor_refuses():
    with pytest.raises(RunInvalid, match="floor of"):
        collapse_ratio_lo({"S1": _shape(forward=_side(1, [1.0], [0.0]))})


def test_a_shape_naming_no_direction_refuses():
    """Silence about both directions is not a shape that took no time."""
    with pytest.raises(RunInvalid, match="names no"):
        collapse_ratio_lo({"S1": {}})


def test_a_direction_left_out_refuses_rather_than_weighing_as_empty():
    """Absent and zero are different claims. A shape that never ran backward
    says so with a count of zero; a shape whose backward was never measured is
    a hole in the floor, and weighting it as an empty measurement would hide
    the hole behind a number."""
    with pytest.raises(RunInvalid, match="names no backward"):
        collapse_ratio_lo({"S1": {"forward": _side(1, [2.0], [1.0])}})


def test_a_direction_that_never_ran_is_written_as_a_count_of_zero():
    collapsed = collapse_ratio_lo({
        "S1": {"forward": _side(1, [2.0], [1.0]),
               "backward": _side(0, [], [])}})
    assert collapsed["shapes"]["S1"]["backward"]["count"] == 0
    assert collapsed["ratio_lo"] == pytest.approx(2.0)


def test_a_shape_carrying_an_unknown_key_refuses():
    """A caller passing the old single-count shape must fail loudly rather
    than have one direction silently read as the whole operation."""
    with pytest.raises(RunInvalid, match="weighted per direction"):
        collapse_ratio_lo({"S1": {"count": 5, "numerator": [1.0],
                                  "denominator": [1.0]}})


def test_the_collapse_refuses_an_empty_field():
    with pytest.raises(RunInvalid, match="at least one shape"):
        collapse_ratio_lo({})


def test_the_kill_rule_counts_only_registered_shapes():
    """Counting an unregistered key would let shapes nobody registered reach
    the kill threshold, and the verdict would report "4 of 5" about a set that
    was never five."""
    ratios = {name: 5.0 for name in SHAPES}
    ratios.update({f"invented{i}": 1.0 for i in range(4)})
    with pytest.raises(RunInvalid, match="not registered"):
        kill_q(ratios)


# ---------------------------------------------------------------------------
# The footprint tie-break needs every tied candidate measured (Amendment 5)
# ---------------------------------------------------------------------------
def test_an_unmeasured_footprint_sends_the_whole_band_to_table_order():
    """Ranking a measured delta against an unmeasured one would decide the
    sprint on which candidate happened to get a number."""
    ruling = select_first_operation([
        Reading("A", 0.50, 2.0, footprint_delta=None),
        Reading("L", 0.50, 2.0, footprint_delta=1),
    ])
    assert ruling["selected"] == "L"          # L is the earlier table row
    assert ruling["footprint_measured"] is False
    assert "table order" in ruling["reason"]


def test_a_fully_measured_band_still_uses_the_footprint():
    ruling = select_first_operation([
        Reading("L", 0.50, 2.0, footprint_delta=9),
        Reading("A", 0.50, 2.0, footprint_delta=1),
    ])
    assert ruling["selected"] == "A"
    assert ruling["footprint_measured"] is True
    assert "peak-footprint delta" in ruling["reason"]


def test_an_unmeasured_footprint_outside_the_band_does_not_reach_the_rule():
    """Only the tied band is tie-broken, so a distant candidate with no
    footprint number cannot drag the winner to table order."""
    ruling = select_first_operation([
        Reading("L", 0.50, 4.0, footprint_delta=9),
        Reading("A", 0.50, 4.0, footprint_delta=1),
        Reading("Q", 0.01, 1.01, footprint_delta=None),
    ])
    assert ruling["selected"] == "A"
    assert ruling["footprint_measured"] is True


def test_the_footprint_defaults_to_unmeasured():
    """A reading that says nothing about footprint must not read as zero,
    which would silently win every tie-break."""
    assert Reading("A", 0.5, 2.0).footprint_delta is None


# ---------------------------------------------------------------------------
# `R`, the calibrated demand `C`, and what Amendment 6 denominates in each
#
# Committed clause 21's wording admits two readings and clause 33 fixes the
# of-medians one, because nothing in these rules consumes a single round.
# Committed clause 16's `2 * R` is a false-separation RATE rather than a
# resolution test, and clause 33 replaces the judgement with a registered
# rate and a measured critical value.
# ---------------------------------------------------------------------------
def test_the_null_contrast_reduces_the_arms_the_way_the_rules_do():
    """The two readings of clause 21's sentence differ, and this is the case
    that separates them. The rules consume medians, so the null does too."""
    a, b = [0.0, 0.0, 100.0], [0.0, 100.0, 100.0]
    assert profile_rules.null_contrast(a, b) == 100.0
    rejected = statistics.median([abs(x - y) for x, y in zip(a, b)])
    assert rejected == 0.0, "the paired reading is the one NOT taken"


def test_arms_that_agree_every_round_have_a_null_of_zero():
    """The withdrawn rationale claimed of-medians could be large here, which
    is arithmetically impossible."""
    a = [10.0, 11.0, 12.0]
    assert profile_rules.null_contrast(a, list(a)) == 0.0


def test_a_null_contrast_pairs_its_arms_and_refuses_otherwise():
    with pytest.raises(RunInvalid, match="same rounds"):
        profile_rules.null_contrast([1.0, 2.0], [1.0])
    with pytest.raises(RunInvalid, match="at least one round"):
        profile_rules.null_contrast([], [])


def test_r_is_a_median_over_blocks_because_one_block_cannot_estimate_it():
    blocks = [0.5, 0.9, 1.2, 0.7, 1.8]
    assert profile_rules.resolution(blocks) == pytest.approx(0.9)
    with pytest.raises(RunInvalid, match="at least one"):
        profile_rules.resolution([])


def test_the_critical_value_is_an_order_statistic_not_a_multiple_of_r():
    """`C` holds a future exchangeable null exceedance at or below alpha and
    assumes nothing about the null's shape."""
    blocks = [0.5, 0.9, 1.2, 0.7, 1.8, 0.4, 1.1, 0.6, 2.4, 0.8]
    got = profile_rules.critical_value(blocks, alpha=0.05)
    assert got == 2.4, "at m=10 and alpha=0.05 the demand is the largest block"
    looser = profile_rules.critical_value(blocks, alpha=0.5)
    assert looser < got, "a laxer rate demands less"
    assert profile_rules.critical_value(blocks) == got, "alpha defaults to 0.05"


def test_the_critical_value_refuses_a_rate_that_is_not_a_rate():
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(RunInvalid, match="not a rate"):
            profile_rules.critical_value([1.0, 2.0], alpha=bad)
    with pytest.raises(RunInvalid, match="at least one block"):
        profile_rules.critical_value([])


def test_the_attainable_rate_bounds_what_a_block_count_can_promise():
    """With m blocks the smallest holdable rate is 1/(m+1), so a registered
    alpha of 0.05 needs at least 19 blocks to be attainable at all."""
    assert profile_rules.attainable_rate(19) == pytest.approx(0.05)
    assert profile_rules.attainable_rate(9) > profile_rules.ALPHA
    assert profile_rules.attainable_rate(39) < profile_rules.ALPHA
    with pytest.raises(RunInvalid):
        profile_rules.attainable_rate(0)


def test_a_resolution_floor_of_zero_is_refused():
    """A floor of zero claims the machine resolves any difference at all,
    which would let every positive slope clear 10R at once."""
    assert profile_rules.resolution_admissible(0.155) == []
    assert profile_rules.resolution_admissible(0.0)
    assert profile_rules.resolution_admissible(-1.0)
    assert profile_rules.resolution_admissible(float("nan"))
    assert profile_rules.resolution_admissible(float("inf"))


def test_the_effective_shipping_floor_sits_above_the_registered_one():
    """Gating the floor on resolution raises the gain a candidate must
    actually beat above the registered 1.10, by an amount set by C over T."""
    got = profile_rules.smallest_shippable_gain(100.0, 0.31)
    assert got > profile_rules.GAIN_FLOOR
    assert profile_rules.smallest_shippable_gain(50.0, 2.0) > got, \
        "a coarser demand, or a shorter step, raises it further"


def test_a_coarse_enough_demand_admits_no_gain_at_all():
    """Once C reaches T/1.10 nothing can ship, and step 10 reports that
    before the binding window is spent rather than after."""
    assert profile_rules.floor_is_clearable(10.0, 9.0)
    assert not profile_rules.floor_is_clearable(10.0, 9.2)
    assert profile_rules.smallest_shippable_gain(10.0, 9.2) is None
    edge = 10.0 / profile_rules.GAIN_FLOOR
    assert not profile_rules.floor_is_clearable(10.0, edge)


# ---------------------------------------------------------------------------
# Clause 31 v4: bound the ACTION, not the gap
#
# Three earlier versions of clause 31 compared a converted gap against a
# scalar demand, which assumes each measurement's uncertainty reaches the
# compared quantity one for one. It does not: the share takes the median
# per-round numerator while `ratio_lo` takes smallest-over-largest, so a
# floor movement is amplified by M/N.
# ---------------------------------------------------------------------------
def test_the_saving_reproduces_the_amplification_that_forced_the_rewrite():
    """dK/dF = -M/N, so the predicted step moves by M/N times a floor move."""
    T, M, N = 100.0, 50.0, 10.0
    before = T - profile_rules.credited_saving(M, N, 3.00)
    after = T - profile_rules.credited_saving(M, N, 3.11)
    assert after - before == pytest.approx(0.55, abs=1e-9)
    assert (after - before) / 0.11 == pytest.approx(M / N, abs=1e-9)


def test_the_saving_and_the_registered_gain_agree():
    """`gain = T/(T - K)` must equal the committed formula on the same
    reduction, or the saving is a different quantity wearing its name."""
    T, M, N, F = 100.0, 50.0, 50.0, 15.0
    K = profile_rules.credited_saving(M, N, F)
    assert T / (T - K) == pytest.approx(gain(M / T, N / F))


def test_the_saving_refuses_a_denominator_outside_its_domain():
    with pytest.raises(RunInvalid, match="not inside"):
        profile_rules.credited_saving(50.0, 10.0, 10.0)
    with pytest.raises(RunInvalid, match="not inside"):
        profile_rules.credited_saving(50.0, 10.0, 0.0)
    with pytest.raises(RunInvalid, match="divides by it"):
        profile_rules.credited_saving(50.0, 0.0, 1.0)


def test_the_saving_bounds_contain_every_point_of_their_box():
    lo, hi = profile_rules.saving_bounds((49.0, 51.0), (9.8, 10.2), (2.9, 3.1))
    for m in (49.0, 50.0, 51.0):
        for n in (9.8, 10.0, 10.2):
            for f in (2.9, 3.0, 3.1):
                assert lo - 1e-9 <= profile_rules.credited_saving(m, n, f) <= hi + 1e-9
    assert lo < hi


def test_the_saving_bounds_refuse_a_box_that_leaves_the_domain():
    with pytest.raises(RunInvalid, match="everywhere in the box"):
        profile_rules.saving_bounds((49.0, 51.0), (3.0, 4.0), (2.9, 5.0))
    with pytest.raises(RunInvalid, match="inverted"):
        profile_rules.saving_bounds((51.0, 49.0), (9.8, 10.2), (2.9, 3.1))


def test_shipping_needs_the_saving_to_beat_a_gain_of_1_10_everywhere():
    """A gain of 1.10 IS a saving of T/11, so the margin is K - T/11."""
    assert profile_rules.ships_above_floor(saving_low=10.0, step_total_high=100.0)
    assert not profile_rules.ships_above_floor(saving_low=9.0, step_total_high=100.0)
    edge = 100.0 / 11.0
    assert not profile_rules.ships_above_floor(edge, 100.0), \
        "exactly at the floor is not a supported action"


def test_a_pair_separates_only_when_its_savings_do_not_overlap():
    assert profile_rules.separable(saving_low=20.0, other_saving_high=19.0)
    assert not profile_rules.separable(saving_low=20.0, other_saving_high=20.5)


def test_the_band_expression_tracks_the_two_point_gain_test():
    """`E_a - E_l - tau(1-E_a)(1-E_l)` has the same sign as `S_a - S_l - tau`
    with `S = 1/(1-E)`, which is what lets the band be tested on savings."""
    for ea, el in ((0.30, 0.10), (0.20, 0.19), (0.50, 0.49), (-0.10, -0.30)):
        expr = profile_rules.outside_band(ea, el)
        direct = (1 / (1 - ea)) - (1 / (1 - el)) - profile_rules.TIE_GAIN > 0
        assert expr == direct, (ea, el)


def test_a_kill_shape_counts_only_on_an_operand_margin():
    """Converting the ratio through a shape's stock slope would amplify the
    floor's uncertainty; the operand margin does not."""
    assert profile_rules.shape_counts_toward_kill(
        denominator_low=1.0, numerator_high=1.05)
    assert not profile_rules.shape_counts_toward_kill(
        denominator_low=1.0, numerator_high=1.15)
    assert not profile_rules.shape_counts_toward_kill(1.0, 1.10), \
        "exactly at the kill ratio is not a supported action"
