"""The Day 1 profile's decision arithmetic, checked against its own document.

These rules were written before the profile runs, which is the whole of their
value: a rule that can still be adjusted after seeing the numbers is not a
rule. So the tests here pin the shape of each decision rather than a value the
run happens to produce, and every constant is checked against the section of
docs/research/2026-08-19-metalrunner-sprint1-prereg.md that fixed it.
"""

import pytest

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
