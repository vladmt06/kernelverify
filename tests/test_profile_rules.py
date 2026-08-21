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
    CandidateInput,
    CeilingEntry,
    GAIN_FLOOR,
    INCOMPLETE,
    Interval,
    KILL_RATIO,
    KILL_SHAPES,
    MISSING_RATIO,
    MISSING_SHARE,
    NO_SELECTION,
    PRIMARY_CELL,
    SELECTED,
    SHAPES,
    ScoredEntry,
    ShapeAtWidth,
    WIDTH_ORDER,
    collapse_ratio_lo,
    gain,
    kill_q,
    largest_round_ratio,
    ratio_lo,
    select_first_operation,
)


def _iv(value, spread=0.0):
    return Interval(value=value, low=value - spread, high=value + spread)


def _scored(name, savings, *, total=100.0, spread=0.5, **over):
    """One candidate L or Q, with a saving at each registered width.

    The default spread is small against the savings below, so a test that
    wants a comparison to fail the certified margin widens it deliberately
    rather than relying on the numbers happening to overlap.
    """
    return CandidateInput(
        candidate=name,
        entries=tuple(ScoredEntry(width=width, step_total=_iv(total, spread),
                                  saving=_iv(saving, spread))
                      for width, saving in savings.items()), **over)


def _ceiling(shares, *, total=100.0, spread=0.001):
    return CandidateInput(
        candidate="A",
        entries=tuple(CeilingEntry(width=width, step_total=_iv(total, 0.5),
                                   share=_iv(share, spread),
                                   attributed=_iv(share * total,
                                                  spread * total))
                      for width, share in shares.items()))


def _field(**over):
    """The three registered candidates, all present and all scoreable."""
    field = {"L": _scored("L", {"short": 25.0, "long": 24.0}),
             "Q": _scored("Q", {"short": 12.0, "long": 11.0}),
             "A": _ceiling({"long": 0.05})}
    field.update(over)
    return list(field.values())


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


def test_the_six_shapes_are_the_model_geometry():
    """Amendment 5 clause 7: section 3.2 listed five and the model has six.
    The key and value projections are counted in candidate Q's share, so a
    floor over five would credit a ratio measured on part of the operation."""
    assert SHAPES == {"S1": (4096, 2560), "S2": (2560, 4096),
                      "S3": (9728, 2560), "S4": (2560, 9728),
                      "S5": (151936, 2560), "S6": (1024, 2560)}


def test_the_kill_count_is_all_but_one_and_moves_with_the_shape_count():
    """Inheriting the number 4 once S6 exists would have loosened the rule
    from four-fifths to two-thirds without saying so."""
    assert KILL_SHAPES == len(SHAPES) - 1 == 5


def test_the_certified_and_reported_sites_are_stated_rather_than_inferred():
    assert len(profile_rules.CERTIFIED_SITES) == 3
    assert len(profile_rules.REPORTED_SITES) == 4
    assert not set(profile_rules.CERTIFIED_SITES) & set(
        profile_rules.REPORTED_SITES)


def test_the_rules_amendment_five_voided_are_gone_and_not_merely_unused():
    """A voided rule that still answers is a rule someone can call and
    believe, so the reconciliation, the compile transfer and the
    instrument-cost band are removed rather than deprecated."""
    for name in ("reconciles", "compile_transfer", "instrument_cost",
                 "INSTRUMENT_COST_BAND", "RECONCILE_PCT",
                 "COMPILE_RATIO_BAND", "Reading"):
        assert not hasattr(profile_rules, name), name


# ---------------------------------------------------------------------------
# The gain formula, section 4.1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("share,ratio,expected", [
    (0.0, 2.0, 1.0),      # accelerating nothing gains nothing
      # accelerating everything gains the ratio
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
# The kill rule, section 5 over Amendment 5's six shapes, REPORTED
# ---------------------------------------------------------------------------
def _kill_shape(ratio):
    """One shape at one width, with operands that place it either side of the
    kill line at the same place the reported ratio does."""
    return ShapeAtWidth(ratio=ratio, numerator_high=ratio,
                        denominator_low=1.0)


def _kill(**over):
    """Every registered shape at both widths, clear of the ceiling by default."""
    field = {name: {width: _kill_shape(2.0) for width in WIDTH_ORDER}
             for name in SHAPES}
    for name, value in over.items():
        if isinstance(value, dict):
            field[name] = {width: _kill_shape(one)
                           for width, one in value.items()}
        else:
            field[name] = {width: _kill_shape(value) for width in WIDTH_ORDER}
    return field


def test_q_is_at_the_ceiling_at_all_but_one_of_six_shapes():
    got = kill_q(_kill(S1=1.02, S2=1.05, S3=1.02, S4=1.05, S5=1.02))
    assert got["killed"]
    assert got["at_ceiling"] == ["S1", "S2", "S3", "S4", "S5"]


def test_four_shapes_at_the_ceiling_is_not_enough_over_six():
    assert not kill_q(_kill(S1=1.02, S2=1.05, S3=1.02, S4=1.05))["killed"]


def test_the_certified_kill_boundary_is_strict_where_the_bare_ratio_was_not():
    """Clause 31 turns an inclusive threshold into a strict one: an action
    supported exactly at the line is supported at no margin at all."""
    assert not profile_rules.shape_counts_toward_kill(1.0, KILL_RATIO)
    at_line = _kill(**{name: KILL_RATIO for name in list(SHAPES)[:5]})
    assert not kill_q(at_line)["killed"]


def test_headroom_at_either_width_keeps_the_candidate_alive():
    """Clause 8's two-width reduction: a shape counts only where stock is
    within the ceiling at BOTH widths."""
    field = _kill(S1=1.02, S2=1.02, S3=1.02, S4=1.02)
    field["S5"] = {"short": _kill_shape(1.02), "long": _kill_shape(2.0)}
    got = kill_q(field)
    assert not got["killed"]
    assert got["shapes"]["S5"]["widths"]["short"]["counts"]
    assert not got["shapes"]["S5"]["widths"]["long"]["counts"]


def test_the_kill_verdict_says_in_writing_that_it_gates_nothing():
    got = kill_q(_kill(S1=1.02, S2=1.02, S3=1.02, S4=1.02, S5=1.02))
    assert got["certified"] is False
    assert "removes no candidate" in got["gates_nothing"]


def test_the_kill_rule_reads_every_registered_shape_or_refuses():
    field = _kill()
    del field["S6"]
    with pytest.raises(RunInvalid, match="missing"):
        kill_q(field)


def test_the_kill_rule_reads_both_widths_at_every_shape_or_refuses():
    field = _kill()
    field["S3"] = {"short": _kill_shape(1.02)}
    with pytest.raises(RunInvalid, match="both registered widths"):
        kill_q(field)


def test_a_shape_nobody_registered_cannot_reach_the_threshold():
    field = _kill()
    field["S7"] = {width: _kill_shape(1.0) for width in WIDTH_ORDER}
    with pytest.raises(RunInvalid, match="not registered"):
        kill_q(field)


def test_the_ratio_a_shape_contributes_is_the_largest_of_its_rounds():
    """The largest is the one furthest from the ceiling and so the least
    likely to kill, which is the conservative direction for a rule whose
    effect is to remove a candidate."""
    assert largest_round_ratio([1.05, 1.20, 1.02]) == 1.20
    with pytest.raises(RunInvalid, match="at least one round"):
        largest_round_ratio([])
    with pytest.raises(RunInvalid, match="not finite"):
        largest_round_ratio([1.0, float("nan")])


# ---------------------------------------------------------------------------
# The selection rule, section 4.3 under Amendments 6 and 7
# ---------------------------------------------------------------------------
def test_the_certified_winner_is_selected_and_says_what_it_rested_on():
    ruling = select_first_operation(_field())
    assert ruling["verdict"] == SELECTED
    assert ruling["selected"] == "L"
    assert ruling["band"] == ["L"]
    assert set(ruling["certified_sites"]) == set(profile_rules.CERTIFIED_SITES)


def test_the_score_is_the_candidate_s_gain_at_its_WORSE_width():
    """Clause 14. A candidate strong at one width and weak at the other is
    scored on the weak one, so a winner has to win where it is worst."""
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 40.0, "long": 12.0})))
    row, = [one for one in ruling["scored"] if one["candidate"] == "L"]
    assert row["worst_width"] == "long"
    assert row["score"] == pytest.approx(1 / (1 - 0.12))


def test_the_floor_is_certified_over_the_box_and_not_at_the_measured_value():
    """A gain of 1.10 IS a saving of T/11, and clause 31 supports shipping
    only where the saving beats it everywhere the measurements admit."""
    tight = _scored("L", {"short": 9.5, "long": 9.5}, spread=0.2)
    wide = _scored("L", {"short": 9.5, "long": 9.5}, spread=2.0)
    assert select_first_operation(
        _field(L=tight, Q=_scored("Q", {"short": 4.0, "long": 4.0}))
    )["verdict"] == SELECTED
    assert select_first_operation(
        _field(L=wide, Q=_scored("Q", {"short": 4.0, "long": 4.0}))
    )["verdict"] == NO_SELECTION


def test_a_candidate_above_the_floor_at_one_width_only_does_not_ship():
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 25.0, "long": 5.0}),
        Q=_scored("Q", {"short": 4.0, "long": 4.0})))
    assert ruling["verdict"] == NO_SELECTION


def test_every_score_below_the_floor_keeps_an_unordered_set_not_a_build_order():
    """Clause 34 REVERSES the committed kept list. It used to be the top two
    in score order, and section 4.3 makes that list the Day 2 build order, so
    a quantity that certifies nothing arrived somewhere as an instruction."""
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 4.0, "long": 4.0}),
        Q=_scored("Q", {"short": 6.0, "long": 6.0})))
    assert ruling["verdict"] == NO_SELECTION
    assert ruling["keep"] == ["L", "Q"], "section 4.2's table order"
    assert ruling["score_order"] == ["Q", "L"], "evidence, not a build order"
    assert "NOT a build order" in ruling["keep_is_unordered"]


def test_no_scores_at_all_is_an_answer_rather_than_an_absence():
    ruling = select_first_operation(_field(
        L=CandidateInput("L", absence=MISSING_RATIO),
        Q=CandidateInput("Q", absence=MISSING_RATIO)))
    assert ruling["verdict"] == NO_SELECTION
    assert ruling["keep"] == []
    assert ruling["absences"] == {"L": MISSING_RATIO, "Q": MISSING_RATIO}


@pytest.mark.parametrize("name", ["L", "Q"])
def test_a_certified_candidate_missing_a_share_makes_the_profile_incomplete(
        name):
    """Clause 38's guard runs before the table: a required measurement does
    not exist and no remaining score can substitute for it."""
    ruling = select_first_operation(_field(
        **{name: CandidateInput(name, absence=MISSING_SHARE)}))
    assert ruling["verdict"] == INCOMPLETE
    assert ruling["incomplete"] == [name]
    assert ruling["selected"] is None


def test_a_missing_ratio_leaves_a_candidate_excluded_and_continuable():
    ruling = select_first_operation(_field(
        L=CandidateInput("L", absence=MISSING_RATIO)))
    assert ruling["verdict"] == SELECTED
    assert ruling["selected"] == "Q"
    assert ruling["absences"] == {"L": MISSING_RATIO}


def test_a_killed_q_is_a_recorded_flag_and_removes_nothing():
    """Amendment 7 clause 36 demotes the ACTION and keeps the MEASUREMENT."""
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 12.0, "long": 11.0}),
        Q=_scored("Q", {"short": 25.0, "long": 24.0}, killed=True)))
    assert ruling["verdict"] == SELECTED
    assert ruling["selected"] == "Q"
    assert ruling["kill"] == {"L": False, "Q": True}


def test_a_kill_flag_on_a_candidate_the_rule_does_not_kill_refuses():
    """Clause 23's rule reaches candidate Q alone, so a killed candidate L is
    not a state this profile can be in, and an earlier enumeration that
    invented one described 864 states as 1728."""
    with pytest.raises(RunInvalid, match="reaches candidate Q alone"):
        select_first_operation(_field(
            L=_scored("L", {"short": 25.0, "long": 24.0}, killed=True)))


def test_a_pair_inside_the_band_goes_to_the_registered_tie_breaks():
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 24.6, "long": 24.6}, spread=2.0),
        Q=_scored("Q", {"short": 25.0, "long": 25.0}, spread=2.0)))
    assert set(ruling["band"]) == {"L", "Q"}
    assert ruling["anchor"] == "Q"
    assert ruling["selected"] == "L", "section 4.2's table order"


def test_both_tie_routes_are_unions_and_not_intersections():
    """A candidate joins the band unless it is certified outside the two-point
    band AND certified resolvably worse. An earlier draft used the second
    route alone, which left a candidate inside the two-point band but
    resolvably worse simultaneously in and out of it."""
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 24.0, "long": 24.0}, spread=0.05),
        Q=_scored("Q", {"short": 25.0, "long": 25.0}, spread=0.05)))
    evidence = ruling["band_evidence"]["L"]
    assert evidence["certified_resolvably_worse"] is True
    assert evidence["certified_outside_the_two_point_band"] is False
    assert evidence["in_band"] is True
    assert set(ruling["band"]) == {"L", "Q"}


def test_a_candidate_both_outside_the_band_and_resolvably_worse_is_excluded():
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 10.0, "long": 10.0}, spread=0.05),
        Q=_scored("Q", {"short": 30.0, "long": 30.0}, spread=0.05)))
    evidence = ruling["band_evidence"]["L"]
    assert evidence["certified_outside_the_two_point_band"]
    assert evidence["certified_resolvably_worse"]
    assert ruling["band"] == ["Q"]
    assert ruling["selected"] == "Q"


def test_a_pair_resolvable_at_one_width_only_is_not_resolvable():
    """Clause 16 fixes the quantifier: a pair the machine can separate at one
    width is not a pair the machine can separate."""
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 24.0, "long": 24.9}, spread=0.05),
        Q=_scored("Q", {"short": 30.0, "long": 25.0}, spread=0.05)))
    assert ruling["band_evidence"]["L"]["certified_resolvably_worse"] is False
    assert set(ruling["band"]) == {"L", "Q"}


def test_no_tie_break_of_any_kind_can_select_below_the_floor():
    """The band is drawn only from candidates the floor already admitted."""
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 9.0, "long": 9.0}, footprint_delta=-1),
        Q=_scored("Q", {"short": 25.0, "long": 24.0}, footprint_delta=99)))
    assert ruling["selected"] == "Q"
    assert "L" not in ruling["band"]


# ---------------------------------------------------------------------------
# Candidate A, which is measured, reported, and gates nothing
# ---------------------------------------------------------------------------
def test_candidate_a_never_scores_and_never_enters_the_band():
    ruling = select_first_operation(_field())
    assert [row["candidate"] for row in ruling["scored"]] == ["L", "Q"]
    assert "A" not in ruling["band"]


def test_a_low_ceiling_excludes_candidate_a_by_arithmetic_and_gates_nothing():
    ruling = select_first_operation(_field(A=_ceiling({"long": 0.05})))
    report = ruling["candidate_a"]
    assert report["route_1_excludes"] == ["long"]
    assert report["certified"] is False
    assert ruling["verdict"] == SELECTED


def test_a_high_ceiling_records_an_open_question_and_still_selects():
    """Amendment 7 clause 35: the profile can no longer decline to name an
    operation on the ground that candidate A might have been better."""
    ruling = select_first_operation(_field(A=_ceiling({"long": 0.45})))
    assert ruling["verdict"] == SELECTED
    assert ruling["selected"] == "L"
    question, = ruling["open_questions"]
    assert question["u_min"] == pytest.approx(1 / (1 - 0.45))
    assert question["set_by"] == ["long"]
    assert question["shortfall"] > 0


def test_a_winner_that_beats_the_ceiling_leaves_no_open_question():
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 40.0, "long": 40.0}),
        A=_ceiling({"long": 0.20})))
    assert ruling["candidate_a"]["route_2"]["excludes"] is True
    assert ruling["open_questions"] == []


def test_the_ceiling_is_read_against_the_SELECTED_candidate_not_the_top_score():
    """Clause 27 registers the selected candidate, on the same principle
    `ratio_lo` follows: each reduction is the one that makes its own action
    harder to take, and exclusion is the only action this route takes."""
    ruling = select_first_operation(_field(
        L=_scored("L", {"short": 24.6, "long": 24.6}, spread=2.0),
        Q=_scored("Q", {"short": 25.0, "long": 25.0}, spread=2.0),
        A=_ceiling({"long": 0.248})))
    assert ruling["anchor"] == "Q"
    assert ruling["selected"] == "L"
    assert ruling["candidate_a"]["route_2"]["against"] == "L"


def test_a_ceiling_valid_at_one_width_narrows_rather_than_refusing():
    ruling = select_first_operation(_field(A=_ceiling({"long": 0.05})))
    assert ruling["candidate_a"]["valid_widths"] == ["long"]
    assert ruling["candidate_a"]["u_min_set_by"] == ["long"]


def test_candidate_a_absent_at_both_widths_is_never_incompleteness():
    """REVERSED from Amendment 6, where a candidate A with no valid width
    reached INCOMPLETE."""
    ruling = select_first_operation(_field(
        A=CandidateInput("A", absence=MISSING_SHARE)))
    assert ruling["verdict"] == SELECTED
    assert ruling["candidate_a"]["ceiling"] is None
    assert ruling["open_questions"] == []


def test_the_ceiling_carries_its_own_stated_bias_beside_the_number():
    """With no action left, an inflated share only makes candidate A look
    better in the report than it is, so the bias goes where the number is."""
    report = select_first_operation(_field())["candidate_a"]
    assert "inflated" in report["stated_bias"]


# ---------------------------------------------------------------------------
# The refusal class, which is not a terminal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("share", [1.0, 1.01, 0.0, -0.1])
def test_a_share_that_is_not_a_fraction_refuses_the_whole_profile(share):
    """Clause 27: the sibling width is not used, because it came from the
    same instrument, and a share of 1.401 is exactly what exposed the
    instrument Amendment 5 exists to replace."""
    with pytest.raises(RunInvalid, match="not a fraction"):
        select_first_operation(_field(A=_ceiling({"long": share})))


def test_a_non_finite_reading_refuses_before_any_rule_reads_it():
    """A NaN satisfies neither branch of any test, so one of them makes the
    terminal table neither total nor disjoint."""
    with pytest.raises(RunInvalid, match="must be finite"):
        Interval(value=float("nan"), low=0.0, high=1.0)
    with pytest.raises(RunInvalid, match="must be finite"):
        Interval(value=1.0, low=0.0, high=float("inf"))


def test_the_saving_fraction_bounds_hold_at_every_corner_of_its_box():
    """`K/T` is monotone in `T` for fixed `K`, and which corner of `T` gives
    the extreme depends on the SIGN of `K`. Pairing the smallest saving with
    the largest step unconditionally is right only while `K` is positive, and
    a wide box is exactly where a saving reaches below zero and exactly where
    the band test and route 2 read the low end."""
    entry = ScoredEntry(width="short",
                        step_total=Interval(100.0, 94.0, 106.0),
                        saving=Interval(5.0, -1.0, 11.0))
    fraction = profile_rules._fraction(entry)
    for saving in (-1.0, 0.0, 5.0, 11.0):
        for total in (94.0, 100.0, 106.0):
            assert fraction.low - 1e-12 <= saving / total <= fraction.high + 1e-12
    assert fraction.low == pytest.approx(-1.0 / 94.0)


def test_candidate_a_is_never_typed_as_missing_a_ratio():
    """Clause 30: for candidate A that is not an absence at all, it is the
    amendment's permanent state, and typing it would discard the share the
    ceiling is built from."""
    with pytest.raises(RunInvalid, match="permanent state"):
        select_first_operation(_field(
            A=CandidateInput("A", absence=MISSING_RATIO)))


def test_candidate_a_is_recorded_even_when_the_profile_is_incomplete():
    """A reported quantity reaches no terminal and is still evidence."""
    ruling = select_first_operation(_field(
        L=CandidateInput("L", absence=MISSING_SHARE)))
    assert ruling["verdict"] == INCOMPLETE
    assert ruling["candidate_a"]["u_min"] == pytest.approx(1 / (1 - 0.05))


def test_a_measured_value_outside_its_own_box_refuses():
    with pytest.raises(RunInvalid, match="outside its own box"):
        Interval(value=5.0, low=1.0, high=2.0)


def test_a_step_with_no_measured_time_cannot_be_a_denominator():
    with pytest.raises(RunInvalid, match="step with no measured time"):
        select_first_operation(_field(
            L=_scored("L", {"short": 25.0, "long": 24.0}, total=0.4,
                      spread=0.5)))


def test_the_gain_formula_is_strict_at_a_share_of_one():
    """Committed code returned `r` there and raised nothing; clause 27 makes
    it a fault and registers the fix as step 8's obligation."""
    assert gain(0.5, 2.0) == pytest.approx(4 / 3)
    with pytest.raises(RunInvalid, match="not a fraction"):
        gain(1.0, 3.0)


# ---------------------------------------------------------------------------
# The input contract, clause 30
# ---------------------------------------------------------------------------
def test_the_rule_refuses_a_candidate_it_never_registered():
    with pytest.raises(RunInvalid, match="not candidates"):
        select_first_operation(_field() + [_scored("Z", {"short": 1.0})])


def test_the_rule_refuses_a_candidate_simply_left_out():
    """A candidate that could disappear between the profile and the ruling is
    the fault clause 30 exists to make impossible."""
    field = [one for one in _field() if one.candidate != "Q"]
    with pytest.raises(RunInvalid, match="no state for"):
        select_first_operation(field)


def test_the_rule_refuses_two_states_for_one_candidate():
    with pytest.raises(RunInvalid, match="more than one input"):
        select_first_operation(_field() + [_scored("L", {"short": 1.0})])


def test_the_rule_refuses_an_untyped_absence():
    with pytest.raises(RunInvalid, match="untyped absence"):
        select_first_operation(_field(
            L=CandidateInput("L", absence="vanished")))


def test_the_rule_refuses_a_candidate_that_is_both_present_and_absent():
    with pytest.raises(RunInvalid, match="both a typed absence and readings"):
        select_first_operation(_field(
            L=_scored("L", {"short": 25.0, "long": 24.0},
                      absence=MISSING_RATIO)))


def test_the_rule_refuses_a_candidate_that_is_neither():
    with pytest.raises(RunInvalid, match="neither readings nor"):
        select_first_operation(_field(L=CandidateInput("L")))


def test_a_scored_candidate_measured_at_one_width_has_no_worse_to_take():
    with pytest.raises(RunInvalid, match="WORSE of both"):
        select_first_operation(_field(L=_scored("L", {"short": 25.0})))


def test_the_rule_refuses_a_width_nobody_registered():
    with pytest.raises(RunInvalid, match="not registered"):
        select_first_operation(_field(
            L=_scored("L", {"short": 25.0, "medium": 24.0})))


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
# Clause 38's terminal table, verified by enumeration rather than by reading
# ---------------------------------------------------------------------------
_STATES = ("scored", MISSING_SHARE, MISSING_RATIO)
_CEILINGS = {"absent": None, "low": 0.05, "high": 0.45}


def _enumerate():
    """Every legal state of the reduced space, with its ruling.

    The axes are clause 38's: each of candidates L and Q's three typed states,
    candidate Q's kill flag, each scored candidate's shipping margin, whether
    clause 16's pair test resolves, and candidate A's width states with both
    readings of its exclusion.

    A killed candidate L is NOT enumerated, because clause 23's rule reaches
    candidate Q alone. An earlier draft of the amendment enumerated one and
    described 864 states as 1728, half of them states no run can reach.
    """
    for l_state in _STATES:
        for q_state in _STATES:
            for l_ships in (True, False):
                for q_ships in (True, False):
                    for spread in (0.5, 6.0):
                        for killed in (False, True):
                            for a_state, share in _CEILINGS.items():
                                field = {}
                                for name, state, ships in (
                                        ("L", l_state, l_ships),
                                        ("Q", q_state, q_ships)):
                                    if state != "scored":
                                        field[name] = CandidateInput(
                                            name, absence=state,
                                            killed=killed and name == "Q")
                                        continue
                                    saving = 25.0 if ships else 5.0
                                    if name == "Q":
                                        saving -= 1.0
                                    field[name] = _scored(
                                        name, {"short": saving,
                                               "long": saving},
                                        spread=spread,
                                        killed=killed and name == "Q")
                                field["A"] = (
                                    CandidateInput("A", absence=MISSING_SHARE)
                                    if share is None
                                    else _ceiling({"long": share}))
                                key = (l_state, q_state, l_ships, q_ships,
                                       spread, killed, a_state)
                                yield key, select_first_operation(
                                    list(field.values()))


def _payload(ruling):
    """What a terminal NAMES, not merely what it is labelled.

    A table can hold its terminal invariant while the operation it names
    moves, and a profile that names a different operation has made a different
    ruling whatever its label says.
    """
    if ruling["verdict"] == SELECTED:
        return ("ships", tuple(sorted([ruling["selected"]]
                                      + ruling["tied_with"])))
    if ruling["verdict"] == NO_SELECTION:
        return ("keep", tuple(ruling["keep"]))
    return ("incomplete", tuple(ruling["incomplete"]))


def test_every_legal_state_lands_on_exactly_one_of_three_terminals():
    seen, count = set(), 0
    for _key, ruling in _enumerate():
        assert ruling["verdict"] in (SELECTED, NO_SELECTION, INCOMPLETE)
        seen.add(ruling["verdict"])
        # Disjointness is structural: a verdict names a selection or it does
        # not, and INCOMPLETE names neither a selection nor a kept list.
        if ruling["verdict"] == SELECTED:
            assert ruling["selected"] is not None
            assert "keep" not in ruling and "incomplete" not in ruling
        else:
            assert ruling["selected"] is None
        count += 1
    assert count == 432
    assert seen == {SELECTED, NO_SELECTION, INCOMPLETE}, (
        "an enumeration that never reaches a terminal proves nothing about it")


def test_no_state_s_terminal_or_payload_moves_with_candidate_a():
    """Amendment 7 clause 35's assertion, checked on the pair of terminal AND
    payload because an earlier draft compared bare labels."""
    grouped = {}
    for key, ruling in _enumerate():
        without_a = key[:-1]
        grouped.setdefault(without_a, set()).add(
            (ruling["verdict"], _payload(ruling)))
    moved = {key: values for key, values in grouped.items()
             if len(values) > 1}
    assert not moved, f"{len(moved)} states move with candidate A"


def test_no_state_s_terminal_or_payload_moves_with_the_kill_reading():
    """Amendment 7 clause 36's assertion, on the same stronger comparison."""
    grouped = {}
    for key, ruling in _enumerate():
        without_kill = key[:5] + key[6:]
        grouped.setdefault(without_kill, set()).add(
            (ruling["verdict"], _payload(ruling)))
    moved = {key: values for key, values in grouped.items()
             if len(values) > 1}
    assert not moved, f"{len(moved)} states move with the kill reading"


def test_candidate_a_still_reaches_the_artifact_in_every_state_it_is_valid():
    """Invariance is not silence: the ceiling and both routes are recorded
    wherever candidate A has a valid share, and only the terminal is deaf."""
    for key, ruling in _enumerate():
        if key[-1] == "absent":
            continue
        report = ruling["candidate_a"]
        assert report["u_min"] is not None
        assert report["certified"] is False


def test_the_worked_disagreement_with_amendment_six_reaches_SELECTED():
    """Clause 38's own worked case: both candidates scoreable and above the
    floor, candidate A valid and neither route excluding it. Amendment 6
    returns UNRESOLVED and this amendment returns SELECTED."""
    ruling = select_first_operation(_field(A=_ceiling({"long": 0.45})))
    assert ruling["verdict"] == SELECTED
    assert ruling["open_questions"], "the veto becomes a recorded risk"
    assert not hasattr(profile_rules, "UNRESOLVED")


# ---------------------------------------------------------------------------
# Clause 12's footprint tie-break, which has TWO fall-through conditions
# ---------------------------------------------------------------------------
def _tied(**over):
    """Two candidates inside the band, so a tie-break decides the winner."""
    field = {"L": _scored("L", {"short": 24.6, "long": 24.6}, spread=2.0),
             "Q": _scored("Q", {"short": 25.0, "long": 25.0}, spread=2.0),
             "A": _ceiling({"long": 0.05})}
    field.update(over)
    return list(field.values())


def test_candidate_l_in_the_band_sends_it_to_table_order_whatever_was_measured():
    """Clause 12's SECOND condition, which committed code did not carry.
    Candidate L's baseline is measured on the bench and the other two in the
    step, so its delta is not comparable with theirs."""
    ruling = select_first_operation(_tied(
        L=_scored("L", {"short": 24.6, "long": 24.6}, spread=2.0,
                  footprint_delta=99),
        Q=_scored("Q", {"short": 25.0, "long": 25.0}, spread=2.0,
                  footprint_delta=1)))
    assert set(ruling["band"]) == {"L", "Q"}
    assert ruling["footprint_measured"] is False
    assert ruling["selected"] == "L", "table order, not the smaller delta"
    assert "not comparable" in ruling["reason"]


def _row(candidate, footprint):
    return {"candidate": candidate, "footprint_delta": footprint}


def test_an_unmeasured_delta_sends_the_whole_band_to_table_order():
    """Clause 12's FIRST condition. Ranking a measured delta against an
    unmeasured one would decide the sprint on which candidate happened to get
    a number."""
    tied, reason, measured = profile_rules.break_band(
        [_row("Q", 5), _row("A", None)])
    assert measured is False
    assert "not measured for every candidate" in reason
    assert [row["candidate"] for row in tied] == ["A", "Q"]


def test_a_fully_measured_band_without_candidate_l_uses_the_footprint():
    tied, reason, measured = profile_rules.break_band(
        [_row("Q", 5), _row("A", 1)])
    assert measured is True
    assert "smaller peak-footprint delta" in reason
    assert [row["candidate"] for row in tied] == ["A", "Q"]


def test_the_tie_break_is_unreachable_under_this_manifest_and_says_so():
    """Only candidates L and Q ever score, so any band with more than one
    member contains candidate L and clause 12's second condition always fires.
    The rule is carried anyway so a later amendment adding a scoreable
    candidate gets it rather than a rediscovery."""
    assert "unreachable in practice" in profile_rules.break_band.__doc__
    _tied_rows, _reason, measured = profile_rules.break_band(
        [_row("L", 1), _row("Q", 5)])
    assert measured is False


def test_a_band_of_one_needs_no_tie_break_at_all():
    ruling = select_first_operation(_field())
    assert ruling["band"] == ["L"]
    assert ruling["tied_with"] == []
    assert ruling["reason"].startswith("the largest certified score")


def test_the_footprint_defaults_to_unmeasured():
    """A candidate that says nothing about footprint must not read as zero,
    which would silently win every tie-break."""
    assert _scored("L", {"short": 1.0}).footprint_delta is None


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
