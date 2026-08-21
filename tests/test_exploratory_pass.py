"""Amendment 13 clause 57's pass and Amendment 14's construction of its terms.

Every test here is device-free. What the pass MEASURES needs a GPU; what it
READS is arithmetic over recordings, and that is what decides whether three
hours of a single-GPU machine buy a branch anyone can act on.
"""

import json

import pytest

import exploratory_pass as ep
import profile_rules as rules
import profile_stock as ps
from decode_rules import RunInvalid

WIDTHS = rules.WIDTH_ORDER


def _pass(medians, *, contrasts=None, kind="exploratory"):
    """One profile recording, carrying only what the reader reads.

    `medians` is `{width: {label: median_ms}}`. Every identical arm is given
    two samples straddling its median, so the reader's own median is exercised
    rather than handed a single number to pass through.
    """
    cells, readings = {}, {}
    widths = {}
    for width, labels in medians.items():
        arms = {label: {"samples_ms": [value - 1.0, value + 1.0]}
                for label, value in labels.items()}
        widths[width] = {"width": width, "arms": arms}
        readings[width] = {
            "pair_contrasts_ms": (contrasts or {}).get(width, {"pair0": 0.5,
                                                               "pair1": 0.5})}
    cells[rules.PRIMARY_CELL] = {"widths": widths, "readings": readings}
    return {"kind": kind, "cells": cells}


def _flat(value, *, kind="exploratory"):
    """Three passes whose identical arms all sit at one median."""
    return _pass({width: {label: value for label in ep._pair_labels()}
                  for width in WIDTHS}, kind=kind)


def _margins(per_width):
    """What `exploratory_margins` returns, reduced to what the branch reads."""
    return {"cell": rules.PRIMARY_CELL, "binds": False,
            "widths": {width: {"margin_ms": value}
                       for width, value in per_width.items()}}


# ---------------------------------------------------------------------------
# Clause 58: the window-separated family is built across passes
# ---------------------------------------------------------------------------
def test_three_passes_give_exactly_three_window_separated_contrasts():
    """Clause 57's branch reads three, and three passes is what makes three."""
    profiles = [_flat(100.0), _flat(100.0), _flat(100.0)]
    for width in WIDTHS:
        separated = ep.window_separated_contrasts(profiles, width)
        assert len(separated) == 3
        assert [one["passes"] for one in separated] == [[1, 2], [1, 3], [2, 3]]
        assert all(one["separated_by_a_pass"] for one in separated)


def test_the_window_separated_contrast_is_the_maximum_over_the_four_labels():
    """Clause 58 takes the maximum, and the reason is the branch's direction.

    A larger denominator makes the LEAN schedule harder to take, and the lean
    schedule is the cheaper one, so the maximum is the conservative choice.
    Designating one label of four would leave the branch movable by which
    label was designated.
    """
    first = _pass({width: {"pair0:a": 100.0, "pair0:b": 100.0,
                           "pair1:a": 100.0, "pair1:b": 100.0}
                   for width in WIDTHS})
    second = _pass({width: {"pair0:a": 101.0, "pair0:b": 100.0,
                            "pair1:a": 100.0, "pair1:b": 107.0}
                    for width in WIDTHS})
    third = _flat(100.0)
    separated = ep.window_separated_contrasts([first, second, third], "short")
    by_passes = {tuple(one["passes"]): one for one in separated}
    assert by_passes[(1, 2)]["contrast_ms"] == pytest.approx(7.0)
    assert by_passes[(1, 2)]["by_label_ms"]["pair0:a"] == pytest.approx(1.0)
    assert by_passes[(1, 3)]["contrast_ms"] == pytest.approx(0.0)


def test_every_within_pass_contrast_is_marked_as_sitting_inside_one_pass():
    """Clause 58's correction: all six of clause 57's six are within-pass.

    All four pair arms are arms of a manifest `timed_rounds` interleaves
    inside every round, so no contrast the manifest itself produces can be
    separated by a whole pass.
    """
    profiles = [_flat(100.0)] * 3
    within = ep.within_pass_contrasts(profiles, "short")
    assert len(within) == 6
    assert not any(one["separated_by_a_pass"] for one in within)
    assert sorted({one["pass"] for one in within}) == [1, 2, 3]


def test_the_manifest_really_does_interleave_all_four_pair_arms():
    """The premise of clause 58, checked against the code rather than quoted.

    If the pair arms were ever timed in their own set of rounds, clause 58's
    whole argument would be wrong and the within-pass family would not be
    within-pass at all.
    """
    manifest = ps.arm_manifest("short", exploratory=True)
    labels = [arm.label for arm in manifest]
    for label in ep._pair_labels():
        assert label in labels
    assert all(arm.role == ps.pk.STOCK for arm in manifest
               if arm.label.startswith(ps.PAIR_PREFIX))


def test_a_pass_missing_an_identical_arm_refuses_rather_than_reading_three():
    profiles = [_flat(100.0), _flat(100.0), _flat(100.0)]
    del profiles[1]["cells"][rules.PRIMARY_CELL]["widths"]["short"]["arms"]["pair1:b"]
    with pytest.raises(RunInvalid, match="pair1:b"):
        ep.window_separated_contrasts(profiles, "short")


# ---------------------------------------------------------------------------
# Clause 59: the ratio is formed inside a width
# ---------------------------------------------------------------------------
def test_the_ratio_is_formed_inside_a_width_and_never_across_two():
    """The case where a per-width reading and a cross-width one disagree.

    Short carries a margin of 100 against a null of 10 and long a margin of
    1000 against a null of 100, so each width clears 8 on its own while the
    smaller margin over the LARGEST null of either width is 1.0. Clause 21
    makes the two widths two contexts and clause 31 says why the second
    reading is not available.
    """
    separated = {"short": [{"contrast_ms": 10.0}],
                 "long": [{"contrast_ms": 100.0}]}
    margins = [_margins({"short": 100.0, "long": 1000.0})] * 3
    reading = ep.branch_reading(margins, separated)
    assert reading["clears"] is True
    assert reading["per_width"]["short"]["ratio"] == pytest.approx(10.0)
    assert reading["per_width"]["long"]["ratio"] == pytest.approx(10.0)
    crossed = min(100.0, 1000.0) / max(10.0, 100.0)
    assert crossed < ep.MULTIPLIER


def test_the_branch_reads_the_least_generous_of_the_three_margins():
    """Clause 57 reads the margin three times to learn its stability.

    Taking the largest would let one lucky pass buy the cheaper schedule,
    which is the one direction a pass that binds nothing must not be able to
    move.
    """
    separated = {width: [{"contrast_ms": 10.0}] for width in WIDTHS}
    margins = [_margins({width: value for width in WIDTHS})
               for value in (100.0, 100.0, 20.0)]
    reading = ep.branch_reading(margins, separated)
    assert reading["per_width"]["short"]["margin_read_ms"] == pytest.approx(20.0)
    assert reading["clears"] is False
    assert reading["follows"] == ep.RULED


def test_the_branch_clears_at_exactly_the_multiplier_and_not_below():
    separated = {width: [{"contrast_ms": 10.0}] for width in WIDTHS}
    at = ep.branch_reading([_margins({w: 80.0 for w in WIDTHS})] * 3, separated)
    below = ep.branch_reading([_margins({w: 79.9 for w in WIDTHS})] * 3,
                              separated)
    assert at["clears"] is True and at["follows"] == ep.LEAN
    assert below["clears"] is False and below["follows"] == ep.RULED


def test_the_largest_of_a_width_s_three_contrasts_is_what_a_margin_clears():
    separated = {"short": [{"contrast_ms": 1.0}, {"contrast_ms": 9.0},
                           {"contrast_ms": 2.0}],
                 "long": [{"contrast_ms": 1.0}]}
    reading = ep.branch_reading([_margins({"short": 20.0, "long": 800.0})] * 3,
                                separated)
    assert reading["per_width"]["short"]["window_separated_null_ms"] == 9.0
    assert reading["per_width"]["short"]["ratio"] == pytest.approx(20.0 / 9.0)
    # Against the smallest of the three it would have been 20.0, and against
    # the largest it is 2.2, so this is the assertion that pins WHICH.
    assert reading["clears"] is False


def test_a_width_with_no_margin_makes_the_branch_unreadable_not_zero():
    """An absent margin is a measurement that does not exist, never a zero."""
    separated = {width: [{"contrast_ms": 10.0}] for width in WIDTHS}
    margins = [_margins({"short": 100.0, "long": 1000.0}),
               _margins({"short": 100.0, "long": None}),
               _margins({"short": 100.0, "long": 1000.0})]
    reading = ep.branch_reading(margins, separated)
    assert reading["clears"] is None
    assert reading["per_width"]["long"]["passes_without_a_margin"] == [2]
    assert "neither branch" in reading["follows"]


def test_a_null_of_zero_refuses_rather_than_dividing_by_a_still_machine():
    separated = {"short": [{"contrast_ms": 0.0}],
                 "long": [{"contrast_ms": 10.0}]}
    reading = ep.branch_reading([_margins({"short": 5.0, "long": 500.0})] * 3,
                                separated)
    assert reading["per_width"]["short"]["ratio"] is None
    assert reading["clears"] is None


# ---------------------------------------------------------------------------
# What the report is not
# ---------------------------------------------------------------------------
def _aggregate(profiles, sweeps, margins, monkeypatch):
    """Drive `aggregate` with the margin stubbed.

    The margin has its own tests beside the code that computes it; what is
    under test here is the aggregation across three passes, and building three
    complete profile and sweep recordings to reach it would test the reduction
    twice and this once.
    """
    calls = iter(margins)
    monkeypatch.setattr(ps, "exploratory_margins",
                        lambda profile, sweep: next(calls))
    return ep.aggregate(list(zip(profiles, sweeps)))


def test_a_pass_count_clause_fifty_seven_does_not_register_is_refused(monkeypatch):
    with pytest.raises(RunInvalid, match="registers 3 passes"):
        _aggregate([_flat(100.0)] * 2, [{"kind": "exploratory"}] * 2,
                   [_margins({w: 1.0 for w in WIDTHS})] * 2, monkeypatch)


def test_a_binding_profile_recording_is_refused_by_this_reader(monkeypatch):
    profiles = [_flat(100.0), _flat(100.0, kind="binding"), _flat(100.0)]
    with pytest.raises(RunInvalid, match="pass 2 profile"):
        _aggregate(profiles, [{"kind": "exploratory"}] * 3,
                   [_margins({w: 1.0 for w in WIDTHS})] * 3, monkeypatch)


def test_a_binding_sweep_recording_is_refused_with_its_own_message(monkeypatch):
    sweeps = [{"kind": "exploratory"}, {"kind": "exploratory"},
              {"kind": "binding"}]
    with pytest.raises(RunInvalid, match="pass 3 sweep"):
        _aggregate([_flat(100.0)] * 3, sweeps,
                   [_margins({w: 1.0 for w in WIDTHS})] * 3, monkeypatch)


def test_the_report_names_no_operation_and_binds_nothing(monkeypatch):
    """Clause 57 registers that this pass names no candidate.

    Checked as a key search rather than a substring of the serialised report,
    because the report deliberately QUOTES the clause that says it decides
    nothing, and a prose match on that quote would pass for the wrong reason.
    """
    report = _aggregate([_flat(100.0)] * 3, [{"kind": "exploratory"}] * 3,
                        [_margins({w: 80.0 for w in WIDTHS})] * 3, monkeypatch)

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    found = set(keys(report))
    assert not found & {"verdict", "selected", "terminal", "operation",
                        "score", "gain"}
    assert report["branch"]["binds"] is False
    assert report["kind"] == "exploratory"
    json.dumps(report)


def test_the_report_carries_both_contrast_families_at_both_widths(monkeypatch):
    report = _aggregate([_flat(100.0)] * 3, [{"kind": "exploratory"}] * 3,
                        [_margins({w: 80.0 for w in WIDTHS})] * 3, monkeypatch)
    for width in WIDTHS:
        assert len(report["window_separated_contrasts"][width]) == 3
        assert len(report["within_pass_contrasts"][width]) == 6


# ---------------------------------------------------------------------------
# End to end, over the real reduction rather than a stubbed margin
# ---------------------------------------------------------------------------
def _real_pass(shift):
    """One pass built from the same fixtures the reduction's own tests use.

    `shift` moves every identical arm together, so the within-pass contrasts
    stay where they were and only the WINDOW-SEPARATED family moves. That is
    the separation clause 58 exists to construct, and a fixture that moved
    both at once could not tell the two families apart.
    """
    from test_profile_stock import R, _width

    readings, widths = {}, {}
    for width in WIDTHS:
        measured = _width(width, exploratory=True, jitter={
            label: [100.0 + shift] * ps.ROUNDS for label in ep._pair_labels()})
        widths[width] = measured
        readings[width] = ps.width_reading(measured, resolution_floor_ms=R,
                                           exploratory=True)
    profile = {"kind": "exploratory",
               "profile_matrix": ps.profile_matrix(readings),
               "cells": {rules.PRIMARY_CELL: {"widths": widths,
                                              "readings": readings}}}
    sweep = {"kind": "exploratory",
             "contexts": {width: widths[width]["context"]
                          for width in WIDTHS},
             "loss_bench": {"readings": {
                 width: {"per_round_slopes_ms": [1.0, 1.2, 1.1, 1.05, 1.15]}
                 for width in WIDTHS}}}
    return profile, sweep


def test_the_reader_composes_with_the_real_margin_over_three_real_passes():
    passes = [_real_pass(0.0), _real_pass(2.0), _real_pass(5.0)]
    report = ep.aggregate(passes)

    for width in WIDTHS:
        separated = {tuple(one["passes"]): one["contrast_ms"]
                     for one in report["window_separated_contrasts"][width]}
        assert separated[(1, 2)] == pytest.approx(2.0)
        assert separated[(1, 3)] == pytest.approx(5.0)
        assert separated[(2, 3)] == pytest.approx(3.0)
        row = report["branch"]["per_width"][width]
        assert row["window_separated_null_ms"] == pytest.approx(5.0)
        # Every within-pass contrast is untouched by the shift, which is what
        # says the two families measure different things.
        assert all(one["contrast_ms"] == 0.0
                   for one in report["within_pass_contrasts"][width])
        assert row["margin_read_ms"] == pytest.approx(min(row["margins_ms"]))
        assert row["ratio"] == pytest.approx(row["margin_read_ms"] / 5.0)

    assert report["branch"]["clears"] in (True, False)
    assert report["branch"]["binds"] is False
