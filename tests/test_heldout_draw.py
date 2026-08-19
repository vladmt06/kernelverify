"""The sealed draw: unpredictable, unchangeable, and unable to leak.

A generator judged on cases it can anticipate optimises for them, so three
things have to hold and each has a mechanism rather than a promise. The
draw must be unguessable without the secret. The secret must be the one the
sprint committed to, so nobody can re-roll it into a friendlier draw. And a
failure must have nowhere to put a reason, because a reason is feedback and
feedback turns a held-out set into a training set.

Pure Python, no MLX and no GPU. The real salt on this machine is read by the
tests that check the commitment; every other test passes its own.
"""

import hashlib

import pytest

from kernelverify.compiler.heldout import (
    SALT_SHA256,
    SaltError,
    Verdict,
    draw,
    load_salt,
    seed_for,
)

SPACE = {
    "tokens": [1, 7, 31, 2048, 8192],
    "shape": [(4096, 2560), (2560, 4096), (9728, 2560)],
    "dtype": ["bfloat16", "float32"],
    "distribution": ["normal", "near-zero", "opposed-signs", "constant-rows"],
}
SALT = b"a" * 64
OTHER_SALT = b"b" * 64
CANDIDATE = "a" * 64


# ---------------------------------------------------------------------------
# Sealed: there is nowhere to put a reason
# ---------------------------------------------------------------------------
def test_a_verdict_carries_a_boolean_and_nothing_else():
    assert str(Verdict(True)) == "PASS" and str(Verdict(False)) == "FAIL"
    with pytest.raises(TypeError):
        Verdict(False, "failed at tokens=7")  # no field exists to hold it


def test_a_verdict_is_frozen_so_a_reason_cannot_be_attached_later():
    verdict = Verdict(False)
    with pytest.raises(Exception):
        verdict.detail = "failed at tokens=7"


# ---------------------------------------------------------------------------
# Unpredictable: the salt is what makes the check mean anything
# ---------------------------------------------------------------------------
def test_the_same_candidate_and_salt_always_draw_the_same_cells():
    """A draw that varied between runs would make a held-out failure
    unreproducible, and an unreproducible failure is not evidence."""
    assert draw(CANDIDATE, SPACE, salt=SALT) == draw(CANDIDATE, SPACE, salt=SALT)


def test_a_different_salt_draws_differently_for_the_same_candidate():
    """The whole point of the secret: without it the draw is not computable."""
    assert draw(CANDIDATE, SPACE, salt=SALT) != draw(CANDIDATE, SPACE, salt=OTHER_SALT)


def test_different_candidates_get_different_draws():
    draws = {tuple(sorted(draw("c" * 63 + str(i), SPACE, salt=SALT).items()))
             for i in range(8)}
    assert len(draws) > 1, "every candidate drawing the same cells is one cell"


def test_the_seed_depends_on_both_the_candidate_and_the_salt():
    assert seed_for(CANDIDATE, SALT) != seed_for(CANDIDATE, OTHER_SALT)
    assert seed_for(CANDIDATE, SALT) != seed_for("b" * 64, SALT)


def test_a_draw_picks_one_real_choice_from_every_axis():
    got = draw(CANDIDATE, SPACE, salt=SALT)
    assert set(got) == set(SPACE)
    for axis, chosen in got.items():
        assert chosen in SPACE[axis]


def test_an_empty_space_is_refused_because_it_withholds_nothing():
    with pytest.raises(ValueError, match="withholds nothing"):
        draw(CANDIDATE, {}, salt=SALT)


def test_an_axis_with_no_choices_is_refused():
    with pytest.raises(ValueError, match="no choices"):
        draw(CANDIDATE, {"tokens": []}, salt=SALT)


# ---------------------------------------------------------------------------
# Unchangeable: the commitment made before any draw
# ---------------------------------------------------------------------------
def test_the_salt_on_this_machine_is_the_one_the_sprint_committed_to():
    assert hashlib.sha256(load_salt()).hexdigest() == SALT_SHA256


def test_a_re_rolled_salt_is_refused_because_it_is_a_re_rolled_draw(tmp_path):
    forged = tmp_path / "salt"
    forged.write_text("a friendlier salt\n")
    with pytest.raises(SaltError, match="not the salt the sprint committed"):
        load_salt(forged)


def test_a_missing_salt_refuses_rather_than_drawing_unsalted(tmp_path):
    """An unsalted draw is one the generator could compute for itself, which
    is worse than no held-out check at all because it looks like one."""
    with pytest.raises(SaltError, match="no unsalted"):
        load_salt(tmp_path / "absent")
