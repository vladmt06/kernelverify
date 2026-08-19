"""The knob sweep: what this chip can launch, and an honest account of the rest.

The temptation in a knob search is to drop settings that look unpromising,
which turns a sweep of the space into a sweep of somebody's taste while the
report still says "swept". So only two things are pruned here, both
properties of the device, and both are checked against limits read from the
device rather than written down.

The rejections are kept, because a session that reports 96 launchable
settings without saying it enumerated 240 has described a different search
than the one it ran.
"""

import pytest

from conftest import requires_metal
from kernelverify.compiler.search_space import (
    Limits,
    NoDevice,
    enumerate_knobs,
)
from kernelverify.runners import MetalRunner

LIMITS = Limits(max_threads_per_threadgroup=1024, max_threadgroup_memory=32768)
AXES = {"tile": [8, 16, 32, 64], "simdgroups": [1, 2, 4, 8]}


def _threads(knobs):
    return 32 * knobs["simdgroups"]


def _bytes(knobs):
    return 4 * knobs["tile"] * knobs["simdgroups"] * 32


# ---------------------------------------------------------------------------
# The limits come from the chip, never from a constant
# ---------------------------------------------------------------------------
@requires_metal
def test_the_limits_are_read_from_the_device():
    limits = Limits.probe(MetalRunner())
    assert limits.max_threads_per_threadgroup > 0
    assert limits.max_threadgroup_memory > 0


@requires_metal
def test_this_machine_reports_the_figures_the_plan_assumed():
    """The plan says "the 32 KB threadgroup limit". That is true here and is
    a per-chip fact, so it is checked rather than trusted, and it lives in a
    test rather than in the module."""
    limits = Limits.probe(MetalRunner())
    assert limits.max_threadgroup_memory == 32768
    assert limits.max_threads_per_threadgroup == 1024


def test_a_machine_with_no_device_refuses_rather_than_guessing():
    class Absent:
        def probe(self):
            return None

    with pytest.raises(NoDevice, match="nothing to fall back to"):
        Limits.probe(Absent())


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------
def test_every_setting_is_either_launchable_or_rejected_with_a_reason():
    space = enumerate_knobs(AXES, LIMITS, threads=_threads,
                            threadgroup_bytes=_bytes)
    assert space.enumerated == 16
    assert all(entry.reason for entry in space.rejected)
    assert len(space.launchable) + len(space.rejected) == space.enumerated


def test_a_setting_over_the_thread_limit_is_rejected_and_the_reason_says_so():
    space = enumerate_knobs({"simdgroups": [64]}, LIMITS,
                            threads=_threads, threadgroup_bytes=lambda k: 0)
    assert not space.launchable
    assert "2048 threads is over this device's 1024" in space.rejected[0].reason


def test_a_setting_over_the_memory_limit_is_rejected_and_the_reason_says_so():
    space = enumerate_knobs({"tile": [1024]}, LIMITS, threads=lambda k: 32,
                            threadgroup_bytes=lambda k: k["tile"] * 64)
    assert not space.launchable
    assert "65536 bytes" in space.rejected[0].reason
    assert "32768" in space.rejected[0].reason


def test_a_merely_wasteful_setting_is_kept_because_the_clock_should_decide():
    """One thread per threadgroup is a terrible setting and a launchable one.
    It stays in the sweep and loses on time, which is a measurement; dropping
    it here would be an opinion wearing a measurement's clothes."""
    space = enumerate_knobs({"threads": [1]}, LIMITS,
                            threads=lambda k: k["threads"],
                            threadgroup_bytes=lambda k: 0)
    assert space.launchable == ({"threads": 1},)


def test_zero_threads_is_rejected_because_it_is_not_a_launch():
    space = enumerate_knobs({"threads": [0]}, LIMITS,
                            threads=lambda k: k["threads"],
                            threadgroup_bytes=lambda k: 0)
    assert "at least one thread" in space.rejected[0].reason


# ---------------------------------------------------------------------------
# The account of the sweep
# ---------------------------------------------------------------------------
def test_the_census_names_the_survivors_and_every_reason():
    space = enumerate_knobs(AXES, LIMITS, threads=_threads,
                            threadgroup_bytes=_bytes)
    census = space.census
    assert census["launchable"] == len(space.launchable)
    assert sum(census.values()) == space.enumerated
    assert len(census) > 1, "a sweep that rejected nothing here would be wrong"


def test_the_walk_is_deterministic_so_two_runs_can_be_compared():
    first = enumerate_knobs(AXES, LIMITS, threads=_threads, threadgroup_bytes=_bytes)
    second = enumerate_knobs(AXES, LIMITS, threads=_threads, threadgroup_bytes=_bytes)
    assert first.launchable == second.launchable
    assert [r.knobs for r in first.rejected] == [r.knobs for r in second.rejected]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_an_empty_space_is_refused():
    with pytest.raises(ValueError, match="nothing to search"):
        enumerate_knobs({}, LIMITS, threads=_threads, threadgroup_bytes=_bytes)


def test_an_axis_with_no_choices_is_refused():
    with pytest.raises(ValueError, match="no choices"):
        enumerate_knobs({"tile": []}, LIMITS, threads=_threads,
                        threadgroup_bytes=_bytes)
