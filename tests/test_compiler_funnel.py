"""The funnel's one job: nothing is timed before it is verified.

Rule V1 of the sprint pre-registration is a sentence in a document, and a
sentence in a document is what someone breaks while wiring stages together.
These tests are the enforcement: a stage list that would time before it
verifies must be refused at construction, and a candidate that fails
anything must never reach a later stage at run time.

The third property is about honesty rather than order. A stage that raises
has not judged the candidate, so it is recorded as errored and never as
failed; otherwise a bug in our own stage code would show up in the census as
a pile of bad kernels.

Pure Python, no MLX and no GPU.
"""

import pytest

from kernelverify.compiler.funnel import (
    Funnel,
    FunnelOrderError,
    Stage,
    StageOutcome,
)
from kernelverify.compiler.store import CandidateStore

SOURCE = "kernel void a() { }"


@pytest.fixture()
def store(tmp_path):
    return CandidateStore(tmp_path)


def _stage(name, passed, *, verifies=True, detail="", seen=None):
    def run(source, **_context):
        if seen is not None:
            seen.append(name)
        return StageOutcome(passed=passed, detail=detail)

    return Stage(name=name, run=run, verifies=verifies)


# ---------------------------------------------------------------------------
# V1, enforced before anything runs
# ---------------------------------------------------------------------------
def test_a_timing_stage_before_a_verification_stage_is_refused():
    with pytest.raises(FunnelOrderError, match="V1"):
        Funnel([_stage("price", True, verifies=False), _stage("gate", True)])


def test_verification_then_timing_is_the_allowed_order():
    funnel = Funnel([_stage("gate", True), _stage("price", True, verifies=False)])
    assert [s.name for s in funnel.stages] == ["gate", "price"]


def test_an_empty_funnel_is_refused():
    with pytest.raises(FunnelOrderError, match="verifies nothing"):
        Funnel([])


def test_duplicate_stage_names_are_refused():
    """Two stages of the same name write two records nothing can tell apart,
    which makes the census unreadable."""
    with pytest.raises(FunnelOrderError, match="unique"):
        Funnel([_stage("gate", True), _stage("gate", True)])


# ---------------------------------------------------------------------------
# V1, enforced while running
# ---------------------------------------------------------------------------
def test_a_failed_candidate_never_reaches_a_later_stage(store):
    seen = []
    funnel = Funnel([_stage("compile", True, seen=seen),
                     _stage("lint", False, detail="half accumulator", seen=seen),
                     _stage("gate", True, seen=seen),
                     _stage("price", True, verifies=False, seen=seen)])

    result = funnel.screen(store, store.propose(SOURCE, origin="seed"))

    assert seen == ["compile", "lint"], "nothing may run behind a failure"
    assert not result.passed and result.reached == "lint"
    assert result.detail == "half accumulator"


def test_a_candidate_that_passes_everything_reaches_the_last_stage(store):
    seen = []
    funnel = Funnel([_stage("compile", True, seen=seen),
                     _stage("gate", True, seen=seen),
                     _stage("price", True, verifies=False, seen=seen)])

    result = funnel.screen(store, store.propose(SOURCE, origin="seed"))

    assert seen == ["compile", "gate", "price"]
    assert result.passed and result.reached == "price"


# ---------------------------------------------------------------------------
# What the store ends up holding
# ---------------------------------------------------------------------------
def test_every_stage_the_candidate_reached_is_recorded_including_the_failure(store):
    candidate = store.propose(SOURCE, origin="generator")
    Funnel([_stage("compile", True), _stage("lint", False, detail="bf16 tile"),
            _stage("gate", True)]).screen(store, candidate)

    events = store.get(candidate).events
    assert [(e["stage"], e["passed"]) for e in events] == \
        [("compile", True), ("lint", False)]
    assert store.get(candidate).outcome == "lint:failed"


def test_the_source_reaches_the_stage(store):
    got = []
    Funnel([Stage("peek", lambda source, **_: got.append(source)
                  or StageOutcome(True))]).screen(
        store, store.propose(SOURCE, origin="seed"))
    assert got == [SOURCE]


def test_context_is_passed_through_to_every_stage(store):
    got = []
    funnel = Funnel([Stage("a", lambda source, **kw: got.append(kw)
                           or StageOutcome(True)),
                     Stage("b", lambda source, **kw: got.append(kw)
                           or StageOutcome(True))])
    funnel.screen(store, store.propose(SOURCE, origin="seed"), runner="R")
    assert got == [{"runner": "R"}, {"runner": "R"}]


# ---------------------------------------------------------------------------
# A stage that raises has not judged anything
# ---------------------------------------------------------------------------
def test_a_raising_stage_is_recorded_as_errored_not_failed(store):
    def explode(source, **_context):
        raise RuntimeError("the compiler segfaulted")

    candidate = store.propose(SOURCE, origin="generator")
    result = Funnel([Stage("compile", explode), _stage("gate", True)]).screen(
        store, candidate)

    assert result.errored and not result.passed
    assert store.get(candidate).outcome == "compile:errored", (
        "an error must not be counted as a refusal, or a bug in our own stage "
        "code reads as a census of bad kernels")
    assert "RuntimeError" in result.detail and "segfaulted" in result.detail


def test_a_raising_stage_still_stops_the_funnel(store):
    seen = []
    def explode(source, **_context):
        seen.append("compile")
        raise RuntimeError("boom")

    Funnel([Stage("compile", explode), _stage("gate", True, seen=seen)]).screen(
        store, store.propose(SOURCE, origin="seed"))
    assert seen == ["compile"]


def test_the_census_separates_errors_from_refusals(store):
    def explode(source, **_context):
        raise RuntimeError("boom")

    funnel_ok = Funnel([_stage("lint", False, detail="bf16")])
    funnel_bad = Funnel([Stage("lint", explode)])
    funnel_ok.screen(store, store.propose("a", origin="generator"))
    funnel_bad.screen(store, store.propose("b", origin="generator"))

    assert store.census() == {"lint:failed": 1, "lint:errored": 1}
