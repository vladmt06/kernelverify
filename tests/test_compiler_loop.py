"""The loop running for real: a seed body, a knob sweep, and a session record.

Nothing so far has shown the machine actually turning. This does: one
hand-written Metal body with holes in it, swept across every knob setting the
chip can launch, each specialization compiled, linted and screened for
unwritten cells on the real GPU, all of it recorded, and a session report at
the end.

The property that earns its own tests is the accounting. A knob sweep
produces the same source twice whenever a knob does not appear in the body,
and those re-proposals must be counted separately: as candidates they would
inflate the loop's yield, as failures they would invent faults that never
happened.
"""

import numpy as np
import pytest

from conftest import requires_metal
from kernelverify.compiler import loop
from kernelverify.compiler.funnel import Funnel, Stage, StageOutcome
from kernelverify.compiler.generator import describe, sweep
from kernelverify.compiler.loop import Proposal, Session
from kernelverify.compiler.search_space import Limits
from kernelverify.compiler.stages import compile_stage, lint_stage, unwritten_stage
from kernelverify.compiler.store import CandidateStore
from kernelverify.runners import MetalRunner
from kernelverify.runners.spec import (
    Binding,
    BindingKind,
    KernelSpec,
    LaunchSpec,
    RunCase,
)

N = 256

# One body, two holes: the number of elements a thread handles, and a scale it
# multiplies by. UNROLL changes the work per thread, so the launch changes with
# it, which is exactly the case Proposal.extra exists for.
SEED = """
[[kernel]] void f(device const float* x [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  constant uint& n [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    for (uint i = 0; i < $UNROLL; ++i) {
        uint at = gid * $UNROLL + i;
        if (at < n) { float acc = x[at] * $SCALE.0f; out[at] = acc; }
    }
}
"""

BINDINGS = (Binding(BindingKind.INPUT, "x"), Binding(BindingKind.OUTPUT),
            Binding(BindingKind.SCALAR, "n", "uint32"))
AXES = {"UNROLL": [1, 2, 4], "SCALE": [2, 3]}
LIMITS = Limits(max_threads_per_threadgroup=1024, max_threadgroup_memory=32768)


def _template() -> KernelSpec:
    return KernelSpec(source=SEED, entry_point="f", bindings=BINDINGS,
                      launch=LaunchSpec(grid=("n", 1, 1), threadgroup=(64, 1, 1)))


def _launch_for(knobs) -> LaunchSpec:
    """Fewer threads when each one does more, so the grid stays covered."""
    return LaunchSpec(grid=(N // knobs["UNROLL"], 1, 1), threadgroup=(64, 1, 1))


@pytest.fixture(scope="module")
def runner():
    return MetalRunner()


@pytest.fixture()
def context(runner):
    case = RunCase(inputs={"x": np.arange(1, N + 1, dtype=np.float32)},
                   params={"n": N}, output_shapes=[((N,), "float32")],
                   label="loop")
    return {"runner": runner, "entry_point": "f", "bindings": BINDINGS,
            "launch": _template().launch, "probe_case": case, "gate_cases": [case]}


@pytest.fixture()
def store(tmp_path):
    return CandidateStore(tmp_path)


@pytest.fixture()
def funnel():
    return Funnel([compile_stage(), lint_stage(), unwritten_stage()])


# ---------------------------------------------------------------------------
# The loop, live
# ---------------------------------------------------------------------------
@requires_metal
def test_a_seed_body_sweeps_into_candidates_that_all_survive(store, funnel,
                                                              context):
    proposals, space = sweep(_template(), AXES, LIMITS,
                             threads=lambda k: 64,
                             threadgroup_bytes=lambda k: 0,
                             launch_for=_launch_for)
    session = loop.run(proposals, funnel, store, context=context)

    assert space.enumerated == 6, "3 unrolls x 2 scales"
    assert session.proposed == 6 and session.reproposed == 0
    assert len(session.survived) == 6, session.summary()
    assert session.census == {"unwritten:passed": 6}
    assert session.seconds > 0


@requires_metal
def test_a_broken_seed_dies_at_the_same_stage_in_every_specialization(store,
                                                                      funnel,
                                                                      context):
    """The sweep varies the body, so a fault in the seed reaches every
    candidate, and the census says so rather than reporting one failure."""
    broken = KernelSpec(source=SEED.replace("float acc", "half acc"),
                        entry_point="f", bindings=BINDINGS,
                        launch=_template().launch)
    proposals, _ = sweep(broken, AXES, LIMITS, threads=lambda k: 64,
                         threadgroup_bytes=lambda k: 0, launch_for=_launch_for)
    session = loop.run(proposals, funnel, store, context=context)

    assert session.census == {"lint:failed": 6}
    assert not session.survived


# ---------------------------------------------------------------------------
# The accounting
# ---------------------------------------------------------------------------
@requires_metal
def test_a_knob_the_body_ignores_produces_re_proposals_counted_apart(store,
                                                                     funnel,
                                                                     context):
    """SCALE alone reaches the source; two settings of a knob the body never
    mentions collapse to the same text, and the store refuses the second by
    hash before it costs a compile. Those are neither candidates nor faults."""
    body_without_unroll = SEED.replace("$UNROLL", "1")
    template = KernelSpec(source=body_without_unroll.replace("$SCALE", "$SCALE"),
                          entry_point="f", bindings=BINDINGS,
                          launch=_template().launch)
    proposals, _ = sweep(template, {"SCALE": [2, 2, 3]}, LIMITS,
                         threads=lambda k: 64, threadgroup_bytes=lambda k: 0)
    session = loop.run(proposals, funnel, store, context=context)

    assert session.offered == 3
    assert session.proposed == 2 and session.reproposed == 1
    assert sum(session.census.values()) == 2, (
        "a re-proposal must not appear in the census as a candidate")


def test_the_limit_stops_after_that_many_new_candidates(store):
    """The limit counts new candidates, not proposals offered, so a sweep full
    of duplicates cannot exhaust a budget without doing any work."""
    seen = []
    noting = Funnel([Stage("note", lambda source, **_:
                           seen.append(source) or StageOutcome(True))])
    proposals = (Proposal(source=f"kernel void k{i}() {{ }}", origin="test")
                 for i in range(10))
    session = loop.run(proposals, noting, store, context={}, limit=3)

    assert session.proposed == 3 and len(seen) == 3


def test_a_session_reports_itself_even_when_nothing_survived(store):
    failing = Funnel([Stage("lint", lambda source, **_:
                            StageOutcome(False, "bf16 tile"))])
    proposals = (Proposal(source=f"kernel void k{i}() {{ }}", origin="test")
                 for i in range(4))
    session = loop.run(proposals, failing, store, context={})

    assert isinstance(session, Session)
    assert not session.survived
    assert session.census == {"lint:failed": 4}
    assert "survived 0" in session.summary() and "lint:failed: 4" in session.summary()


def test_the_knob_description_is_stable_so_two_runs_read_the_same():
    assert describe({"UNROLL": 4, "SCALE": 2}) == "SCALE=2 UNROLL=4"
