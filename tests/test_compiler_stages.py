"""The loop's spine, running for real: store, funnel, compiler, lint.

Everything below drives the actual MetalRunner on actual Metal source. The
unit tests elsewhere use fake stages to pin ordering rules; this file exists
because a spine that only ever ran against fakes has not been shown to work.

Three candidates go down the same funnel and land in three different places,
which is the property the census depends on: a kernel that does not compile,
a kernel that compiles but accumulates in half, and a kernel that is fine.
"""

import numpy as np
import pytest

from conftest import requires_metal
from kernelverify.compiler.funnel import Funnel
from kernelverify.compiler.stages import HarnessError, compile_stage, lint_stage
from kernelverify.compiler.store import CandidateStore
from kernelverify.runners import MetalRunner
from kernelverify.runners.spec import Binding, BindingKind, LaunchSpec, RunCase

GOOD = """
[[kernel]] void scale(device const float* x [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < n) { float acc = x[gid] * 2.0f; out[gid] = acc; }
}
"""

BROKEN = GOOD.replace("float acc = x[gid] * 2.0f;", "float acc = ;")

NARROW = GOOD.replace("float acc = x[gid] * 2.0f;", "half acc = x[gid] * 2.0h;")

BINDINGS = (Binding(BindingKind.INPUT, "x"), Binding(BindingKind.OUTPUT),
            Binding(BindingKind.SCALAR, "n", "uint32"))
LAUNCH = LaunchSpec(grid=("n", 1, 1), threadgroup=(32, 1, 1))


@pytest.fixture(scope="module")
def runner():
    return MetalRunner()


@pytest.fixture()
def context(runner):
    return {
        "runner": runner,
        "entry_point": "scale",
        "bindings": BINDINGS,
        "launch": LAUNCH,
        "probe_case": RunCase(inputs={"x": np.arange(32, dtype=np.float32)},
                              params={"n": 32},
                              output_shapes=[((32,), "float32")], label="probe"),
    }


@pytest.fixture()
def funnel():
    return Funnel([compile_stage(), lint_stage()])


@pytest.fixture()
def store(tmp_path):
    return CandidateStore(tmp_path)


@requires_metal
def test_a_correct_kernel_passes_both_stages(store, funnel, context):
    result = funnel.screen(store, store.propose(GOOD, origin="seed"), **context)
    assert result.passed, result.detail
    assert result.reached == "lint"


@requires_metal
def test_a_kernel_that_does_not_compile_dies_at_compile_and_says_so(store, funnel,
                                                                    context):
    candidate = store.propose(BROKEN, origin="generator")
    result = funnel.screen(store, candidate, **context)

    assert not result.passed and not result.errored
    assert result.reached == "compile"
    assert "did not compile" in result.detail
    assert store.get(candidate).outcome == "compile:failed"


@requires_metal
def test_a_half_accumulator_compiles_fine_and_is_caught_by_the_lint(store, funnel,
                                                                    context):
    """The one that matters. A narrow accumulator is perfectly valid Metal,
    so the compiler has no objection; only reading the source catches it."""
    candidate = store.propose(NARROW, origin="generator")
    result = funnel.screen(store, candidate, **context)

    assert not result.passed
    assert result.reached == "lint", "it must get past compile to prove the point"
    assert "half" in result.detail and "C1" in result.detail
    assert store.get(candidate).outcome == "lint:failed"


@requires_metal
def test_the_census_separates_the_three_of_them(store, funnel, context):
    for source, origin in ((GOOD, "seed"), (BROKEN, "generator"),
                           (NARROW, "generator")):
        funnel.screen(store, store.propose(source, origin=origin), **context)

    assert store.census() == {"lint:passed": 1, "compile:failed": 1,
                              "lint:failed": 1}


@requires_metal
def test_a_mismatched_probe_case_is_our_bug_and_is_recorded_as_errored(store,
                                                                       context):
    """A case that does not match the spec says nothing about the candidate.
    Counting it as a bad kernel would put our own bugs in the census."""
    context["probe_case"] = RunCase(inputs={"wrong_name": np.zeros(4, np.float32)},
                                    params={"n": 4},
                                    output_shapes=[((4,), "float32")])
    candidate = store.propose(GOOD, origin="seed")
    result = Funnel([compile_stage()]).screen(store, candidate, **context)

    assert result.errored and not result.passed
    assert store.get(candidate).outcome == "compile:errored"
    assert "HarnessError" in result.detail or "SpecError" in result.detail


def test_the_two_fault_tables_do_not_overlap():
    """Every runner status is either a verdict about the candidate or a
    problem with the harness, and never quietly both."""
    from kernelverify.compiler.stages import CANDIDATE_FAULTS, HARNESS_FAULTS
    from kernelverify.runners.result import RunStatus

    assert not (set(CANDIDATE_FAULTS) & set(HARNESS_FAULTS))
    assert set(CANDIDATE_FAULTS) | set(HARNESS_FAULTS) | {RunStatus.OK} == \
        set(RunStatus), "a new runner status must be classified, not defaulted"


def test_harness_error_is_what_the_stage_raises():
    assert issubclass(HarnessError, RuntimeError)


# ---------------------------------------------------------------------------
# The unwritten-output gate, as a funnel stage
# ---------------------------------------------------------------------------
HALF_WRITER = GOOD.replace("if (gid < n)", "if (gid < n / 2)")


@requires_metal
def test_the_unwritten_stage_kills_a_half_writing_kernel(store, context):
    from kernelverify.compiler.stages import unwritten_stage

    context["gate_cases"] = [context["probe_case"]]
    funnel = Funnel([compile_stage(), lint_stage(), unwritten_stage()])
    candidate = store.propose(HALF_WRITER, origin="generator")
    result = funnel.screen(store, candidate, **context)

    assert not result.passed and result.reached == "unwritten"
    assert "never written" in result.detail
    assert store.get(candidate).outcome == "unwritten:failed"


@requires_metal
def test_the_unwritten_stage_screens_the_source_the_store_recorded(store, context):
    """The stage builds its own spec from the positional source. A spec handed
    in through the context would let the gate judge text the journal never
    recorded, so there is no parameter for one."""
    import inspect

    from kernelverify.compiler.stages import unwritten_stage

    parameters = inspect.signature(unwritten_stage().run).parameters
    assert "spec" not in parameters
    assert list(parameters)[0] == "source"

    context["gate_cases"] = [context["probe_case"]]
    candidate = store.propose(GOOD, origin="seed")
    assert Funnel([unwritten_stage()]).screen(store, candidate, **context).passed
