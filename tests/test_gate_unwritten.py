"""The unwritten-output screen, and the two attacks that shaped it.

An adversarial review of the first design found both a false positive and a
miss, measured on this machine. Both survive here as tests, because a fix
without a test that would have caught the original is a fix on trust.

The false positive was a kernel that accumulates into its own output. The
tolerance contract grants that (a reduction blocked across threadgroups can
only be assembled through the output buffer on Metal), and the first design
refused it. The gate now abstains on it instead, which costs coverage and
costs no promise.

The miss was a kernel that clears its own output and then applies a wrong
bound. Every cell receives a store, so the gate passes it, and no choice of
fill pattern changes that: the mechanism can only see "no store reached this
cell". That is now what the policy string claims, and the test below pins the
miss so nobody later reads a clean verdict as more than it is.
"""

import numpy as np
import pytest

from conftest import requires_metal
from kernelverify.compiler.unwritten import (
    CLEAN,
    NO_RUN,
    POLICY,
    PREFILL_DEPENDENT,
    UNWRITTEN,
    screen,
)
from kernelverify.runners import MetalRunner
from kernelverify.runners.spec import (
    Binding,
    BindingKind,
    KernelSpec,
    LaunchSpec,
    RunCase,
    SpecError,
)

N = 64

# The shader's element type has to match the case's dtype, or the kernel
# writes the wrong number of bytes and the gate is measuring the test's bug.
MSL = {"float32": ("float", "2.0f", "0.0f"), "float16": ("half", "2.0h", "0.0h"),
       "uint32": ("uint", "2u", "0u"), "int32": ("int", "2", "0")}

BODY = """
[[kernel]] void f(device const T* x [[buffer(0)]],
                  device T* out [[buffer(1)]],
                  constant uint& n [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    BODY_HERE
}
"""


def source(body: str, dtype: str = "float32") -> str:
    element, two, zero = MSL[dtype]
    return (BODY.replace("BODY_HERE", body).replace("T*", f"{element}*")
            .replace("TWO", two).replace("ZERO", zero))


FULL = "if (gid < n) out[gid] = x[gid] * TWO;"
HALF = "if (gid < n / 2) out[gid] = x[gid] * TWO;"
# Writes NaN into every cell: a shipped kernel in this repo does exactly this
# on its overflow path, and a NaN-sentinel gate would refuse it.
ALL_NAN = "if (gid < n) out[gid] = NAN;"
# Clears its own output, then applies a wrong bound. Every cell is stored to.
CLEAR_THEN_HALF = "out[gid] = ZERO; if (gid < n / 2) out[gid] = x[gid] * TWO;"
# Reads its own output, which the contract grants.
ACCUMULATES = "if (gid < n) out[gid] += x[gid] * TWO;"


def _spec(source: str) -> KernelSpec:
    return KernelSpec(
        source=source, entry_point="f",
        bindings=(Binding(BindingKind.INPUT, "x"), Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "n", "uint32")),
        launch=LaunchSpec(grid=("n", 1, 1), threadgroup=(32, 1, 1)))


def _case(label="c", dtype="float32"):
    return RunCase(inputs={"x": np.arange(1, N + 1, dtype=dtype)},
                   params={"n": N}, output_shapes=[((N,), dtype)], label=label)


@pytest.fixture(scope="module")
def runner():
    return MetalRunner()


# ---------------------------------------------------------------------------
# The contract on RunCase
# ---------------------------------------------------------------------------
def test_the_prefill_defaults_to_zero_so_every_existing_case_is_unchanged():
    assert _case().prefill == "zero"


def test_an_unknown_prefill_is_refused_by_the_case_itself():
    with pytest.raises(SpecError, match="prefill"):
        RunCase(inputs={}, output_shapes=[((4,), "float32")], prefill="0xFF")


def test_the_prefill_crosses_the_process_boundary():
    """The worker rebuilds cases from JSON, so a field that does not survive
    the round trip would silently make the gate screen nothing."""
    case = RunCase(inputs={"x": np.zeros(4, np.float32)},
                   output_shapes=[((4,), "float32")], prefill="sentinel_a")
    assert RunCase.from_json(case.to_json()).prefill == "sentinel_a"


def test_an_old_wire_form_without_a_prefill_still_loads_as_zero():
    raw = RunCase(inputs={"x": np.zeros(4, np.float32)},
                  output_shapes=[((4,), "float32")]).to_json()
    del raw["prefill"]
    assert RunCase.from_json(raw).prefill == "zero"


# ---------------------------------------------------------------------------
# What it catches
# ---------------------------------------------------------------------------
@requires_metal
def test_a_kernel_that_writes_everything_is_clean(runner):
    report = screen(runner, _spec(source(FULL)), [_case()])
    assert report.ok and report.cases[0].verdict == CLEAN
    assert report.screened == 1


@requires_metal
def test_a_kernel_that_writes_half_is_refused_with_a_count_and_a_coordinate(runner):
    report = screen(runner, _spec(source(HALF)), [_case()])
    assert not report.ok
    [case] = report.cases
    assert case.verdict == UNWRITTEN
    assert case.unwritten_cells == N // 2
    assert case.first[0] == (N // 2,), "the first unwritten cell is the tail's start"
    assert "never written" in report.reason


@requires_metal
@pytest.mark.parametrize("dtype", ["float32", "float16", "uint32", "int32"])
def test_the_screen_works_on_every_carrier_not_only_the_float_ones(runner, dtype):
    """Half the tensor dtypes are integer carriers where no value is out of
    range, which is why the fill is a byte pattern and the comparison runs
    through an unsigned view."""
    assert screen(runner, _spec(source(FULL, dtype)), [_case(dtype=dtype)]).ok
    assert not screen(runner, _spec(source(HALF, dtype)), [_case(dtype=dtype)]).ok


@requires_metal
def test_a_kernel_whose_correct_output_is_all_nan_is_clean(runner):
    """A NaN-sentinel gate would refuse this. A shipped kernel in this
    repository writes NAN into every output cell on its overflow path, and the
    next gate in the funnel runs kernels whose correct output is NaN."""
    assert screen(runner, _spec(source(ALL_NAN)), [_case()]).ok


# ---------------------------------------------------------------------------
# The false positive the review found: abstain, do not refuse
# ---------------------------------------------------------------------------
@requires_metal
def test_a_kernel_that_accumulates_into_its_output_is_abstained_not_refused(runner):
    """The contract grants blocking a reduction at any width, and on Metal a
    reduction blocked across threadgroups can only be assembled through the
    output buffer. The first design refused this class. Refusing it would have
    been the gate deciding a contract question by accident."""
    report = screen(runner, _spec(source(ACCUMULATES)), [_case()])

    [case] = report.cases
    assert case.verdict == PREFILL_DEPENDENT
    assert report.ok, "an abstention must not refuse the candidate"
    assert report.screened == 0, "and must not be counted as coverage either"
    assert "reads its own output" in case.detail
    assert "not screened" in report.reason


# ---------------------------------------------------------------------------
# The miss the review found: named, not hidden
# ---------------------------------------------------------------------------
@requires_metal
def test_a_kernel_that_clears_then_writes_half_passes_and_that_is_the_scope(runner):
    """Every cell receives a store, so this passes, and no fill pattern
    changes that. The gate attests that a store reached a cell, never that the
    kernel produced its value. Pinned so a clean verdict is never read as
    coverage of the wrong-bounds class."""
    assert screen(runner, _spec(source(CLEAR_THEN_HALF)), [_case()]).ok


def test_the_policy_states_that_scope_rather_than_implying_coverage():
    assert "NOT that the kernel produced that cell's value" in POLICY
    assert "tolerance gate's to catch" in POLICY


# ---------------------------------------------------------------------------
# A run that did not happen claims nothing
# ---------------------------------------------------------------------------
@requires_metal
def test_a_kernel_that_does_not_compile_yields_no_unwritten_claim(runner):
    report = screen(runner, _spec(FULL.replace("x[gid] * 2.0f", "")), [_case()])
    [case] = report.cases
    assert case.verdict == NO_RUN
    assert report.ok and report.screened == 0, (
        "a compile failure is the compile stage's finding, not this gate's")
