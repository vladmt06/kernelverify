"""Eight mutants, each caught by the stage that claims it, and by no other.

A funnel of stages is only worth its order if each stage earns its place. Two
things have to be true and neither is obvious: every named fault class is
actually caught, and it is caught by the stage that claims it rather than
falling into an earlier one by accident.

The second half is what makes this more than a smoke test. A half-precision
accumulator caught at compile would prove nothing about the lint, so every
lint mutant must be shown to COMPILE first, and every unwritten-output mutant
must be shown to compile and lint clean. The `reached` field is the assertion:
it names the stage that stopped the candidate, so a mutant caught early fails
this file loudly rather than quietly flattering the funnel.

The control is the ninth: the same kernel with no fault must pass all three,
or the battery is measuring a broken harness rather than eight faults.
"""

import numpy as np
import pytest

from conftest import requires_metal
from kernelverify.compiler.funnel import Funnel
from kernelverify.compiler.stages import compile_stage, lint_stage, unwritten_stage
from kernelverify.compiler.store import CandidateStore
from kernelverify.runners import MetalRunner
from kernelverify.runners.spec import Binding, BindingKind, LaunchSpec, RunCase

N = 64

SHELL = """
[[kernel]] void f(device const float* x [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  constant uint& n [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
BODY
}
"""

CORRECT = "    if (gid < n) { float acc = x[gid] * 2.0f; out[gid] = acc; }"

# Named fault class per stage. The stage name here is a claim this file tests,
# not a label it trusts.
MUTANTS = [
    ("compile", "malformed expression",
     "    if (gid < n) { float acc = ; out[gid] = acc; }"),
    ("compile", "calls a function that does not exist",
     "    if (gid < n) { float acc = definitely_not_a_builtin(x[gid]); out[gid] = acc; }"),

    ("lint", "half accumulator",
     "    if (gid < n) { half acc = (half)x[gid] * 2.0h; out[gid] = (float)acc; }"),
    ("lint", "half vector tile",
     "    half4 tile = half4(0.0h);\n"
     "    if (gid < n) { tile.x = (half)x[gid]; out[gid] = (float)tile.x * 2.0f; }"),
    ("lint", "half scalar holding an intermediate",
     "    if (gid < n) { half narrow = (half)(x[gid] * 2.0f); out[gid] = (float)narrow; }"),

    ("unwritten", "bound halved, tail never written",
     "    if (gid < n / 2) { float acc = x[gid] * 2.0f; out[gid] = acc; }"),
    ("unwritten", "early return on odd lanes",
     "    if (gid % 2 == 1) return;\n"
     "    if (gid < n) { float acc = x[gid] * 2.0f; out[gid] = acc; }"),
    ("unwritten", "off by one at the very last cell",
     "    if (gid + 1 < n) { float acc = x[gid] * 2.0f; out[gid] = acc; }"),
]


def _source(body: str) -> str:
    return SHELL.replace("BODY", body)


@pytest.fixture(scope="module")
def runner():
    return MetalRunner()


@pytest.fixture()
def context(runner):
    case = RunCase(inputs={"x": np.arange(1, N + 1, dtype=np.float32)},
                   params={"n": N}, output_shapes=[((N,), "float32")],
                   label="battery")
    return {"runner": runner, "entry_point": "f",
            "bindings": (Binding(BindingKind.INPUT, "x"),
                         Binding(BindingKind.OUTPUT),
                         Binding(BindingKind.SCALAR, "n", "uint32")),
            "launch": LaunchSpec(grid=("n", 1, 1), threadgroup=(32, 1, 1)),
            "probe_case": case, "gate_cases": [case]}


@pytest.fixture()
def funnel():
    return Funnel([compile_stage(), lint_stage(), unwritten_stage()])


@pytest.fixture()
def store(tmp_path):
    return CandidateStore(tmp_path)


@requires_metal
def test_the_control_passes_every_stage(store, funnel, context):
    """Without this the battery proves nothing: eight refusals from a harness
    that refuses everything is not eight detections."""
    result = funnel.screen(store, store.propose(_source(CORRECT), origin="seed"),
                           **context)
    assert result.passed, result.detail


@requires_metal
@pytest.mark.parametrize("stage,fault,body", MUTANTS,
                         ids=[f"{s}:{f}" for s, f, _ in MUTANTS])
def test_each_mutant_is_caught_by_the_stage_that_claims_it(store, funnel, context,
                                                            stage, fault, body):
    candidate = store.propose(_source(body), origin=f"mutant:{fault}")
    result = funnel.screen(store, candidate, **context)

    assert not result.passed, f"{fault} was not caught at all"
    assert not result.errored, f"{fault} crashed a stage instead of being judged"
    assert result.reached == stage, (
        f"{fault} is a {stage} fault but was stopped at {result.reached}; a "
        f"mutant caught early proves nothing about the stage that claims it")
    assert store.get(candidate).outcome == f"{stage}:failed"


@requires_metal
def test_the_lint_mutants_all_compile_so_the_lint_is_doing_the_catching(store,
                                                                        context):
    """The point of a source attestation. Every one of these is valid Metal,
    so a compile-only funnel passes all three."""
    only_compile = Funnel([compile_stage()])
    for stage, fault, body in MUTANTS:
        if stage != "lint":
            continue
        result = only_compile.screen(store, store.propose(_source(body),
                                                          origin=fault), **context)
        assert result.passed, f"{fault} failed to compile: {result.detail}"


@requires_metal
def test_the_unwritten_mutants_compile_and_lint_clean(store, context):
    """Likewise for the gate: these are valid Metal with no narrow arithmetic
    anywhere, so nothing ahead of the gate has any objection to them."""
    ahead = Funnel([compile_stage(), lint_stage()])
    for stage, fault, body in MUTANTS:
        if stage != "unwritten":
            continue
        result = ahead.screen(store, store.propose(_source(body), origin=fault),
                              **context)
        assert result.passed, f"{fault} was stopped before the gate: {result.detail}"


@requires_metal
def test_the_gate_counts_what_each_unwritten_mutant_actually_missed(store, funnel,
                                                                    context):
    """The counts are the evidence that the gate saw the real fault and not
    some other one: half the cells, the odd lanes, and exactly one cell."""
    expected = {"bound halved, tail never written": N // 2,
                "early return on odd lanes": N // 2,
                "off by one at the very last cell": 1}
    for stage, fault, body in MUTANTS:
        if stage != "unwritten":
            continue
        result = funnel.screen(store, store.propose(_source(body), origin=fault),
                               **context)
        assert f"{expected[fault]} cells never written" in result.detail, (
            f"{fault}: {result.detail}")


@requires_metal
def test_the_census_of_the_whole_battery_is_three_stages_and_a_survivor(store,
                                                                        funnel,
                                                                        context):
    funnel.screen(store, store.propose(_source(CORRECT), origin="seed"), **context)
    for stage, fault, body in MUTANTS:
        funnel.screen(store, store.propose(_source(body), origin=fault), **context)

    assert store.census() == {"compile:failed": 2, "lint:failed": 3,
                              "unwritten:failed": 3, "unwritten:passed": 1}
