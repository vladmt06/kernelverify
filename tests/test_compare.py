"""compare(): interleaved A/B between two live workers, canaried by arm A.

The speedup test uses two specializations of the same spin kernel at a known
iteration ratio, sized in the tens-of-milliseconds regime where ADR 0006
measured ratios to be exactly reproducible. The rejection test does not try
to make the machine noisy on cue; it tightens the spread limit until honest
samples cannot pass it, which exercises the rejection path deterministically.
"""

import numpy as np
import pytest

from kernelverify.runners import (
    Binding,
    BindingKind,
    KernelSpec,
    LaunchSpec,
    MetalRunner,
    RunCase,
    compare,
    specialize,
)

RUNNER = MetalRunner()
DEVICE = RUNNER.probe()
requires_metal = pytest.mark.skipif(DEVICE is None, reason="no Metal device on this machine")

# The iteration count is baked into the source so the two arms are genuinely
# different compiled pipelines, the way a real A/B compares two candidates.
SPIN_TEMPLATE_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void spin(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                 constant uint& n [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    float v = x[gid % n];
    for (uint i = 0; i < ${ITERS}u; ++i) { v = fma(v, 1.0000001f, 1e-7f); }
    if (gid < n) out[gid] = v;
}
"""

HEAVY_ITERS = 2_000_000  # ~29 ms on the M3 Pro: inside the reproducible regime
LIGHT_ITERS = 500_000    # ~7 ms, same regime; the true ratio is exactly 4.0


def spin_spec(iters: int) -> KernelSpec:
    template = KernelSpec(
        source=SPIN_TEMPLATE_SRC, entry_point="spin", name="spin-$ITERS",
        bindings=(Binding(BindingKind.INPUT, "input"), Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "n", "uint32")),
        launch=LaunchSpec(grid=("n", 1, 1), threadgroup=(64, 1, 1)),
    )
    return specialize(template, {"ITERS": iters})


def spin_case(n: int = 64) -> RunCase:
    return RunCase(inputs={"input": np.ones(n, dtype=np.float32)}, params={"n": n},
                   output_shapes=[((n,), "float32")], label=f"n={n}")


@requires_metal
def test_compare_reports_the_speedup_at_a_known_ratio():
    report = compare(spin_spec(HEAVY_ITERS), spin_spec(LIGHT_ITERS), [spin_case()],
                     rounds=2, warmup=1, repeats=2)
    assert report.ok, report.summary()
    case = report.cases[0]
    assert len(case.accepted) == 2 and case.rejected == 0
    assert 3.0 < case.speedup < 5.0, report.summary()
    assert "speedup" in report.summary()


@requires_metal
def test_a_round_the_reference_arm_cannot_back_is_rejected_and_fails_the_run():
    """An honest pair of GPU samples never agrees to five decimal places, so a
    limit of 1.00001x rejects every round; D6 requires that a rejected round
    fails the comparison rather than being dropped in silence."""
    report = compare(spin_spec(LIGHT_ITERS), spin_spec(LIGHT_ITERS), [spin_case()],
                     rounds=2, warmup=1, repeats=3, spread_limit=1.00001)
    assert not report.ok
    assert report.rejected_rounds == 2
    assert "REJECTED" in report.summary()
    # The evidence survives rejection: the samples are recorded and labelled.
    assert len(report.cases[0].samples) == 2
    assert all(not s.accepted for s in report.cases[0].samples)


@requires_metal
def test_an_arm_that_cannot_compile_aborts_the_comparison():
    broken = spin_spec(LIGHT_ITERS)
    broken = KernelSpec(source=broken.source.replace("float v", "float v = ;"),
                        entry_point="spin", bindings=broken.bindings,
                        launch=broken.launch)
    report = compare(spin_spec(LIGHT_ITERS), broken, [spin_case()], rounds=1)
    assert not report.ok
    assert "B (candidate)" in report.failure and "compile" in report.failure


def test_an_invalid_case_fails_before_any_worker_starts():
    runner = MetalRunner(python="/definitely/not/an/interpreter")
    case = RunCase(inputs={"input": np.ones(4, np.float32)}, params={},
                   output_shapes=[((4,), "float32")])
    report = compare(spin_spec(1), spin_spec(2), [case], runner=runner)
    assert not report.ok
    assert "scalar 'n'" in report.failure
