"""The Metal runner: the contract, the numbers it returns, and the failures.

The failure tests carry as much weight as the numerical ones. A verifier's
runner exists to execute code that a language model wrote and nobody reviewed,
so "this kernel does not compile", "this kernel never returned" and "this
kernel took the worker down with it" all have to come back as statuses that a
battery run survives. Anything that escapes as an exception would abort a sweep
partway and leave the escape numbers half measured.

The spec tests need no GPU and run anywhere. Everything else is skipped when
the machine has no Metal device.
"""

import numpy as np
import pytest

from conftest import METAL_DEVICE as DEVICE, requires_metal
from kernelverify.runners import (
    Binding,
    BindingKind,
    KernelSpec,
    LaunchMode,
    LaunchSpec,
    MetalRunner,
    RunCase,
    RunStatus,
    SpecError,
    specialize,
)
from kernelverify.runners.spec import resolve_extent

RUNNER = MetalRunner()

RNG = np.random.default_rng(11)

# ---------------------------------------------------------------------------
# Kernels under test. Deliberately written the way a generator emits them:
# raw MSL, buffer indices in declaration order, dimensions passed as scalars.
# ---------------------------------------------------------------------------
GELU_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void gelu(device const float* x [[buffer(0)]],
                 device float* out     [[buffer(1)]],
                 constant uint& n      [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float v = x[gid];
    out[gid] = 0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
"""

HALF_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void scale(device const half* x [[buffer(0)]], device half* out [[buffer(1)]],
                  constant uint& n [[buffer(2)]], constant float& k [[buffer(3)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = (half)((float)x[gid] * k);
}
"""

# The same scaling kernel as a template: $T is the element type, and it names
# the entry point too, the way a generator emits one source per dtype family.
SCALE_TEMPLATE_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void scale_$T(device const $T* x [[buffer(0)]], device $T* out [[buffer(1)]],
                     constant uint& n [[buffer(2)]], constant float& k [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = ($T)((float)x[gid] * k);
}
"""

# Two outputs of different dtypes from one dispatch: the transport must hand
# each output binding its own shape and dtype and return them in binding order.
TWIN_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void twin(device const float* x [[buffer(0)]],
                 device float* doubled [[buffer(1)]],
                 device half*  halved  [[buffer(2)]],
                 constant uint& n      [[buffer(3)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    doubled[gid] = x[gid] + x[gid];
    halved[gid]  = (half)(0.5f * x[gid]);
}
"""

# One threadgroup per row, dynamically sized threadgroup scratch, tree
# reduction: the shape a real reduction kernel has, and the only test that
# exercises threadgroup memory and threadgroups-mode dispatch.
SOFTMAX_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void softmax_rows(device const float* x [[buffer(0)]],
                         device float* out     [[buffer(1)]],
                         constant uint& n_cols [[buffer(2)]],
                         threadgroup float* scratch [[threadgroup(0)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]],
                         uint width [[threads_per_threadgroup]]) {
    device const float* r = x + row * n_cols;
    float local = -INFINITY;
    for (uint i = lane; i < n_cols; i += width) local = max(local, r[i]);
    scratch[lane] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = width / 2; s > 0; s >>= 1) {
        if (lane < s) scratch[lane] = max(scratch[lane], scratch[lane + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float partial = 0.0f;
    for (uint i = lane; i < n_cols; i += width) partial += EXPFN(r[i] - m);
    scratch[lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = width / 2; s > 0; s >>= 1) {
        if (lane < s) scratch[lane] += scratch[lane + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denom = scratch[0];
    for (uint i = lane; i < n_cols; i += width) out[row * n_cols + i] = EXPFN(r[i] - m) / denom;
}
"""

MATMUL_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void matmul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                   device float* out [[buffer(2)]], constant uint& K [[buffer(3)]],
                   constant uint& N [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) acc += a[gid.y * K + k] * b[k * N + gid.x];
    out[gid.y * N + gid.x] = acc;
}
"""

# A bounded spin, used to make a kernel that is slow on purpose. The iteration
# count is a scalar so one compiled pipeline gives both a fast and a slow case;
# 40 million iterations is about 0.6s on an M3 Pro, far below any GPU watchdog.
SPIN_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void spin(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                 constant uint& n [[buffer(2)]], constant uint& iters [[buffer(3)]],
                 uint gid [[thread_position_in_grid]]) {
    float v = x[gid % n];
    for (uint i = 0; i < iters; ++i) { v = fma(v, 1.0000001f, 1e-7f); }
    if (gid < n) out[gid] = v;
}
"""

HALF_WRITTEN_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void evens(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                  constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n && gid % 2 == 0) out[gid] = x[gid] + 1.0f;
}
"""

SLOW_ITERS = 40_000_000  # about 0.6s: slow enough to hang, short of any watchdog
LIGHT_ITERS = 10_000  # 150 to 600us, depending on what else the GPU is doing
HEAVY_ITERS = 4_000_000  # about 58ms, well inside the reproducible regime


def elementwise_spec(source: str, entry: str, extra=()) -> KernelSpec:
    return KernelSpec(
        source=source,
        entry_point=entry,
        bindings=(Binding(BindingKind.INPUT, "input"), Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "n", "uint32")) + tuple(extra),
        launch=LaunchSpec(grid=("n", 1, 1), threadgroup=(128, 1, 1)),
    )


def vector_case(n: int, label: str = "", dtype: str = "float32") -> RunCase:
    data = (RNG.random(n, dtype=np.float32) * 20.0 - 10.0).astype(dtype)
    return RunCase(inputs={"input": data}, params={"n": n},
                   output_shapes=[((n,), dtype)], label=label or f"n={n}")


# ---------------------------------------------------------------------------
# The contract, checked without a GPU
# ---------------------------------------------------------------------------
def test_a_spec_needs_an_output_but_may_have_several():
    with pytest.raises(SpecError, match="no output binding"):
        KernelSpec(source=GELU_SRC, entry_point="gelu",
                   bindings=(Binding(BindingKind.INPUT, "input"),), launch=LaunchSpec())
    two = KernelSpec(source=TWIN_SRC, entry_point="twin",
                     bindings=(Binding(BindingKind.INPUT, "input"),
                               Binding(BindingKind.OUTPUT), Binding(BindingKind.OUTPUT),
                               Binding(BindingKind.SCALAR, "n", "uint32")),
                     launch=LaunchSpec(grid=("n", 1, 1), threadgroup=(128, 1, 1)))
    assert two.output_count() == 2


def test_a_case_must_describe_every_output_the_kernel_binds():
    spec = elementwise_spec(GELU_SRC, "gelu")  # one output
    case = RunCase(inputs={"input": np.zeros(4, np.float32)}, params={"n": 4},
                   output_shapes=[((4,), "float32"), ((4,), "float32")])
    with pytest.raises(SpecError, match="binds 1 outputs"):
        case.validate_against(spec)


def test_two_bindings_cannot_share_a_buffer_index():
    with pytest.raises(SpecError, match="bound twice"):
        KernelSpec(source=GELU_SRC, entry_point="gelu",
                   bindings=(Binding(BindingKind.INPUT, "input", index=0),
                             Binding(BindingKind.OUTPUT, index=0)),
                   launch=LaunchSpec())


def test_extents_are_literals_parameters_or_products():
    params = {"B": 2, "S": 7}
    assert resolve_extent(16, params, "grid") == 16
    assert resolve_extent("S", params, "grid") == 7
    assert resolve_extent(["B", "S", 2], params, "grid") == 28
    with pytest.raises(SpecError, match="no run parameter"):
        resolve_extent("H", params, "grid")


def test_a_case_missing_a_bound_name_says_which_one():
    spec = elementwise_spec(GELU_SRC, "gelu")
    case = RunCase(inputs={"input": np.zeros(4, np.float32)}, params={},
                   output_shapes=[((4,), "float32")])
    with pytest.raises(SpecError, match="scalar 'n'"):
        case.validate_against(spec)


def test_an_invalid_spec_is_caught_before_a_worker_is_started():
    """The interpreter path is nonsense, so a spawn would come back as a crash."""
    runner = MetalRunner(python="/definitely/not/an/interpreter")
    case = RunCase(inputs={"input": np.zeros(4, np.float32)}, params={},
                   output_shapes=[((4,), "float32")])
    result = runner.run(elementwise_spec(GELU_SRC, "gelu"), [case]).results[0]
    assert result.status is RunStatus.INVALID_SPEC


def test_a_json_round_trip_preserves_the_spec():
    spec = KernelSpec(
        source=SOFTMAX_SRC, entry_point="softmax_rows", name="softmax",
        bindings=(Binding(BindingKind.INPUT, "input"), Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "n_cols", "uint32")),
        launch=LaunchSpec(grid=(["B", "S"], 1, 1), threadgroup=(64, 1, 1),
                          mode=LaunchMode.THREADGROUPS, threadgroup_memory=("bytes",)),
    )
    assert KernelSpec.from_json(spec.to_json()) == spec


def test_a_json_round_trip_preserves_a_multi_output_case():
    case = RunCase(inputs={"input": RNG.random(8, dtype=np.float32)}, params={"n": 8},
                   output_shapes=[((8,), "float32"), ((2, 4), "float16")], label="twin")
    back = RunCase.from_json(case.to_json())
    assert back.output_shapes == case.output_shapes
    assert back.params == case.params and back.label == case.label
    np.testing.assert_array_equal(back.inputs["input"], case.inputs["input"])


def test_specialize_fills_the_placeholders_and_flags_both_mistakes():
    template = KernelSpec(
        source=SCALE_TEMPLATE_SRC, entry_point="scale_$T", name="scale-$T",
        bindings=(Binding(BindingKind.INPUT, "input"), Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "n", "uint32"),
                  Binding(BindingKind.SCALAR, "k", "float32")),
        launch=LaunchSpec(grid=("n", 1, 1), threadgroup=(128, 1, 1)),
    )
    spec = specialize(template, {"T": "half"})
    assert spec.entry_point == "scale_half"
    assert spec.name == "scale-half"
    assert "device const half* x" in spec.source
    assert spec.bindings == template.bindings and spec.launch == template.launch
    with pytest.raises(SpecError, match="do not supply"):
        specialize(template, {})
    with pytest.raises(SpecError, match="appear nowhere"):
        specialize(template, {"T": "half", "TILE": 8})


# ---------------------------------------------------------------------------
# Numerics on the device
# ---------------------------------------------------------------------------
@requires_metal
def test_an_elementwise_kernel_matches_numpy():
    spec = elementwise_spec(GELU_SRC, "gelu")
    cases = [vector_case(n) for n in (1, 3, 1025, 4096)]
    batch = RUNNER.run(spec, cases)
    assert batch.all_ok, [r.detail for r in batch if not r.ok]
    for case, result in zip(cases, batch):
        x = case.inputs["input"].astype(np.float64)
        want = 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
        np.testing.assert_allclose(result.outputs[0].astype(np.float64), want, atol=1e-6)


@requires_metal
def test_half_precision_tensors_round_trip():
    spec = elementwise_spec(HALF_SRC, "scale",
                            extra=(Binding(BindingKind.SCALAR, "k", "float32"),))
    case = vector_case(1000, dtype="float16")
    case.params["k"] = 2.5
    result = RUNNER.run_one(spec, case)
    assert result.ok, result.detail
    assert result.outputs[0].dtype == np.float16
    np.testing.assert_array_equal(
        result.outputs[0], (case.inputs["input"].astype(np.float32) * 2.5).astype(np.float16)
    )


@requires_metal
def test_two_outputs_come_back_in_binding_order_with_their_own_dtypes():
    spec = KernelSpec(
        source=TWIN_SRC, entry_point="twin",
        bindings=(Binding(BindingKind.INPUT, "input"),
                  Binding(BindingKind.OUTPUT), Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "n", "uint32")),
        launch=LaunchSpec(grid=("n", 1, 1), threadgroup=(128, 1, 1)),
    )
    n = 513
    x = RNG.random(n, dtype=np.float32) * 20.0 - 10.0
    case = RunCase(inputs={"input": x}, params={"n": n},
                   output_shapes=[((n,), "float32"), ((n,), "float16")])
    result = RUNNER.run_one(spec, case)
    assert result.ok, result.detail
    doubled, halved = result.outputs
    assert doubled.dtype == np.float32 and halved.dtype == np.float16
    np.testing.assert_array_equal(doubled, x + x)
    np.testing.assert_array_equal(halved, (0.5 * x).astype(np.float16))


@requires_metal
def test_a_row_reduction_uses_threadgroup_memory_and_threadgroup_dispatch():
    lanes = 64
    spec = KernelSpec(
        source=SOFTMAX_SRC.replace("EXPFN", "exp"), entry_point="softmax_rows",
        bindings=(Binding(BindingKind.INPUT, "input"), Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "n_cols", "uint32")),
        launch=LaunchSpec(grid=(["B", "S"], 1, 1), threadgroup=(lanes, 1, 1),
                          mode=LaunchMode.THREADGROUPS, threadgroup_memory=("scratch",)),
    )
    for batch_size, rows, cols in ((1, 1, 3), (2, 7, 256), (1, 3, 1025)):
        x = (RNG.random((batch_size, rows, cols), dtype=np.float32) * 20 - 10).astype(np.float32)
        case = RunCase(inputs={"input": x},
                       params={"B": batch_size, "S": rows, "n_cols": cols, "scratch": lanes * 4},
                       output_shapes=[(x.shape, "float32")], label=f"{batch_size}x{rows}x{cols}")
        result = RUNNER.run_one(spec, case)
        assert result.ok, result.detail
        exponent = np.exp(x.astype(np.float64) - x.astype(np.float64).max(-1, keepdims=True))
        np.testing.assert_allclose(result.outputs[0].astype(np.float64),
                                   exponent / exponent.sum(-1, keepdims=True), atol=1e-6)


@requires_metal
def test_a_two_dimensional_grid_dispatches_over_both_axes():
    spec = KernelSpec(
        source=MATMUL_SRC, entry_point="matmul",
        bindings=(Binding(BindingKind.INPUT, "a"), Binding(BindingKind.INPUT, "b"),
                  Binding(BindingKind.OUTPUT), Binding(BindingKind.SCALAR, "K", "uint32"),
                  Binding(BindingKind.SCALAR, "N", "uint32")),
        launch=LaunchSpec(grid=("N", "M", 1), threadgroup=(16, 16, 1)),
    )
    for m, k, n in ((1, 1, 1), (8, 32, 64), (64, 128, 64)):
        a = RNG.standard_normal((m, k), dtype=np.float32)
        b = RNG.standard_normal((k, n), dtype=np.float32)
        result = RUNNER.run_one(spec, RunCase(inputs={"a": a, "b": b},
                                              params={"M": m, "K": k, "N": n},
                                              output_shapes=[((m, n), "float32")]))
        assert result.ok, result.detail
        np.testing.assert_allclose(result.outputs[0].astype(np.float64),
                                   a.astype(np.float64) @ b.astype(np.float64), atol=1e-4)


@requires_metal
def test_the_output_buffers_start_zeroed():
    """A kernel that writes half its output must be judged on that, not on
    whatever the allocator last left in the buffer."""
    result = RUNNER.run_one(elementwise_spec(HALF_WRITTEN_SRC, "evens"), vector_case(512))
    assert result.ok, result.detail
    assert np.count_nonzero(result.outputs[0][1::2]) == 0


@requires_metal
def test_a_batch_returns_one_result_per_case_in_order():
    cases = [vector_case(n, label=f"case-{n}") for n in (7, 64, 4096, 3)]
    batch = RUNNER.run(elementwise_spec(GELU_SRC, "gelu"), cases)
    assert [r.label for r in batch] == [c.label for c in cases]
    assert [r.outputs[0].shape for r in batch] == [c.output_shapes[0][0] for c in cases]
    assert batch.device.name == DEVICE.name


# ---------------------------------------------------------------------------
# One candidate, several specializations, one worker session
# ---------------------------------------------------------------------------
@requires_metal
def test_the_specializations_of_one_candidate_share_one_worker_session():
    template = KernelSpec(
        source=SCALE_TEMPLATE_SRC, entry_point="scale_$T", name="scale-$T",
        bindings=(Binding(BindingKind.INPUT, "input"), Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "n", "uint32"),
                  Binding(BindingKind.SCALAR, "k", "float32")),
        launch=LaunchSpec(grid=("n", 1, 1), threadgroup=(128, 1, 1)),
    )
    batches = []
    for msl_type, dtype in (("float", "float32"), ("half", "float16")):
        case = vector_case(512, dtype=dtype, label=msl_type)
        case.params["k"] = 3.0
        batches.append((specialize(template, {"T": msl_type}), [case]))

    runner = MetalRunner()
    spawned = []
    original = runner._spawn
    runner._spawn = lambda probe=False: (spawned.append(1), original(probe=probe))[1]
    results = runner.run_candidate(batches)

    assert len(spawned) == 1, "one candidate must cost one worker process"
    for (spec, cases), batch in zip(batches, results):
        assert batch.all_ok, [r.detail for r in batch if not r.ok]
        shape, dtype = cases[0].output_shapes[0]
        want = (cases[0].inputs["input"].astype(np.float32) * 3.0).astype(dtype)
        np.testing.assert_array_equal(batch.results[0].outputs[0], want)


@requires_metal
def test_a_spec_that_fails_to_compile_does_not_sink_its_siblings():
    broken = GELU_SRC.replace("float v = x[gid];", "float v = ;")
    bad, good = RUNNER.run_candidate([
        (elementwise_spec(broken, "gelu"), [vector_case(16)]),
        (elementwise_spec(GELU_SRC, "gelu"), [vector_case(16)]),
    ])
    assert bad.results[0].status is RunStatus.COMPILE_ERROR
    assert "error" in bad.results[0].detail
    assert good.all_ok, [r.detail for r in good if not r.ok]


# ---------------------------------------------------------------------------
# Timing, which is what the optimiser phase gates on
# ---------------------------------------------------------------------------
@requires_metal
def test_timing_separates_a_heavy_kernel_from_a_light_one():
    """The separation is asserted; the noise floor is not, because it moves.

    Measured on a busy M3 Pro, a 14.5 millisecond kernel repeats to within
    0.1 percent, 24 samples between 14495 and 14505 microseconds. The same
    measurement at 200 microseconds drifts 1.35x within one run and up to 4x
    between runs as the GPU changes power state. So the only claim this test
    makes about a short kernel is that a long one is clearly slower than it,
    and the tightness assertion is made where tightness exists.

    Two earlier versions of this test failed in the suite by asserting a spread
    bound on a short kernel. Both failures were the hardware telling the truth,
    which is why the finding went into ADR 0006 instead of into a wider
    threshold.
    """
    spec = elementwise_spec(SPIN_SRC, "spin",
                            extra=(Binding(BindingKind.SCALAR, "iters", "uint32"),))
    results = []
    for iters in (LIGHT_ITERS, HEAVY_ITERS):
        case = vector_case(64, label=f"iters={iters}")
        case.params["iters"] = iters
        results.append(RUNNER.run(spec, [case], warmup=1, repeats=3).results[0])
    light, heavy = results
    assert light.ok and heavy.ok
    assert light.timing.repeats == 3
    assert heavy.timing.gpu_best > 20 * light.timing.gpu_best
    assert heavy.timing.gpu_spread < 1.25, "milliseconds of work should measure cleanly"
    assert heavy.timing.wall_median >= heavy.timing.gpu_median
    assert light.timing.gpu_spread >= 1.0  # reported for every run, believed selectively


# ---------------------------------------------------------------------------
# Failures, which are the reason the worker is a separate process
# ---------------------------------------------------------------------------
@requires_metal
def test_source_that_does_not_compile_is_a_status_with_the_diagnostic():
    broken = GELU_SRC.replace("float v = x[gid];", "float v = ;")
    result = RUNNER.run_one(elementwise_spec(broken, "gelu"), vector_case(16))
    assert result.status is RunStatus.COMPILE_ERROR
    assert "error" in result.detail


@requires_metal
def test_a_missing_entry_point_reports_the_functions_that_do_exist():
    result = RUNNER.run_one(elementwise_spec(GELU_SRC, "not_the_name"), vector_case(16))
    assert result.status is RunStatus.COMPILE_ERROR
    assert "gelu" in result.detail


@requires_metal
def test_an_oversized_threadgroup_is_a_launch_error_not_an_abort():
    """Metal treats this as a programmer error and kills the process, so the
    runner has to check it against the compiled pipeline first."""
    spec = elementwise_spec(GELU_SRC, "gelu")
    oversized = KernelSpec(source=spec.source, entry_point=spec.entry_point,
                           bindings=spec.bindings,
                           launch=LaunchSpec(grid=("n", 1, 1), threadgroup=(4096, 1, 1)))
    result = RUNNER.run_one(oversized, vector_case(4096))
    assert result.status is RunStatus.LAUNCH_ERROR
    assert "4096" in result.detail


@requires_metal
def test_a_worker_that_dies_becomes_a_crash_status():
    runner = MetalRunner(python="/usr/bin/false")
    result = runner.run_one(elementwise_spec(GELU_SRC, "gelu"), vector_case(16))
    assert result.status is RunStatus.CRASH


@requires_metal
def test_a_hung_kernel_is_killed_and_the_batch_resumes():
    """A hang costs the case that hung, never the rest of the candidate.

    The hung case is killed at the case budget and recorded as a timeout, a
    fresh worker resumes from the case after it, and the candidate's next
    spec still runs. [Ported from the retired runner, which pioneered the
    resume; its version asserted a follow-up batch, this one asserts the
    resumed cases inside the same candidate.]
    """
    spec = elementwise_spec(SPIN_SRC, "spin",
                            extra=(Binding(BindingKind.SCALAR, "iters", "uint32"),))
    cases = []
    for iters, label in ((1, "fast"), (SLOW_ITERS, "hangs"), (1, "after the hang")):
        case = vector_case(64, label=label)
        case.params["iters"] = iters
        cases.append(case)

    runner = MetalRunner(case_timeout=0.25, startup_timeout=10.0, warmup=0, repeats=1)
    spin_batch, gelu_batch = runner.run_candidate([
        (spec, cases),
        (elementwise_spec(GELU_SRC, "gelu"), [vector_case(64, label="next spec")]),
    ])
    assert [r.status for r in spin_batch] == [
        RunStatus.OK, RunStatus.TIMEOUT, RunStatus.OK]
    assert spin_batch.results[0].outputs, "a measured case was lost to a later hang"
    assert spin_batch.results[2].outputs, "the batch did not resume after the hang"
    assert gelu_batch.all_ok, "the candidate's next spec was lost to the hang"

    # The GPU is still usable afterwards, which is the point of killing the
    # process group rather than waiting for the kernel to finish.
    assert RUNNER.run_one(elementwise_spec(GELU_SRC, "gelu"), vector_case(64)).ok


# ---------------------------------------------------------------------------
# The runner meeting the shipped oracle
# ---------------------------------------------------------------------------
@requires_metal
def test_a_metal_kernel_is_judged_by_the_shipped_oracle():
    """A real GPU kernel against the corpus fp64 reference and the ADR 0004
    tolerance, which was calibrated entirely on numpy implementations.

    This is the claim worth pinning. The ensemble floor bounds what a correct
    working-precision implementation may deviate, and its members are all
    numpy: pairwise and sequential accumulation. This kernel reduces in a
    different structure again, a strided tree across 64 threadgroup lanes, so
    it is the first evidence that the floor covers an implementation the
    calibration never saw. Measured margin on these cases: the correct kernel
    errs 1.9e-9 to 3.0e-8 against a 1e-5 tolerance.

    The fault case is the tl.exp2 confusion, computing 2^x where the reference
    computes e^x. It is caught everywhere the mechanism can express, and on
    constant-rows input it cannot: every element of a row is identical, so the
    softmax is 1/H under any exponential base and the two kernels agree
    exactly. That is ADR 0001's thesis in one row, so it is asserted as an
    escape rather than papered over.
    """
    from measure_escape import corpus_oracle_passes, load_meta, max_abs_error, reference
    from score_oracles import SWEEP_SEED, make_mode_inputs

    from kernelverify.tolerance.floor import conditioned_tolerance

    lanes = 64

    def softmax_spec(exponential: str) -> KernelSpec:
        return KernelSpec(
            source=SOFTMAX_SRC.replace("EXPFN", exponential), entry_point="softmax_rows",
            bindings=(Binding(BindingKind.INPUT, "input"), Binding(BindingKind.OUTPUT),
                      Binding(BindingKind.SCALAR, "n_cols", "uint32")),
            launch=LaunchSpec(grid=(["B", "S"], 1, 1), threadgroup=(lanes, 1, 1),
                              mode=LaunchMode.THREADGROUPS, threadgroup_memory=("scratch",)),
        )

    op = "softmax_triton"
    meta = load_meta(op)
    base = float(meta["tolerances"]["float32"])
    cases = [
        ({"B": 2, "S": 7, "H": 256}, "corpus uniform[-10,10]", True),
        ({"B": 2, "S": 7, "H": 3}, "near zero", True),
        ({"B": 8, "S": 1, "H": 1025}, "opposed signs", True),
        ({"B": 1, "S": 7, "H": 256}, "constant rows", False),
    ]

    for dims, mode, fault_expressible in cases:
        inputs = make_mode_inputs(meta, dims, "float32", SWEEP_SEED, mode)
        ref = reference(meta, inputs, (op, tuple(sorted(dims.items())), "float32",
                                       SWEEP_SEED, mode))
        tolerance = conditioned_tolerance(op, inputs, ref, base)
        case = RunCase(inputs=inputs,
                       params={"B": dims["B"], "S": dims["S"], "n_cols": dims["H"],
                               "scratch": lanes * 4},
                       output_shapes=[(inputs["input"].shape, "float32")],
                       label=f"{dims} {mode}")

        correct = RUNNER.run_one(softmax_spec("exp"), case)
        assert correct.ok, correct.detail
        assert corpus_oracle_passes(correct.outputs[0], ref, tolerance), (
            f"the shipped tolerance false-positives on a correct Metal kernel at "
            f"{case.label}: error {max_abs_error(correct.outputs[0], ref):.3e} "
            f"against tolerance {tolerance:.3e}"
        )

        faulty = RUNNER.run_one(softmax_spec("exp2"), case)
        assert faulty.ok, faulty.detail
        caught = not corpus_oracle_passes(faulty.outputs[0], ref, tolerance)
        assert caught is fault_expressible, (
            f"exp2 fault at {case.label}: expected "
            f"{'a detection' if fault_expressible else 'an escape'}, "
            f"error {max_abs_error(faulty.outputs[0], ref):.3e} vs {tolerance:.3e}"
        )
