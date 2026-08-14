"""The Metal runner walking skeleton: compile, execute, agree, fail loudly.

These tests exercise every isolation path the eng review committed to
(decisions 1A + 5A): protocol round-trip against the numpy reference,
compile errors surfaced as case failures, a hung kernel killed at the
case timeout with the batch resuming, and batching semantics (one compile,
many cases). Skipped wholesale when MLX/Metal is unavailable so the suite
stays green on non-Mac CI.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal unavailable", allow_module_level=True)

from kernelverify.reference.kernels import silu
from kernelverify.runners.runner import CaseResult, DeviceCase, KernelSpec, MetalRunner

SILU_MSL = """
    uint i = thread_position_in_grid.x;
    if (i < x_shape[0]) {
        float v = (float)x[i];
        out[i] = (T)(v / (1.0f + metal::exp(-v)));
    }
"""


def silu_kernel() -> KernelSpec:
    return KernelSpec(name="kv_silu", source=SILU_MSL,
                      input_names=["x"], output_names=["out"])


def silu_case(n: int, dtype: str, seed: int) -> DeviceCase:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(dtype)
    return DeviceCase(inputs={"x": x}, output_shapes=[((n,), dtype)],
                      grid=(n, 1, 1), threadgroup=(min(n, 256), 1, 1),
                      template={"T": dtype})


def test_round_trip_agrees_with_numpy_reference():
    runner = MetalRunner()
    cases = [silu_case(n, dt, seed)
             for n in (7, 256, 4097) for dt in ("float32", "float16")
             for seed in (0, 1)]
    results = runner.run(silu_kernel(), cases)
    assert all(r.ok for r in results)
    for case, result in zip(cases, results):
        expected = silu({"input": case.inputs["x"]})
        atol = 1e-6 if case.inputs["x"].dtype == np.float32 else 2e-3
        np.testing.assert_allclose(result.outputs[0].astype(np.float64),
                                   expected.astype(np.float64), atol=atol)


def test_compile_error_is_a_case_failure_not_a_crash():
    broken = KernelSpec(name="kv_broken", source="this is not MSL;",
                        input_names=["x"], output_names=["out"])
    results = MetalRunner().run(broken, [silu_case(16, "float32", 0)])
    assert len(results) == 1 and not results[0].ok
    assert results[0].error


def test_hung_kernel_is_killed_and_batch_resumes():
    hang = KernelSpec(
        name="kv_hang",
        source="""
        uint i = thread_position_in_grid.x;
        float acc = (float)x[0];
        volatile int guard = 0;
        while (guard < 2000000000) { acc += metal::sin(acc); guard++; }
        out[i] = (T)acc;
        """,
        input_names=["x"], output_names=["out"])
    good = silu_kernel()
    runner = MetalRunner(case_timeout=10.0)
    hung = runner.run(hang, [silu_case(64, "float32", 0)])
    assert not hung[0].ok
    follow_up = runner.run(good, [silu_case(64, "float32", 1)])
    assert follow_up[0].ok


def test_batch_streams_all_cases_through_one_worker():
    cases = [silu_case(64, "float32", s) for s in range(8)]
    results = MetalRunner().run(silu_kernel(), cases)
    assert [r.ok for r in results] == [True] * 8
