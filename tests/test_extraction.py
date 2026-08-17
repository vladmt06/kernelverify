"""Extraction: capture the MSL MLX runs, rebind it, and prove the match.

The unit tests pin the parsing and the alarm logic with no GPU and no MLX.
The E2E tests need both: they capture a real specialization in a fresh
process, run the extracted source under the Metal runner, run the surface
natively in the live arm, and judge the two arms with the shipped oracle's
verdict function across structured input modes - the exact criterion the
consolidation dispatch names. Never bitwise.
"""

import importlib.util

import numpy as np
import pytest

from conftest import METAL_DEVICE
from kernelverify.extraction import (
    ExtractionError,
    LiveCall,
    MLXKernelSurface,
    capture_specialization,
    extracted_spec,
    metadata_inputs,
    run_live,
    signature_bindings,
    validate_extraction,
)
from kernelverify.extraction.mlx_arm import LiveResult, parse_dump
from kernelverify.runners import (
    BindingKind,
    LaunchSpec,
    MetalRunner,
    RunCase,
    RunResult,
    RunStatus,
)

RUNNER = MetalRunner()
HAS_MLX = importlib.util.find_spec("mlx") is not None
requires_arms = pytest.mark.skipif(
    METAL_DEVICE is None or not HAS_MLX,
    reason="extraction E2E needs a Metal device and mlx",
)

# The dump format of mlx 0.32.0, verbatim from a probe run.
CANNED_DUMP = """Generated source code for `probe`:
```
template <typename T>
[[kernel]] void custom_kernel_probe__float_floatc_float(
  const constant float* inp [[buffer(0)]],
  device float* out [[buffer(1)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
uint e = thread_position_in_grid.x; out[e] = (T)inp[e];
}

template [[host_name("custom_kernel_probe__float_floatc_float")]] [[kernel]] decltype(custom_kernel_probe__float_floatc_float<float>) custom_kernel_probe__float_floatc_float<float>;

```
"""

METADATA_SIGNATURE = """
[[kernel]] void custom_kernel_stride_probe_float_float(
  const device float* x [[buffer(0)]],
  const constant int* x_shape [[buffer(1)]],
  const constant int64_t* x_strides [[buffer(2)]],
  device float* out [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {}
"""


# ---------------------------------------------------------------------------
# Parsing, no GPU and no MLX
# ---------------------------------------------------------------------------
def test_the_dump_parser_finds_the_source_and_the_host_name():
    blocks = parse_dump(CANNED_DUMP)
    assert len(blocks) == 1
    name, source, entry = blocks[0]
    assert name == "probe"
    assert entry == "custom_kernel_probe__float_floatc_float"
    assert "[[kernel]] void" in source and "Generated source" not in source


def test_the_signature_parser_classifies_metadata_by_the_measured_abi():
    surface = MLXKernelSurface(name="stride_probe", source="ignored",
                               input_names=("x",), output_names=("out",))
    bindings = signature_bindings(METADATA_SIGNATURE, surface)
    by_name = {b.name: b for b in bindings}
    assert by_name["x"].kind is BindingKind.INPUT and by_name["x"].index == 0
    assert by_name["x_shape"].kind is BindingKind.INPUT and by_name["x_shape"].index == 1
    assert by_name["x_strides"].kind is BindingKind.INPUT and by_name["x_strides"].index == 2
    assert by_name["out"].kind is BindingKind.OUTPUT and by_name["out"].index == 3


def test_an_unrecognised_buffer_parameter_is_a_loud_error_not_a_guess():
    surface = MLXKernelSurface(name="s", source="ignored",
                               input_names=("x",), output_names=("out",))
    rogue = "const device float* mystery [[buffer(0)]], device float* out [[buffer(1)]]"
    with pytest.raises(ExtractionError, match="mystery"):
        signature_bindings(rogue, surface)


def test_metadata_inputs_are_row_contiguous_int32_shape_and_int64_strides():
    surface = MLXKernelSurface(name="s", source="ignored",
                               input_names=("x",), output_names=("out",))
    call = LiveCall(inputs={"x": np.zeros((2, 3, 4), np.float32)},
                    output_shapes=[((2, 3, 4), "float32")])
    extra_inputs, extra_params = metadata_inputs(call, surface)
    assert extra_inputs["x_shape"].dtype == np.int32
    assert extra_inputs["x_strides"].dtype == np.int64
    np.testing.assert_array_equal(extra_inputs["x_shape"], [2, 3, 4])
    np.testing.assert_array_equal(extra_inputs["x_strides"], [12, 4, 1])
    assert extra_params["x_ndim"] == 3
    assert "out_shape" in extra_inputs  # outputs get metadata too


def _outcome_fixture(sensitive_fail, insensitive_fail, insensitive_pass=1):
    """Build aligned result lists whose verdicts fail exactly as described."""
    total = sensitive_fail + insensitive_fail + insensitive_pass + 1
    extracted = [RunResult(status=RunStatus.OK, outputs=[np.zeros(1)])
                 for _ in range(total)]
    live = [LiveResult(ok=True, outputs=[np.zeros(1)]) for _ in range(total)]
    sensitive = ([True] * (sensitive_fail + 1)
                 + [False] * (insensitive_fail + insensitive_pass))
    failing = set(range(sensitive_fail)) | \
        set(range(sensitive_fail + 1, sensitive_fail + 1 + insensitive_fail))

    def verdict(index, ours, theirs):
        return index not in failing, "forced"

    return extracted, live, sensitive, verdict


def test_failures_concentrated_on_sensitive_cases_raise_the_alarm():
    report = validate_extraction(*_outcome_fixture(sensitive_fail=2, insensitive_fail=0))
    assert len(report.mismatches) == 2
    assert report.compile_option_alarm


def test_scattered_failures_do_not_raise_the_alarm():
    report = validate_extraction(*_outcome_fixture(sensitive_fail=1, insensitive_fail=1))
    assert len(report.mismatches) == 2
    assert not report.compile_option_alarm


def test_a_clean_match_raises_nothing():
    report = validate_extraction(*_outcome_fixture(sensitive_fail=0, insensitive_fail=0))
    assert report.all_matched and not report.compile_option_alarm


def test_a_failed_arm_is_unjudgeable_not_a_mismatch():
    extracted, live, sensitive, verdict = _outcome_fixture(0, 0)
    live[0] = LiveResult(ok=False, error="mlx raised")
    report = validate_extraction(extracted, live, sensitive, verdict)
    assert len(report.broken) == 1 and not report.mismatches
    assert not report.all_matched


# ---------------------------------------------------------------------------
# E2E: both arms, judged by the shipped oracle across structured modes
# ---------------------------------------------------------------------------
SILU_BODY = """
    uint i = thread_position_in_grid.x;
    float f = (float)input[i];
    out[i] = (T)(f / (1.0f + metal::exp(-f)));
"""

STRIDED_DOUBLE_BODY = """
    uint i = thread_position_in_grid.x;
    if (i < (uint)x_shape[0]) {
        out[i] = x[i * x_strides[0]] * 2.0f;
    }
"""


def silu_surface() -> MLXKernelSurface:
    return MLXKernelSurface(name="kv_silu", source=SILU_BODY,
                            input_names=("input",), output_names=("out",))


@pytest.mark.gpu
@requires_arms
def test_each_specialization_is_captured_fresh_and_distinct():
    surface = silu_surface()
    records = {}
    for msl_type, dtype in (("float", "float32"), ("half", "float16")):
        data = np.linspace(-4, 4, 32, dtype=np.float32).astype(dtype)
        call = LiveCall(inputs={"input": data}, output_shapes=[((32,), dtype)],
                        grid=(32, 1, 1), threadgroup=(32, 1, 1),
                        template=(("T", dtype),))
        records[msl_type] = capture_specialization(surface, call)
    assert records["float"].entry_point != records["half"].entry_point
    for record in records.values():
        assert "[[kernel]] void" in record.source
        assert record.mlx_version
        assert record.math_mode == "safe"
        assert len(record.outputs) == 1  # the capturing call's own arrays


@pytest.mark.gpu
@requires_arms
def test_the_extracted_kernel_matches_the_live_surface_under_the_shipped_oracle():
    """The dispatch's match criterion, verbatim: the shipped oracle's verdict
    function across the battery including structured modes, never bitwise."""
    from measure_escape import corpus_oracle_passes, load_meta, max_abs_error, reference
    from score_oracles import SWEEP_SEED, make_mode_inputs

    from kernelverify.tolerance.floor import conditioned_tolerance

    op = "silu_triton"
    meta = load_meta(op)
    base = float(meta["tolerances"]["float32"])
    dims = {"B": 2, "S": 7, "H": 256}
    size = dims["B"] * dims["S"] * dims["H"]
    modes = ["corpus uniform[-10,10]", "opposed signs", "near zero", "constant rows"]
    sensitive = [mode == "opposed signs" for mode in modes]

    surface = silu_surface()
    mode_inputs = {mode: make_mode_inputs(meta, dims, "float32", SWEEP_SEED, mode)
                   for mode in modes}
    shape = next(iter(mode_inputs.values()))["input"].shape

    seed_call = LiveCall(inputs=mode_inputs[modes[0]],
                         output_shapes=[(shape, "float32")],
                         grid=(size, 1, 1), threadgroup=(64, 1, 1),
                         template=(("T", "float32"),))
    record = capture_specialization(surface, seed_call)
    spec = extracted_spec(record, surface,
                          LaunchSpec(grid=(["B", "S", "H"], 1, 1),
                                     threadgroup=(64, 1, 1)))

    cases = [RunCase(inputs=mode_inputs[mode], params=dims,
                     output_shapes=[(shape, "float32")], label=mode)
             for mode in modes]
    ours = RUNNER.run(spec, cases)
    assert ours.compile_options == {"math_mode": "safe"}, (
        "the runner must record compiling under MLX's documented default")

    live = run_live(surface, [
        LiveCall(inputs=mode_inputs[mode], output_shapes=[(shape, "float32")],
                 grid=(size, 1, 1), threadgroup=(64, 1, 1),
                 template=(("T", "float32"),), label=mode)
        for mode in modes
    ])

    def verdict(index, extracted_outputs, live_outputs):
        mode = modes[index]
        inputs = mode_inputs[mode]
        ref = reference(meta, inputs,
                        (op, tuple(sorted(dims.items())), "float32", SWEEP_SEED, mode))
        tolerance = conditioned_tolerance(op, inputs, ref, base)
        ok = corpus_oracle_passes(extracted_outputs[0], live_outputs[0], tolerance)
        return ok, (f"err {max_abs_error(extracted_outputs[0], live_outputs[0]):.3e} "
                    f"tol {tolerance:.3e}")

    report = validate_extraction(ours.results, live, sensitive, verdict,
                                 labels=modes)
    assert report.all_matched, report.summary()
    assert not report.compile_option_alarm


@pytest.mark.gpu
@requires_arms
def test_a_metadata_binding_kernel_survives_the_round_trip():
    """The shape/stride ABI, end to end: MLX appends int32 shape and int64
    stride buffers, the bridge parses them into bindings, and the transport
    carries the arrays. This is the extraction path that forced integer
    tensor dtypes into the transport."""
    surface = MLXKernelSurface(name="stride_probe", source=STRIDED_DOUBLE_BODY,
                               input_names=("x",), output_names=("out",),
                               ensure_row_contiguous=False)
    x = np.arange(8, dtype=np.float32)
    call = LiveCall(inputs={"x": x}, output_shapes=[((8,), "float32")],
                    grid=(8, 1, 1), threadgroup=(8, 1, 1))
    record = capture_specialization(surface, call)
    assert "x_shape" in record.source and "x_strides" in record.source

    spec = extracted_spec(record, surface,
                          LaunchSpec(grid=("n", 1, 1), threadgroup=(8, 1, 1)))
    extra_inputs, _ = metadata_inputs(call, surface)
    case = RunCase(inputs={"x": x, **extra_inputs}, params={"n": 8},
                   output_shapes=[((8,), "float32")])
    result = RUNNER.run_one(spec, case)
    assert result.ok, result.detail
    np.testing.assert_array_equal(result.outputs[0], x * 2.0)


@pytest.mark.gpu
@requires_arms
def test_the_live_arm_reports_a_broken_surface_as_an_error_not_a_crash():
    broken = MLXKernelSurface(name="kv_broken", source="this is not MSL;",
                              input_names=("input",), output_names=("out",))
    results = run_live(broken, [
        LiveCall(inputs={"input": np.zeros(4, np.float32)},
                 output_shapes=[((4,), "float32")], grid=(4, 1, 1),
                 threadgroup=(4, 1, 1))
    ])
    assert len(results) == 1 and not results[0].ok
    assert results[0].error


@pytest.mark.gpu
@requires_arms
def test_a_non_default_math_mode_is_recorded_as_used():
    from kernelverify.runners import Binding, KernelSpec

    copy_src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void copy(device const float* x [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) { out[gid] = x[gid]; }
    """
    spec = KernelSpec(source=copy_src, entry_point="copy",
                      bindings=(Binding(BindingKind.INPUT, "x"),
                                Binding(BindingKind.OUTPUT)),
                      launch=LaunchSpec(grid=(16, 1, 1), threadgroup=(16, 1, 1)))
    case = RunCase(inputs={"x": np.arange(16, dtype=np.float32)},
                   output_shapes=[((16,), "float32")])
    runner = MetalRunner(math_mode="fast")
    batch = runner.run(spec, [case])
    assert batch.all_ok, [r.detail for r in batch if not r.ok]
    assert batch.compile_options == {"math_mode": "fast"}
