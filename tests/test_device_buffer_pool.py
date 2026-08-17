"""The device buffer pool: what a case may see in a buffer somebody else used.

The pool exists because a released MTLBuffer's dirty pages never return to the
OS on this stack, so per-case allocation is a leak (the module docstring of
`kernelverify/runners/device.py` carries the measurement and the three Jetsam
kills that forced it). Reuse buys that back and costs an invariant: one buffer
per binding index is shared by every case AND every spec that runs on one
MetalDevice, grown to the largest extent ever bound there.

That invariant is what these tests pin. Before the pool, a case got a fresh
allocation whose bytes past its own data were zeros from a fresh page; with a
pool they would be the previous case's bytes unless the pool clears them. A
candidate kernel reading past its declared extent would then return a value
that depends on what ran before it - an order-dependent verdict, which is the
one failure a verifier may never have.

Every test here runs on ONE MetalDevice, deliberately: a device per test would
give each case a private pool and prove nothing.
"""

import numpy as np
import pytest

pytest.importorskip("Metal")

from kernelverify.runners.device import (  # noqa: E402
    LaunchError,
    MetalDevice,
    _write_into,
)
from kernelverify.runners.result import RunStatus  # noqa: E402
from kernelverify.runners.spec import (  # noqa: E402
    TENSOR_DTYPES,
    Binding,
    BindingKind,
    KernelSpec,
    LaunchSpec,
    RunCase,
)

try:
    _DEVICE = MetalDevice()
except RuntimeError:  # pragma: no cover - a machine without a GPU
    _DEVICE = None

# Whole-module: every test here allocates through the device buffer pool, which is a live Metal allocation,
# so the gpu marker is module-level rather than 62 copies of itself.
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(_DEVICE is None,
                                reason="no Metal device on this machine")]


@pytest.fixture(scope="module")
def device():
    """One device, one pool, for the whole module - the sharing under test."""
    return _DEVICE


# ---------------------------------------------------------------------------
# Kernels. Each one is written the way a generated candidate would be, and
# each reads or writes exactly the extent its case declares - except where a
# test deliberately declares an extent wider than the data it supplies, which
# is how a candidate that runs off the end of its input is expressed.
# ---------------------------------------------------------------------------
COPY_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void copy_n(device const float* x [[buffer(0)]],
                   device float* out     [[buffer(1)]],
                   constant uint& n      [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = x[gid];
}
"""

# The same copy with the roles at index 0 and 1 swapped: output first. Two
# specs of one device therefore disagree about what index 0 is for.
SWAPPED_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void copy_swapped(device float* out       [[buffer(0)]],
                         device const float* x   [[buffer(1)]],
                         constant uint& n        [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = x[gid] * 2.0f;
}
"""

# Writes the first half of its output and leaves the rest alone.
HALF_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void first_half(device const float* x [[buffer(0)]],
                       device float* out     [[buffer(1)]],
                       constant uint& n      [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < n / 2) out[gid] = x[gid] + 1.0f;
}
"""


def _spec(source: str, entry: str, swapped: bool = False) -> KernelSpec:
    bindings = ((Binding(BindingKind.OUTPUT), Binding(BindingKind.INPUT, "x"))
                if swapped else
                (Binding(BindingKind.INPUT, "x"), Binding(BindingKind.OUTPUT)))
    return KernelSpec(
        source=source, entry_point=entry,
        bindings=bindings + (Binding(BindingKind.SCALAR, "n", "uint32"),),
        launch=LaunchSpec(grid=("m", 1, 1), threadgroup=(64, 1, 1)),
    )


def _case(x: np.ndarray, out_n: int, label: str) -> RunCase:
    """`out_n` is what the kernel is TOLD to process; ``x`` is what it is
    given. Declaring more than the data supplies is how a candidate reading
    off the end of its input is written down."""
    return RunCase(inputs={"x": x}, params={"n": out_n, "m": out_n},
                   output_shapes=[((out_n,), "float32")], label=label)


def _run(device, spec, case):
    result = device.compile(spec).run(case, warmup=0, repeats=0)
    assert result.ok, result.detail
    return result.outputs[0]


# ---------------------------------------------------------------------------
# (a) A small case after a large one at the same index. The slack the small
# case never wrote must read as zeros, the way a fresh page did before the
# pool - not as the large case's bytes.
# ---------------------------------------------------------------------------
def test_a_small_case_never_reads_the_previous_case_s_bytes(device):
    spec = _spec(COPY_SRC, "copy_n")
    kernel = device.compile(spec)

    big = np.full(4096, 7.0, dtype=np.float32)
    assert np.array_equal(_run(device, spec, _case(big, 4096, "big")), big)

    small = np.arange(8, dtype=np.float32)
    out = kernel.run(_case(small, 64, "small reading past its data"),
                     warmup=0, repeats=0)
    assert out.ok, out.detail
    expected = np.zeros(64, dtype=np.float32)
    expected[:8] = small
    np.testing.assert_array_equal(
        out.outputs[0], expected,
        "the pooled input buffer served the previous case's bytes past this "
        "case's data: the verdict now depends on what ran before it")


# ---------------------------------------------------------------------------
# (b) The output side of the same property: a kernel that writes only part of
# its output, after a bigger case wrote the whole pooled buffer.
# ---------------------------------------------------------------------------
def test_a_partial_writer_after_a_bigger_case_still_reads_zero(device):
    big_spec, half_spec = _spec(COPY_SRC, "copy_n"), _spec(HALF_SRC, "first_half")
    big = np.full(2048, 9.0, dtype=np.float32)
    assert np.array_equal(_run(device, big_spec, _case(big, 2048, "big")), big)

    x = np.full(128, 3.0, dtype=np.float32)
    out = _run(device, half_spec, _case(x, 128, "half writer"))
    np.testing.assert_array_equal(out[:64], np.full(64, 4.0, dtype=np.float32))
    assert np.count_nonzero(out[64:]) == 0, (
        "the untouched half of a pooled output carried the previous case's "
        "values, so a kernel would be judged on what the pool held")


# ---------------------------------------------------------------------------
# (c) Two specs on one device that disagree about what buffer index 0 is for.
# The pool is on the device, so the index is shared ACROSS kernels.
# ---------------------------------------------------------------------------
def test_two_specs_swapping_the_roles_at_index_zero_stay_exact(device):
    plain, swapped = _spec(COPY_SRC, "copy_n"), _spec(SWAPPED_SRC,
                                                      "copy_swapped", True)
    x_big = np.linspace(-5.0, 5.0, 1024, dtype=np.float32)
    np.testing.assert_array_equal(_run(device, plain, _case(x_big, 1024, "in@0")),
                                  x_big)

    x_small = np.arange(256, dtype=np.float32)
    np.testing.assert_array_equal(
        _run(device, swapped, _case(x_small, 256, "out@0")), x_small * 2.0)

    # ...and back, so neither ordering is a one-way door.
    np.testing.assert_array_equal(_run(device, plain, _case(x_small, 256, "in@0")),
                                  x_small)


# ---------------------------------------------------------------------------
# (d) A LaunchError raised part-way through _make_buffers leaves the pool
# half-written. The next case must still be judged on its own bytes.
# ---------------------------------------------------------------------------
def test_a_launch_error_mid_binding_does_not_poison_the_next_case(device):
    kernel = device.compile(_spec(COPY_SRC, "copy_n"))
    poison = np.full(4096, 11.0, dtype=np.float32)
    bad = RunCase(inputs={"x": poison}, params={"n": "not a number", "m": 4096},
                  output_shapes=[((4096,), "float32")], label="bad scalar")
    failed = kernel.run(bad, warmup=0, repeats=0)
    assert failed.status is RunStatus.LAUNCH_ERROR, failed.status

    small = np.arange(8, dtype=np.float32)
    out = kernel.run(_case(small, 64, "after the failure"), warmup=0, repeats=0)
    assert out.ok, out.detail
    expected = np.zeros(64, dtype=np.float32)
    expected[:8] = small
    np.testing.assert_array_equal(
        out.outputs[0], expected,
        "the abandoned case's input survived in the pool and the next case "
        "read it")


# ---------------------------------------------------------------------------
# (e) The copy into a pooled buffer, per supported tensor dtype. The pooled
# buffer is deliberately larger than the array, which is the case _write_into
# has to get right: it copies the array and touches nothing beyond it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype_name", sorted(TENSOR_DTYPES))
def test_write_into_round_trips_every_supported_dtype(device, dtype_name):
    dtype = np.dtype(TENSOR_DTYPES[dtype_name])
    values = np.arange(1, 65, dtype=dtype)
    if dtype.kind == "f":
        values = values / dtype.type(3.0)
    index = 900  # an index no kernel here binds, so this owns its pool slot
    buffer, held = device.pooled_buffer(index, values.nbytes * 4)
    _write_into(buffer, values)
    back = np.frombuffer(buffer.contents().as_buffer(held), dtype=dtype,
                         count=values.size)
    np.testing.assert_array_equal(back, values)
    assert back.tobytes() == values.tobytes(), "a dtype must survive bit for bit"


# ---------------------------------------------------------------------------
# The pool's own contract, under the buffers above: what it hands back, what
# it reports, and the one size it must refuse.
# ---------------------------------------------------------------------------
def test_the_pool_reports_the_size_it_holds_and_clears_the_slack(device):
    index = 901
    buffer, held = device.pooled_buffer(index, 4096)
    assert held == 4096
    np.frombuffer(buffer.contents().as_buffer(4096), dtype=np.uint8)[...] = 0xAB

    same, still_held = device.pooled_buffer(index, 4096)
    assert same is buffer and still_held == 4096, "a same-size case reuses it"

    smaller, held_for_small = device.pooled_buffer(index, 64)
    assert smaller is buffer, "the pool must not allocate for a smaller case"
    assert held_for_small == 4096, "the caller is told the real extent it got"
    tail = np.frombuffer(buffer.contents().as_buffer(4096), dtype=np.uint8,
                         count=4096 - 64, offset=64)
    assert np.count_nonzero(tail) == 0, "the slack past this case must be zeros"


def test_a_zero_byte_binding_is_a_launch_error_not_a_silent_reuse(device):
    device.pooled_buffer(902, 256)
    with pytest.raises(LaunchError, match="zero"):
        device.pooled_buffer(902, 0)


def test_release_pool_drops_every_held_buffer_and_the_next_one_is_clean(device):
    """Releasing hands the pages back; the buffer that replaces them must
    still start at zeros, which is what every pooled buffer's determinism
    rests on before the first case writes anything."""
    dirty, _held = device.pooled_buffer(903, 4096)
    np.frombuffer(dirty.contents().as_buffer(4096), dtype=np.uint8)[...] = 0xCD
    device.release_pool()
    assert device._buffer_pool == {}

    fresh, held = device.pooled_buffer(903, 4096)
    assert held == 4096
    assert np.count_nonzero(
        np.frombuffer(fresh.contents().as_buffer(4096), dtype=np.uint8)) == 0
    # That release_pool gives the MEMORY back, not just the references, is
    # measured next to the leak test in tests/test_serving_survival.py, which
    # already owns this machine's footprint reader.
