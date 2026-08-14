"""Device-arithmetic ensemble members for the MLX-affine quantization contract.

Why this module exists
----------------------
Phase 0 calibrated the quant tolerance (K = 3 over a five-member floor) from
CPU numpy members alone, and pre-registered the branch that a correct Metal
kernel exceeding that CPU-calibrated tolerance triggers ensemble RE-DERIVATION
with device members, never a false-positive ship. These are those members: a
distinct class of the contract population whose rounding sequence is the one
the device actually executes, not the one numpy simulates.

What makes device arithmetic a class of its own
-----------------------------------------------
The CPU classes differ by evaluation DOMAIN (dequant-domain against
int-domain), because domain is what changes the rounding sequence rather than
merely reordering it. Device execution changes the rounding sequence through a
different mechanism: the Metal compiler contracts multiplies and adds into
fused multiply-adds and may reassociate under its default fast-math contract,
and simdgroup reductions impose the 32-lane tree the hardware ships. Both are
inside the contract's freedoms - an fma is wider than fp32, reassociation is
order freedom - so every member here is correct by construction, but no CPU
member reproduces their rounding. That is the definition of a class the floor
must carry members of, and per the structural gate (ADR 0009 G3) it carries a
minimum of two DISTINCT members, counted on outputs rather than names.

The members, spanning both domains on the device:

- ``device-dequant-loop``: one thread per output element, dequantize on the
  fly, accumulate in fp32 down the row as a single loop. What the compiler
  emits for that loop (vectorised, reassociated, contracted) is the point.
- ``device-dequant-simd``: one simdgroup per output element, lanes strided
  over the reduction, dequant-domain, ``simd_sum`` tree at the end.
- ``device-factored-simd``: one simdgroup per output element, lanes strided
  over GROUPS, int-domain per group (``s * sum(x*q) + b * sum(x)``),
  ``simd_sum`` tree at the end - the shape of MLX's own quantized matvec.

Why the members take unpacked codes
-----------------------------------
A shipped kernel unpacks a packed bit stream; these members take the codes as
a tensor. Unpacking is integer arithmetic and bit-exact, so it cannot move a
tolerance floor - the seam under calibration here is the floating-point
arithmetic, and taking codes directly keeps one compiled spec valid at every
bit width (codes up to 2^bits - 1 are exact in the float16 carrier for every
width this contract states). Pack-format faults are the catalogue's business,
not the floor's.

The build target
----------------
Specs are stated against the runner types at the formal shape freeze
(metal-runner 37892fa): ``RunCase.output_shapes`` is one ``(shape, dtype)``
per output binding, ``RunResult.outputs`` pairs it, ``KernelSpec`` is
unchanged. Specialization is outside the runner by decision: each member is a
template with a ``$XT`` hole for the activation element type, and
``runners.specialize`` fills it, so the spec the runner sees carries final
MSL and the substitution stays this module's record.

Execution
---------
Calibration runs these members in-process through ``runners.device`` -
compile once per spec, dispatch per case - because they are trusted,
hand-written implementations. The subprocess isolation in ``runners.metal``
exists for untrusted generated candidates and stays there.
"""

from __future__ import annotations

import numpy as np

from kernelverify.runners.spec import Binding, KernelSpec, LaunchSpec, RunCase
from kernelverify.runners.specialize import specialize

ENTRY_POINT = "qmv_member"
SIMD_WIDTH = 32  # asserted against the compiled pipeline before any dispatch

DEVICE_CLASS = "device-arithmetic"

_HEADER = """\
#include <metal_stdlib>
using namespace metal;
"""

_ARGS = """\
    device const ${XT}* x        [[buffer(0)]],
    device const half*   q        [[buffer(1)]],
    device const float*  scales   [[buffer(2)]],
    device const float*  biases   [[buffer(3)]],
    device ${XT}*       out      [[buffer(4)]],
    constant uint&       d_in     [[buffer(5)]],
    constant uint&       d_out    [[buffer(6)]],
    constant uint&       n_groups [[buffer(7)]],
    constant uint&       group    [[buffer(8)]],
"""

_DEQUANT_LOOP = _HEADER + "kernel void " + ENTRY_POINT + "(\n" + _ARGS + """\
    uint2 tid [[thread_position_in_grid]])
{
    const uint row = tid.x;
    const uint batch = tid.y;
    if (row >= d_out) { return; }
    device const ${XT}* xrow = x + batch * d_in;
    device const half*   qrow = q + row * d_in;
    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = scales[row * n_groups + g];
        const float b = biases[row * n_groups + g];
        const uint start = g * group;
        for (uint k = 0; k < group; ++k) {
            const float w = fma(s, float(qrow[start + k]), b);
            acc = fma(float(xrow[start + k]), w, acc);
        }
    }
    out[batch * d_out + row] = ${XT}(acc);
}
"""

_DEQUANT_SIMD = _HEADER + "kernel void " + ENTRY_POINT + "(\n" + _ARGS + """\
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint row = tg.x;
    const uint batch = tg.y;
    device const ${XT}* xrow = x + batch * d_in;
    device const half*   qrow = q + row * d_in;
    float acc = 0.0f;
    for (uint k = lane; k < d_in; k += 32u) {
        const uint g = k / group;
        const float w = fma(scales[row * n_groups + g], float(qrow[k]),
                            biases[row * n_groups + g]);
        acc = fma(float(xrow[k]), w, acc);
    }
    const float total = simd_sum(acc);
    if (lane == 0u) { out[batch * d_out + row] = ${XT}(total); }
}
"""

_FACTORED_SIMD = _HEADER + "kernel void " + ENTRY_POINT + "(\n" + _ARGS + """\
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint row = tg.x;
    const uint batch = tg.y;
    device const ${XT}* xrow = x + batch * d_in;
    device const half*   qrow = q + row * d_in;
    float acc = 0.0f;
    for (uint g = lane; g < n_groups; g += 32u) {
        const uint start = g * group;
        float xq = 0.0f;
        float xs = 0.0f;
        for (uint k = 0; k < group; ++k) {
            const float xv = float(xrow[start + k]);
            xq = fma(xv, float(qrow[start + k]), xq);
            xs += xv;
        }
        acc = fma(scales[row * n_groups + g], xq, acc);
        acc = fma(biases[row * n_groups + g], xs, acc);
    }
    const float total = simd_sum(acc);
    if (lane == 0u) { out[batch * d_out + row] = ${XT}(total); }
}
"""

_SOURCES = {
    "device-dequant-loop": _DEQUANT_LOOP,
    "device-dequant-simd": _DEQUANT_SIMD,
    "device-factored-simd": _FACTORED_SIMD,
}

DEVICE_MEMBERS = tuple(_SOURCES)

_XT = {"float32": "float", "float16": "half"}

# One thread per output element for the loop member; one simdgroup (one
# 32-thread threadgroup) per output element for the reduction members.
_LAUNCHES = {
    "device-dequant-loop": LaunchSpec(grid=("D_OUT", "BATCH", 1),
                                      threadgroup=(64, 1, 1), mode="threads"),
    "device-dequant-simd": LaunchSpec(grid=("D_OUT", "BATCH", 1),
                                      threadgroup=(SIMD_WIDTH, 1, 1),
                                      mode="threadgroups"),
    "device-factored-simd": LaunchSpec(grid=("D_OUT", "BATCH", 1),
                                       threadgroup=(SIMD_WIDTH, 1, 1),
                                       mode="threadgroups"),
}


def device_member_template(member: str, x_dtype: str) -> KernelSpec:
    """The member as a template spec: a $XT hole in the source.

    Bindings and launch are already concrete - `specialize` passes them
    through untouched by design - so the only hole is the activation element
    type in the MSL itself.
    """
    if member not in _SOURCES:
        raise KeyError(f"no device member named {member!r}; "
                       f"the class holds {sorted(_SOURCES)}")
    if x_dtype not in _XT:
        raise KeyError(f"activation dtype {x_dtype!r} is not one of {sorted(_XT)}")
    return KernelSpec(
        source=_SOURCES[member],
        entry_point=ENTRY_POINT,
        bindings=(
            Binding(kind="input", name="x", dtype=x_dtype),
            Binding(kind="input", name="q", dtype="float16"),
            Binding(kind="input", name="scales", dtype="float32"),
            Binding(kind="input", name="biases", dtype="float32"),
            Binding(kind="output", dtype=x_dtype),
            Binding(kind="scalar", name="D_IN", dtype="uint32"),
            Binding(kind="scalar", name="D_OUT", dtype="uint32"),
            Binding(kind="scalar", name="N_GROUPS", dtype="uint32"),
            Binding(kind="scalar", name="GROUP", dtype="uint32"),
        ),
        launch=_LAUNCHES[member],
        name=f"{member}[{x_dtype}]",
    )


def device_member_spec(member: str, x_dtype: str) -> KernelSpec:
    """The runnable candidate: raw final MSL plus bindings and launch."""
    return specialize(device_member_template(member, x_dtype),
                      {"XT": _XT[x_dtype]})


# ---------------------------------------------------------------------------
# Frozen-form construction and reading, kept as named functions so the
# calibration harness states its intent once rather than repeating field names.
# ---------------------------------------------------------------------------
def make_run_case(inputs: dict, params: dict, output_shapes: list,
                  label: str = "") -> RunCase:
    return RunCase(inputs=inputs, params=params,
                   output_shapes=output_shapes, label=label)


def result_outputs(result) -> list:
    return list(result.outputs)


# ---------------------------------------------------------------------------
# Cases and execution
# ---------------------------------------------------------------------------
def _x_dtype_name(x: np.ndarray) -> str:
    name = str(x.dtype)
    if name not in _XT:
        raise ValueError(f"activations must be float32 or float16, got {name}")
    return name


def device_member_case(x: np.ndarray, artefact, label: str = "") -> RunCase:
    """One battery case for any device member: codes ride a float16 carrier.

    Codes are exact in float16 up to 2048, which covers every bit width the
    contract states; scales and biases ride float32, an exact upcast of their
    fp16 storage, so the kernel sees the same values the CPU members see.
    """
    max_code = (1 << artefact.contract.bits) - 1
    if max_code > 2048:
        raise ValueError(f"bits={artefact.contract.bits} codes exceed the "
                         "float16 integer-exact range; widen the carrier first")
    d_out, d_in = artefact.q.shape
    n_groups = d_in // artefact.contract.group_size
    return make_run_case(
        inputs={
            "x": np.ascontiguousarray(x),
            "q": np.ascontiguousarray(artefact.q.astype(np.float16)),
            "scales": np.ascontiguousarray(artefact.scales.astype(np.float32)),
            "biases": np.ascontiguousarray(artefact.biases.astype(np.float32)),
        },
        params={"D_IN": d_in, "D_OUT": d_out, "N_GROUPS": n_groups,
                "GROUP": artefact.contract.group_size, "BATCH": int(x.shape[0])},
        output_shapes=[((int(x.shape[0]), d_out), _x_dtype_name(x))],
        label=label,
    )


class DeviceMemberSession:
    """Compile-once, dispatch-many evaluation of the device members.

    One Metal compile per (member, activation dtype) spec, cached; every case
    after that costs only its dispatch. This is the case-grouping the
    calibration harness relies on, expressed as a cache so callers cannot get
    the grouping wrong by iterating in the wrong order.
    """

    def __init__(self, device=None):
        if device is None:
            from kernelverify.runners.device import MetalDevice
            device = MetalDevice()
        self.device = device
        self._compiled = {}

    def compiled(self, member: str, x_dtype: str):
        key = (member, x_dtype)
        if key not in self._compiled:
            kernel = self.device.compile(device_member_spec(member, x_dtype))
            if kernel.execution_width != SIMD_WIDTH:
                raise RuntimeError(
                    f"pipeline execution width {kernel.execution_width} != "
                    f"{SIMD_WIDTH}; the simd members assume one simdgroup per "
                    "threadgroup and would reduce over the wrong lane count")
            self._compiled[key] = kernel
        return self._compiled[key]

    def run(self, member: str, x: np.ndarray, artefact, label: str = "") -> np.ndarray:
        """The member's output for one case, or a loud error - a trusted
        calibration member that fails to run is a harness bug, never a verdict."""
        kernel = self.compiled(member, _x_dtype_name(x))
        result = kernel.run(device_member_case(x, artefact, label),
                            warmup=0, repeats=0)
        if not result.ok:
            raise RuntimeError(f"{member}: {result.status.value} - {result.detail}")
        return result_outputs(result)[0]
