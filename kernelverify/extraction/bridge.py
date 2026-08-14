"""From an extracted translation unit to a runnable `KernelSpec`.

MLX's generated signature is the ground truth for how the kernel wants its
buffers: inputs in declaration order, then shape/stride/ndim metadata for any
tensor the body indexes through, then outputs. Guessing that ABI is exactly
how an extraction gets quietly wrong strides, so the bindings are parsed out
of the signature MLX actually generated, buffer indices included, and any
parameter this bridge does not recognise is a loud error rather than a guess.

The metadata convention, measured on mlx 0.32.0:

    const device float*    x          [[buffer(0)]]   input tensor
    const constant int*    x_shape    [[buffer(1)]]   int32 per-dimension sizes
    const constant int64_t* x_strides [[buffer(2)]]   int64 element strides
    const constant int&    x_ndim     [[buffer(3)]]   scalar rank
    device float16_t*      out        [[buffer(4)]]   output tensor

`metadata_inputs` builds the row-contiguous shape/stride arrays a `RunCase`
must supply for those bindings, which is legitimate for the extracted arm
because the live arm's `ensure_row_contiguous=True` default gives the kernel
the same layout.
"""

from __future__ import annotations

import re

import numpy as np

from kernelverify.extraction.mlx_arm import CaptureRecord, ExtractionError
from kernelverify.extraction.surface import LiveCall, MLXKernelSurface
from kernelverify.runners.spec import Binding, BindingKind, KernelSpec, LaunchSpec

#: What a standalone Metal compile needs that MLX's pipeline supplies itself.
MSL_PREAMBLE = "#include <metal_stdlib>\nusing namespace metal;\n"

# One signature parameter: type tokens, the name, and its buffer index.
_BUFFER_PARAM = re.compile(
    r"(?P<decl>[\w:]+(?:\s+[\w:]+)*\s*(?:\*|&))\s*"
    r"(?P<name>\w+)\s*\[\[buffer\((?P<index>\d+)\)\]\]"
)

# Metadata element types by declaration, per the measured ABI.
_METADATA_DTYPES = {"int": "int32", "int32_t": "int32", "int64_t": "int64",
                    "uint": "int32", "size_t": "int64"}


def _metadata_owner(name: str, owners) -> tuple[str, str] | None:
    for owner in owners:
        for suffix in ("shape", "strides", "ndim"):
            if name == f"{owner}_{suffix}":
                return owner, suffix
    return None


def signature_bindings(source: str, surface: MLXKernelSurface) -> tuple[Binding, ...]:
    """The bindings the generated signature declares, indices included."""
    bindings: list[Binding] = []
    names = surface.input_names + surface.output_names
    for match in _BUFFER_PARAM.finditer(source):
        decl, name, index = match.group("decl"), match.group("name"), int(match.group("index"))
        is_reference = decl.rstrip().endswith("&")
        if name in surface.input_names:
            bindings.append(Binding(BindingKind.INPUT, name, index=index))
        elif name in surface.output_names:
            bindings.append(Binding(BindingKind.OUTPUT, name, index=index))
        elif (owned := _metadata_owner(name, names)) is not None:
            qualifiers = {"const", "constant", "device", "threadgroup"}
            tokens = [t for t in re.split(r"[\s*&]+", decl) if t and t not in qualifiers]
            element = tokens[-1] if tokens else ""
            dtype = _METADATA_DTYPES.get(element)
            if dtype is None:
                raise ExtractionError(
                    f"metadata parameter {name!r} has element type {element!r}, "
                    "which this bridge does not know how to feed"
                )
            if is_reference:  # a scalar, bound with setBytes
                bindings.append(Binding(BindingKind.SCALAR, name, dtype, index=index))
            else:  # a small tensor the case supplies
                bindings.append(Binding(BindingKind.INPUT, name, index=index))
        else:
            raise ExtractionError(
                f"the generated signature binds {name!r} at buffer {index}, which is "
                "neither a declared input or output nor recognised metadata; "
                "extend the bridge before trusting this extraction"
            )
    if not bindings:
        raise ExtractionError("no [[buffer(N)]] parameters found in the extracted source")
    return tuple(bindings)


def extracted_spec(record: CaptureRecord, surface: MLXKernelSurface,
                   launch: LaunchSpec) -> KernelSpec:
    """A `KernelSpec` that runs the extracted source under the Metal runner.

    The grid semantics of `launch` should count total threads
    (`LaunchMode.THREADS`), which is what MLX's `grid` means.
    """
    header = surface.header if surface.header.strip() and \
        surface.header.strip() not in record.source else ""
    return KernelSpec(
        source=MSL_PREAMBLE + header + record.source,
        entry_point=record.entry_point,
        bindings=signature_bindings(record.source, surface),
        launch=launch,
        name=f"extracted:{record.surface_name}",
    )


def metadata_inputs(call: LiveCall, surface: MLXKernelSurface) -> tuple[dict, dict]:
    """The shape/stride arrays and ndim scalars a metadata-binding case needs.

    Returns `(extra_inputs, extra_params)` for every input and output the call
    describes, row-contiguous layout; a `RunCase` ignores entries its spec
    never binds, so over-supplying is harmless.
    """
    described = {name: tuple(call.inputs[name].shape) for name in surface.input_names}
    for name, (shape, _) in zip(surface.output_names, call.output_shapes):
        described[name] = shape

    extra_inputs: dict = {}
    extra_params: dict = {}
    for name, shape in described.items():
        strides = np.empty(len(shape), dtype=np.int64)
        running = 1
        for axis in range(len(shape) - 1, -1, -1):
            strides[axis] = running
            running *= shape[axis]
        extra_inputs[f"{name}_shape"] = np.asarray(shape, dtype=np.int32)
        extra_inputs[f"{name}_strides"] = strides
        extra_params[f"{name}_ndim"] = len(shape)
    return extra_inputs, extra_params
