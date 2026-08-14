"""The contract a candidate Metal kernel arrives in, and its JSON wire form.

A generated kernel is raw Metal shading language plus the information a host
needs to launch it. Metal supplies none of that from the source: the compiler
knows a function takes a `device const float*` at buffer 0, but not that the
tensor bound there is `q`, not what shape it has, and not how many threads the
author intended. So a candidate is a `KernelSpec`, which pairs the source with

- `bindings`, an ordered list saying what goes at each buffer index: an input
  tensor by name, the single output, or a scalar argument by name, and
- `launch`, the grid and threadgroup the kernel was written for.

Shapes and scalar values are deliberately *not* in the spec. One spec runs
against a whole battery of cases, and every case has different dimensions, so
those arrive per case in `RunCase`. That split is what lets the runner compile
a candidate once and dispatch it many times, which matters because compiling
MSL costs far more than running a small kernel.

Sizes that depend on the case are written as extents. An extent is an integer,
the name of a run-time parameter, or a list whose parts multiply together, so
a grid of one thread per element of an (M, D) output is `[["M", "D"], 1, 1]`.
That covers the shapes this battery produces without inventing an expression
language; anything needing arithmetic beyond a product can be passed in as a
parameter the caller computed.

Nothing in this module touches Metal or numpy execution. `device.py` runs a
spec on the GPU inside the worker process; `metal.py` is the parent-side API.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# Tensor element types the battery uses. The shader declares its own types;
# these only decide how many bytes per element the host copies.
TENSOR_DTYPES = {"float32": np.float32, "float16": np.float16}

# Scalar argument types, bound with setBytes rather than a buffer allocation.
SCALAR_DTYPES = {
    "uint32": np.uint32,
    "int32": np.int32,
    "float32": np.float32,
    "float16": np.float16,
}

MAX_DIMENSIONS = 3  # Metal grids are three dimensional


class SpecError(ValueError):
    """The spec is unrunnable as written: bad binding, missing name, bad grid."""


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------
class BindingKind(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    SCALAR = "scalar"


@dataclass(frozen=True)
class Binding:
    """One kernel argument.

    `index` is the Metal buffer index. It defaults to the binding's position in
    the list, which is how a generated kernel almost always numbers its
    arguments, and can be set explicitly when it does not.
    """

    kind: BindingKind
    name: str = ""
    dtype: str = ""
    index: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "kind", BindingKind(self.kind))
        if self.kind is BindingKind.SCALAR:
            if not self.name:
                raise SpecError("a scalar binding needs the name of a run parameter")
            if self.dtype not in SCALAR_DTYPES:
                raise SpecError(
                    f"scalar {self.name!r} has dtype {self.dtype!r}; "
                    f"expected one of {sorted(SCALAR_DTYPES)}"
                )
        elif self.kind is BindingKind.INPUT and not self.name:
            raise SpecError("an input binding needs the name of an input tensor")

    def to_json(self) -> dict:
        return {"kind": self.kind.value, "name": self.name,
                "dtype": self.dtype, "index": self.index}

    @staticmethod
    def from_json(raw: dict) -> "Binding":
        return Binding(kind=BindingKind(raw["kind"]), name=raw.get("name", ""),
                       dtype=raw.get("dtype", ""), index=raw.get("index"))


# ---------------------------------------------------------------------------
# Extents: sizes that are only known once a case supplies its dimensions
# ---------------------------------------------------------------------------
def resolve_extent(extent: Any, params: dict[str, Any], where: str) -> int:
    """int, parameter name, or a list of those multiplied together."""
    if isinstance(extent, bool):  # bool is an int subclass and never a size
        raise SpecError(f"{where}: {extent!r} is not a valid extent")
    if isinstance(extent, int):
        value = extent
    elif isinstance(extent, str):
        if extent not in params:
            raise SpecError(f"{where}: no run parameter named {extent!r}")
        value = params[extent]
        if not isinstance(value, int) or isinstance(value, bool):
            raise SpecError(f"{where}: parameter {extent!r} is {value!r}, not an integer")
    elif isinstance(extent, (list, tuple)):
        if not extent:
            raise SpecError(f"{where}: an empty product is not a size")
        value = 1
        for part in extent:
            value *= resolve_extent(part, params, where)
    else:
        raise SpecError(f"{where}: {extent!r} is not a valid extent")
    if value < 1:
        raise SpecError(f"{where}: resolved to {value}, which is not a positive size")
    return value


def _extent_json(extent: Any) -> Any:
    return list(extent) if isinstance(extent, tuple) else extent


# ---------------------------------------------------------------------------
# Launch configuration
# ---------------------------------------------------------------------------
class LaunchMode(str, Enum):
    #: dispatchThreads: the grid counts threads and Metal handles a ragged
    #: final threadgroup. Needs Apple family 4 or newer, which every Apple
    #: silicon Mac is.
    THREADS = "threads"
    #: dispatchThreadgroups: the grid counts whole threadgroups, so the kernel
    #: is responsible for bounds checking its own tail.
    THREADGROUPS = "threadgroups"


@dataclass(frozen=True)
class LaunchSpec:
    """The grid a candidate wants, in extents that a case resolves to sizes."""

    grid: tuple = (1, 1, 1)
    threadgroup: tuple = (1, 1, 1)
    mode: LaunchMode = LaunchMode.THREADS
    #: Dynamically sized `threadgroup` memory, in bytes, one entry per index.
    #: Statically sized threadgroup arrays declared in the shader need nothing
    #: here.
    threadgroup_memory: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "mode", LaunchMode(self.mode))
        for name in ("grid", "threadgroup"):
            value = tuple(getattr(self, name))
            if len(value) != MAX_DIMENSIONS:
                raise SpecError(f"{name} must have {MAX_DIMENSIONS} components, got {value!r}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "threadgroup_memory", tuple(self.threadgroup_memory))

    def resolve(self, params: dict) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, ...]]:
        grid = tuple(resolve_extent(e, params, f"grid[{i}]") for i, e in enumerate(self.grid))
        group = tuple(resolve_extent(e, params, f"threadgroup[{i}]")
                      for i, e in enumerate(self.threadgroup))
        memory = tuple(resolve_extent(e, params, f"threadgroup_memory[{i}]")
                       for i, e in enumerate(self.threadgroup_memory))
        return grid, group, memory

    def to_json(self) -> dict:
        return {
            "grid": [_extent_json(e) for e in self.grid],
            "threadgroup": [_extent_json(e) for e in self.threadgroup],
            "mode": self.mode.value,
            "threadgroup_memory": [_extent_json(e) for e in self.threadgroup_memory],
        }

    @staticmethod
    def from_json(raw: dict) -> "LaunchSpec":
        return LaunchSpec(
            grid=tuple(raw["grid"]),
            threadgroup=tuple(raw["threadgroup"]),
            mode=LaunchMode(raw.get("mode", LaunchMode.THREADS.value)),
            threadgroup_memory=tuple(raw.get("threadgroup_memory", ())),
        )


# ---------------------------------------------------------------------------
# The candidate kernel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KernelSpec:
    """Raw MSL plus everything the host needs to launch it."""

    source: str
    entry_point: str
    bindings: tuple = ()
    launch: LaunchSpec = field(default_factory=LaunchSpec)
    name: str = ""

    def __post_init__(self):
        object.__setattr__(self, "bindings", tuple(self.bindings))
        if not self.source.strip():
            raise SpecError("kernel source is empty")
        if not self.entry_point:
            raise SpecError("kernel needs an entry point name")
        outputs = [b for b in self.bindings if b.kind is BindingKind.OUTPUT]
        if len(outputs) != 1:
            raise SpecError(
                f"exactly one output binding is required, got {len(outputs)}; "
                "the battery compares a single output tensor per case"
            )
        seen: dict[int, Binding] = {}
        for position, binding in enumerate(self.bindings):
            index = position if binding.index is None else binding.index
            if index < 0:
                raise SpecError(f"buffer index {index} is negative")
            if index in seen:
                raise SpecError(f"buffer index {index} bound twice")
            seen[index] = binding

    def buffer_index(self, position: int) -> int:
        binding = self.bindings[position]
        return position if binding.index is None else binding.index

    def input_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bindings if b.kind is BindingKind.INPUT)

    def scalar_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bindings if b.kind is BindingKind.SCALAR)

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "entry_point": self.entry_point,
            "bindings": [b.to_json() for b in self.bindings],
            "launch": self.launch.to_json(),
            "name": self.name,
        }

    @staticmethod
    def from_json(raw: dict) -> "KernelSpec":
        return KernelSpec(
            source=raw["source"],
            entry_point=raw["entry_point"],
            bindings=tuple(Binding.from_json(b) for b in raw.get("bindings", ())),
            launch=LaunchSpec.from_json(raw["launch"]),
            name=raw.get("name", ""),
        )


# ---------------------------------------------------------------------------
# One case: the inputs, the dimensions, and the output the oracle expects
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class RunCase:
    """One battery case: input tensors, scalar parameters, expected output shape.

    `params` supplies both the scalar bindings the kernel declares and the
    extents the launch refers to, so a case usually passes its schema
    dimensions straight in.
    """

    inputs: dict
    params: dict = field(default_factory=dict)
    output_shape: tuple = ()
    output_dtype: str = "float32"
    label: str = ""

    def __post_init__(self):
        self.output_shape = tuple(int(d) for d in self.output_shape)
        if self.output_dtype not in TENSOR_DTYPES:
            raise SpecError(
                f"output dtype {self.output_dtype!r} is not one of {sorted(TENSOR_DTYPES)}"
            )
        if any(d < 1 for d in self.output_shape):
            raise SpecError(f"output shape {self.output_shape} has a non-positive dimension")

    def validate_against(self, spec: KernelSpec) -> None:
        """Every name the spec binds must exist here, before any process starts."""
        for name in spec.input_names():
            if name not in self.inputs:
                raise SpecError(f"kernel binds input {name!r}, which the case does not supply")
            array = self.inputs[name]
            if str(array.dtype) not in TENSOR_DTYPES:
                raise SpecError(
                    f"input {name!r} has dtype {array.dtype}, "
                    f"expected one of {sorted(TENSOR_DTYPES)}"
                )
        for binding in spec.bindings:
            if binding.kind is BindingKind.SCALAR and binding.name not in self.params:
                raise SpecError(
                    f"kernel binds scalar {binding.name!r}, which the case does not supply"
                )
        spec.launch.resolve(self.params)  # raises on a missing or bad extent

    def to_json(self) -> dict:
        return {
            "inputs": {name: array_to_json(a) for name, a in self.inputs.items()},
            "params": self.params,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "label": self.label,
        }

    @staticmethod
    def from_json(raw: dict) -> "RunCase":
        return RunCase(
            inputs={name: array_from_json(a) for name, a in raw["inputs"].items()},
            params=raw.get("params", {}),
            output_shape=tuple(raw.get("output_shape", ())),
            output_dtype=raw.get("output_dtype", "float32"),
            label=raw.get("label", ""),
        )


# ---------------------------------------------------------------------------
# Array wire form: the same base64 payload shape the corpus references use
# ---------------------------------------------------------------------------
def array_to_json(array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "data": base64.b64encode(contiguous.tobytes()).decode(),
    }


def array_from_json(raw: dict) -> np.ndarray:
    flat = np.frombuffer(base64.b64decode(raw["data"]), dtype=np.dtype(raw["dtype"]))
    return flat.reshape(raw["shape"]).copy()
