"""What an MLX kernel surface is, and one concrete call to it.

`MLXKernelSurface` mirrors the arguments of `mx.fast.metal_kernel`: it is a
description of the kernel as the surface's author wrote it, template holes and
all. `LiveCall` is one invocation - concrete inputs, output descriptions in
the same `(shape, dtype)` convention as `RunCase.output_shapes`, the grid, and
the template values that pick the specialization.

Both serialize to JSON because they cross a process boundary twice: once into
the capture worker, once into the live arm. Arrays use the same base64 wire
form as the runner transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kernelverify.runners.spec import array_from_json, array_to_json

#: MLX's documented default math mode for custom kernels; Metal's own default
#: is fast. Surfaces that deliberately compile differently say so here.
MLX_DEFAULT_MATH_MODE = "safe"


@dataclass(frozen=True)
class MLXKernelSurface:
    """An `mx.fast.metal_kernel` surface, exactly as its author declares it."""

    name: str
    source: str
    input_names: tuple
    output_names: tuple
    header: str = ""
    ensure_row_contiguous: bool = True
    atomic_outputs: bool = False
    math_mode: str = MLX_DEFAULT_MATH_MODE

    def __post_init__(self):
        object.__setattr__(self, "input_names", tuple(self.input_names))
        object.__setattr__(self, "output_names", tuple(self.output_names))

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "header": self.header,
            "ensure_row_contiguous": self.ensure_row_contiguous,
            "atomic_outputs": self.atomic_outputs,
            "math_mode": self.math_mode,
        }

    @staticmethod
    def from_json(raw: dict) -> "MLXKernelSurface":
        return MLXKernelSurface(
            name=raw["name"],
            source=raw["source"],
            input_names=tuple(raw["input_names"]),
            output_names=tuple(raw["output_names"]),
            header=raw.get("header", ""),
            ensure_row_contiguous=raw.get("ensure_row_contiguous", True),
            atomic_outputs=raw.get("atomic_outputs", False),
            math_mode=raw.get("math_mode", MLX_DEFAULT_MATH_MODE),
        )


@dataclass
class LiveCall:
    """One invocation of a surface: inputs, outputs, grid, specialization.

    `template` values may be dtype names ("float16"), integers, or booleans;
    the worker maps dtype names onto `mx` dtypes. The grid counts total
    threads, which is MLX's semantic and the runner's `LaunchMode.THREADS`.
    """

    inputs: dict
    output_shapes: tuple = ()
    grid: tuple = (1, 1, 1)
    threadgroup: tuple = (1, 1, 1)
    template: tuple = ()
    label: str = ""

    def __post_init__(self):
        self.output_shapes = tuple(
            (tuple(int(d) for d in shape), dtype) for shape, dtype in self.output_shapes
        )
        self.grid = tuple(int(g) for g in self.grid)
        self.threadgroup = tuple(int(t) for t in self.threadgroup)
        self.template = tuple((str(name), value) for name, value in self.template)

    def to_json(self) -> dict:
        return {
            "inputs": {name: array_to_json(a) for name, a in self.inputs.items()},
            "output_shapes": [{"shape": list(shape), "dtype": dtype}
                              for shape, dtype in self.output_shapes],
            "grid": list(self.grid),
            "threadgroup": list(self.threadgroup),
            "template": [[name, value] for name, value in self.template],
            "label": self.label,
        }

    @staticmethod
    def from_json(raw: dict) -> "LiveCall":
        return LiveCall(
            inputs={name: array_from_json(a) for name, a in raw["inputs"].items()},
            output_shapes=tuple((tuple(o["shape"]), o["dtype"])
                                for o in raw.get("output_shapes", ())),
            grid=tuple(raw.get("grid", (1, 1, 1))),
            threadgroup=tuple(raw.get("threadgroup", (1, 1, 1))),
            template=tuple((n, v) for n, v in raw.get("template", ())),
            label=raw.get("label", ""),
        )
