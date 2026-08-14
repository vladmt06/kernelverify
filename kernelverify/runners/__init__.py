"""Backends that execute a candidate kernel and hand its output to the oracle.

The battery decides *which* cases to run and the tolerance decides whether an
output is acceptable. A runner is the part in between: it takes a candidate
kernel as the generator wrote it, executes it on real hardware, and returns a
numpy array plus a status and a timing sample.

`metal` is the Apple silicon backend: raw Metal shading language, compiled once
per candidate and dispatched over a batch of cases inside a worker process the
candidate cannot take down with it.

Importing this package does not import Metal, so a machine without a GPU can
still load the harness; ask `MetalRunner().available()` before running.

The Metal backend needs `pyobjc-framework-Metal`, which is the only dependency
this repository has beyond numpy. It is imported in the worker process alone,
so a machine without it reports an `unsupported` status rather than failing to
start:

    pip install pyobjc-framework-Metal
"""

from kernelverify.runners.metal import BatchResult, MetalRunner
from kernelverify.runners.result import DeviceInfo, RunResult, RunStatus, Timing
from kernelverify.runners.spec import (
    Binding,
    BindingKind,
    KernelSpec,
    LaunchMode,
    LaunchSpec,
    RunCase,
    SpecError,
)
from kernelverify.runners.specialize import specialize

__all__ = [
    "BatchResult",
    "Binding",
    "BindingKind",
    "DeviceInfo",
    "KernelSpec",
    "LaunchMode",
    "LaunchSpec",
    "MetalRunner",
    "RunCase",
    "RunResult",
    "RunStatus",
    "SpecError",
    "Timing",
    "specialize",
]
