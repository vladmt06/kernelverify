"""Compile and dispatch a candidate kernel on the Metal device.

This is the only module that touches the GPU, and it always runs inside the
worker subprocess, never in the caller's process. Everything here is written
against the first-party Metal API through PyObjC rather than a higher-level
array framework, because the contract is raw shading language: the host has to
create the library from source, build the pipeline state, bind buffers at the
indices the shader declares, and choose the grid itself.

Three behaviours are load-bearing for a verifier rather than incidental:

- Every failure is turned into a status. A generated kernel that misses a
  semicolon, names a function that does not exist, or asks for more threads per
  threadgroup than the pipeline allows must come back as a rejected candidate,
  not as an exception that takes down a battery run.
- Launch limits are checked against the compiled pipeline before dispatch.
  Metal treats an oversized threadgroup as a programmer error and aborts the
  process, which would look like a crash rather than the invalid launch it is.
- Correctness and timing are separate dispatches. The returned output comes
  from one clean run into a zeroed buffer, so a kernel that reads its output
  buffer cannot have earlier repeats leak into the tensor the oracle sees.

Buffers are allocated once per case and reused across warmup and timed
repeats, because on unified memory the allocation, not the copy, is what a
short kernel would otherwise spend its time on.
"""

from __future__ import annotations

import time

import numpy as np

import Metal  # PyObjC; absent on non-Apple platforms, which the worker reports

from kernelverify.runners.result import DeviceInfo, RunResult, RunStatus, Timing
from kernelverify.runners.spec import (
    SCALAR_DTYPES,
    TENSOR_DTYPES,
    Binding,
    BindingKind,
    KernelSpec,
    LaunchMode,
    RunCase,
    SpecError,
)

_STORAGE_SHARED = Metal.MTLResourceStorageModeShared
_STATUS_ERROR = 5  # MTLCommandBufferStatusError
_MIN_SCALAR_BYTES = 4  # pad half scalars; Metal buffer arguments are word sized


class CompileError(RuntimeError):
    """The source did not become a compute pipeline."""


class LaunchError(RuntimeError):
    """The pipeline exists but this dispatch cannot be made or did not finish."""


def _size(triple) -> object:
    return Metal.MTLSizeMake(int(triple[0]), int(triple[1]), int(triple[2]))


def _describe(error) -> str:
    if error is None:
        return ""
    text = str(error.localizedDescription()) if hasattr(error, "localizedDescription") else str(error)
    return " ".join(text.split())


class MetalDevice:
    """The system default GPU, plus one command queue shared by every dispatch."""

    def __init__(self):
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("no Metal device on this machine")
        self.device = device
        self.queue = device.newCommandQueue()

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name=str(self.device.name()),
            max_threads_per_threadgroup=int(self.device.maxThreadsPerThreadgroup().width),
            max_threadgroup_memory=int(self.device.maxThreadgroupMemoryLength()),
            has_unified_memory=bool(self.device.hasUnifiedMemory()),
            registry_id=int(self.device.registryID()),
        )

    def compile(self, spec: KernelSpec) -> "CompiledKernel":
        library, error = self.device.newLibraryWithSource_options_error_(spec.source, None, None)
        if library is None:
            raise CompileError(_describe(error) or "the Metal compiler rejected the source")
        function = library.newFunctionWithName_(spec.entry_point)
        if function is None:
            available = [str(n) for n in (library.functionNames() or [])]
            raise CompileError(
                f"no kernel function named {spec.entry_point!r}; "
                f"the source defines {available or 'none'}"
            )
        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:
            raise CompileError(
                _describe(error)
                or f"{spec.entry_point!r} compiled but is not a compute function"
            )
        return CompiledKernel(self, spec, pipeline)


class CompiledKernel:
    """One pipeline state, dispatched over as many cases as the battery has."""

    def __init__(self, device: MetalDevice, spec: KernelSpec, pipeline):
        self.device = device
        self.spec = spec
        self.pipeline = pipeline
        self.max_threads = int(pipeline.maxTotalThreadsPerThreadgroup())
        self.execution_width = int(pipeline.threadExecutionWidth())

    # -- buffer plumbing ---------------------------------------------------
    def _make_buffers(self, case: RunCase):
        """One Metal buffer per tensor binding, plus the packed scalar bytes.

        Returns the buffers by Metal index, the scalar payloads, and the
        buffer indices of the output bindings in binding order, which is the
        order `case.output_shapes` describes and `RunResult.outputs` returns.
        """
        buffers: dict[int, object] = {}
        scalars: dict[int, bytes] = {}
        output_indices: list[int] = []
        for position, binding in enumerate(self.spec.bindings):
            index = self.spec.buffer_index(position)
            if binding.kind is BindingKind.SCALAR:
                scalars[index] = _pack_scalar(binding, case.params[binding.name])
                continue
            if binding.kind is BindingKind.INPUT:
                array = np.ascontiguousarray(case.inputs[binding.name])
                buffer = self.device.device.newBufferWithBytes_length_options_(
                    array.tobytes(), array.nbytes, _STORAGE_SHARED
                )
            else:
                shape, dtype = case.output_shapes[len(output_indices)]
                nbytes = int(np.prod(shape)) * np.dtype(TENSOR_DTYPES[dtype]).itemsize
                buffer = self.device.device.newBufferWithLength_options_(nbytes, _STORAGE_SHARED)
                output_indices.append(index)
            if buffer is None:
                raise LaunchError(f"the device would not allocate the buffer at index {index}")
            buffers[index] = buffer
        return buffers, scalars, output_indices

    def _check_launch(self, grid, group, memory) -> None:
        threads = group[0] * group[1] * group[2]
        if threads > self.max_threads:
            raise LaunchError(
                f"threadgroup {group} is {threads} threads, above the "
                f"{self.max_threads} this pipeline allows"
            )
        limit = int(self.device.device.maxThreadgroupMemoryLength())
        total = sum(memory)
        if total > limit:
            raise LaunchError(
                f"threadgroup memory {total} bytes is above the device limit of {limit}"
            )

    def _encode(self, buffers, scalars, grid, group, memory):
        command_buffer = self.device.queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)
        for index, buffer in buffers.items():
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        for index, payload in scalars.items():
            encoder.setBytes_length_atIndex_(payload, len(payload), index)
        for index, nbytes in enumerate(memory):
            encoder.setThreadgroupMemoryLength_atIndex_(nbytes, index)
        if self.spec.launch.mode is LaunchMode.THREADS:
            encoder.dispatchThreads_threadsPerThreadgroup_(_size(grid), _size(group))
        else:
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(_size(grid), _size(group))
        encoder.endEncoding()
        return command_buffer

    def _dispatch(self, buffers, scalars, grid, group, memory) -> tuple[float, float]:
        command_buffer = self._encode(buffers, scalars, grid, group, memory)
        started = time.perf_counter()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        wall = time.perf_counter() - started
        if int(command_buffer.status()) == _STATUS_ERROR:
            raise LaunchError(_describe(command_buffer.error()) or "the command buffer failed")
        gpu = float(command_buffer.GPUEndTime() - command_buffer.GPUStartTime())
        return gpu, wall

    # -- the public entry point -------------------------------------------
    def run(self, case: RunCase, warmup: int = 1, repeats: int = 5) -> RunResult:
        """One clean dispatch for the output, then warmup and timed repeats."""
        try:
            case.validate_against(self.spec)
            grid, group, memory = self.spec.launch.resolve(case.params)
        except SpecError as error:
            return RunResult(status=RunStatus.INVALID_SPEC, detail=str(error), label=case.label)

        try:
            self._check_launch(grid, group, memory)
            buffers, scalars, output_indices = self._make_buffers(case)
            for slot, index in enumerate(output_indices):
                _zero(buffers[index], *case.output_shapes[slot])
            self._dispatch(buffers, scalars, grid, group, memory)
            outputs = [_read_back(buffers[index], *case.output_shapes[slot])
                       for slot, index in enumerate(output_indices)]
            gpu_samples, wall_samples = [], []
            for repeat in range(warmup + repeats):
                gpu, wall = self._dispatch(buffers, scalars, grid, group, memory)
                if repeat >= warmup:
                    gpu_samples.append(gpu)
                    wall_samples.append(wall)
        except LaunchError as error:
            return RunResult(status=RunStatus.LAUNCH_ERROR, detail=str(error), label=case.label)

        return RunResult(
            status=RunStatus.OK,
            outputs=outputs,
            timing=Timing(tuple(gpu_samples), tuple(wall_samples), warmup),
            label=case.label,
        )


def _pack_scalar(binding: Binding, value) -> bytes:
    dtype = np.dtype(SCALAR_DTYPES[binding.dtype])
    try:
        packed = np.asarray(value, dtype=dtype).tobytes()
    except (TypeError, ValueError) as error:
        raise LaunchError(f"scalar {binding.name!r} cannot hold {value!r}: {error}") from error
    if len(packed) < _MIN_SCALAR_BYTES:
        packed = packed.ljust(_MIN_SCALAR_BYTES, b"\x00")
    return packed


def _output_view(buffer, shape: tuple, dtype: str) -> np.ndarray:
    """A numpy view straight onto the shared buffer; unified memory, no copy."""
    numpy_dtype = np.dtype(TENSOR_DTYPES[dtype])
    nbytes = int(np.prod(shape)) * numpy_dtype.itemsize
    memory = buffer.contents().as_buffer(nbytes)
    return np.frombuffer(memory, dtype=numpy_dtype).reshape(shape)


def _zero(buffer, shape: tuple, dtype: str) -> None:
    """Clear an output before the correctness dispatch.

    A kernel that only writes part of its output would otherwise be judged on
    whatever the allocator handed back, which is not a property of the kernel.
    """
    _output_view(buffer, shape, dtype)[...] = 0


def _read_back(buffer, shape: tuple, dtype: str) -> np.ndarray:
    return _output_view(buffer, shape, dtype).copy()
