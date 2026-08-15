"""Compile and dispatch a candidate kernel on the Metal device.

This is the only module that touches the GPU, and it always runs inside the
worker subprocess, never in the caller's process. Everything here is written
against the first-party Metal API through PyObjC rather than a higher-level
array framework, because the contract is raw shading language: the host has to
create the library from source, build the pipeline state, bind buffers at the
indices the shader declares, and choose the grid itself.

Four behaviours are load-bearing for a verifier rather than incidental:

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
- Buffers are POOLED on the device and reused across cases, never allocated
  per case. Measured 2026-08-15 on this stack (PyObjC on Darwin 25): a
  released MTLBuffer's dirty pages NEVER return to the OS while the process
  lives - +4.5 MB footprint per dropped 4.5 MB buffer, linear over hundreds
  of buffers, immune to autorelease-pool drains and gc.collect, identical
  for newBufferWithBytes and for newBufferWithLength once written. Per-case
  allocation therefore IS a leak: it grew the serving calibration by
  +0.546 GB per 96-dispatch sweep at (1024, 2560) and ~0.83 GB per dispatch
  at lm_head (151936, 2560), the mechanism behind that night's three Jetsam
  kills at 66.7, 69.4 and 39.5 GB, invisible to RSS and to
  mx.get_cache_memory().

  What the pool holds, exactly: one buffer per BINDING INDEX, living on the
  MetalDevice and therefore shared by every case AND every spec that device
  compiles - worker.py runs many specs against one device, so buffer index 2
  is one allocation for every kernel in a batch, not one per kernel. Each
  index's buffer is grown to the largest extent ever bound there and never
  shrunk, so device memory is bounded by the SUM over bound indices of the
  largest extent ever bound at each, not by a single case. `release_pool`
  gives that memory back between specs.

  Reuse is sound because every case reads deterministic bytes and nothing
  else: dispatches are synchronous (commit then waitUntilCompleted before
  the next case), each case rewrites every input binding in full, each
  output is zeroed before the correctness dispatch, and the slack past the
  current case's extent is zeroed whenever a pooled buffer is larger than
  the case receiving it. That last one is what a fresh allocation gave for
  free: before the pool, a candidate reading past its declared extent read a
  fresh page's zeros, and with an uncleared pool it would read whichever case
  last used that index - a verdict that depends on what ran before it, which
  is the one property a verifier may never have. A candidate that scribbles
  outside its own bindings can therefore only corrupt bytes that the next
  case overwrites or zeroes before reading, which is a wrong kernel failing,
  never a wrong kernel passing.

Within one case, the pooled buffers are also what warmup and timed repeats
reuse, because on unified memory the allocation, not the copy, is what a
short kernel would otherwise spend its time on. Each case additionally
drains an autorelease pool, so the autoreleased command-buffer and encoder
temporaries cannot accumulate over a long battery.
"""

from __future__ import annotations

import time

import numpy as np

import objc  # PyObjC core: the per-case autorelease pool
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

# Metal math modes, named the way MLX's compile_options documents them. The
# choice matters to a verifier: Metal's own default is FAST math, MLX compiles
# with SAFE (IEEE special values preserved), and a candidate judged under a
# different math regime than the framework it is meant to replace can pass or
# fail on reassociation rather than on its own arithmetic. The runner defaults
# to MLX's documented default and records what it used, so a certificate can
# cite the compile options as an input.
MATH_MODES = {
    "safe": Metal.MTLMathModeSafe,
    "relaxed": Metal.MTLMathModeRelaxed,
    "fast": Metal.MTLMathModeFast,
}
DEFAULT_MATH_MODE = "safe"  # MLX's documented default, mirrored deliberately


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
        self._buffer_pool: dict[int, tuple[int, object]] = {}

    def pooled_buffer(self, index: int, nbytes: int) -> tuple[object, int]:
        """``(buffer, bytes held)``: a shared-storage buffer of at least
        ``nbytes`` for this binding index, reused across cases and across
        specs (see the module docstring for why per-case allocation is a leak
        on this stack). Callers rewrite the region they use, dispatch
        synchronously, and read back a copy, so reuse can never alias a live
        result.

        The held size is returned because it is routinely LARGER than the
        case asked for, and the caller is entitled to know the extent it was
        handed. Everything from ``nbytes`` to that extent is zeroed before
        the buffer changes hands - whenever the case is smaller than the
        buffer it gets, not only on the first such case, because a candidate
        may have written outside its own bindings on any dispatch and nothing
        here can know that it did not.
        """
        if nbytes <= 0:
            raise LaunchError(
                f"buffer index {index} asks for {nbytes} bytes: a zero-sized "
                f"binding has nothing to bind, and before the pool it was a "
                f"nil buffer from newBufferWithLength(0)")
        held_bytes, held_buffer = self._buffer_pool.get(index, (0, None))
        if held_buffer is not None and held_bytes >= nbytes:
            if held_bytes > nbytes:
                _zero_span(held_buffer, nbytes, held_bytes)
            return held_buffer, held_bytes
        buffer = self.device.newBufferWithLength_options_(nbytes, _STORAGE_SHARED)
        if buffer is None:
            raise LaunchError(
                f"the device would not allocate {nbytes} bytes for buffer "
                f"index {index}")
        self._buffer_pool[index] = (nbytes, buffer)
        return buffer, nbytes

    def release_pool(self) -> None:
        """Drop every pooled buffer and give its memory back to the OS.

        The two steps are both load-bearing, measured on this stack
        2026-08-15 with 0.54 GB buffers: dropping a written MTLBuffer plainly
        returns NOTHING while the process lives (+2.15 GB of footprint
        retained after four of them were dropped and collected), while
        marking it MTLPurgeableStateEmpty first and then dropping it returns
        all of it (+0.00 GB for the same four from a clean process). Empty
        discards the contents there and then, which is right here and wrong
        anywhere else: nothing may read a released buffer again, and nothing
        does - the next `pooled_buffer` allocates a new one.

        Callers release between units of work that do not share buffers, so
        one spec's largest case cannot charge the process for the life of a
        batch (worker.py releases between specs).
        """
        for _held_bytes, buffer in self._buffer_pool.values():
            buffer.setPurgeableState_(Metal.MTLPurgeableStateEmpty)
        self._buffer_pool.clear()

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name=str(self.device.name()),
            max_threads_per_threadgroup=int(self.device.maxThreadsPerThreadgroup().width),
            max_threadgroup_memory=int(self.device.maxThreadgroupMemoryLength()),
            has_unified_memory=bool(self.device.hasUnifiedMemory()),
            registry_id=int(self.device.registryID()),
        )

    def compile(self, spec: KernelSpec,
                math_mode: str = DEFAULT_MATH_MODE) -> "CompiledKernel":
        if math_mode not in MATH_MODES:
            raise CompileError(
                f"math mode {math_mode!r} is not one of {sorted(MATH_MODES)}")
        options = Metal.MTLCompileOptions.new()
        options.setMathMode_(MATH_MODES[math_mode])
        library, error = self.device.newLibraryWithSource_options_error_(
            spec.source, options, None)
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
        return CompiledKernel(self, spec, pipeline, math_mode)


class CompiledKernel:
    """One pipeline state, dispatched over as many cases as the battery has."""

    def __init__(self, device: MetalDevice, spec: KernelSpec, pipeline,
                 math_mode: str = DEFAULT_MATH_MODE):
        self.device = device
        self.spec = spec
        self.pipeline = pipeline
        self.math_mode = math_mode
        self.max_threads = int(pipeline.maxTotalThreadsPerThreadgroup())
        self.execution_width = int(pipeline.threadExecutionWidth())

    # -- buffer plumbing ---------------------------------------------------
    def _make_buffers(self, case: RunCase):
        """The pooled Metal buffer per tensor binding (inputs rewritten in
        full for this case), plus the packed scalar bytes.

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
                buffer, _held_bytes = self.device.pooled_buffer(index, array.nbytes)
                _write_into(buffer, array)
            else:
                shape, dtype = case.output_shapes[len(output_indices)]
                nbytes = int(np.prod(shape)) * np.dtype(TENSOR_DTYPES[dtype]).itemsize
                buffer, _held_bytes = self.device.pooled_buffer(index, nbytes)
                output_indices.append(index)
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
            with objc.autorelease_pool():
                self._check_launch(grid, group, memory)
                buffers, scalars, output_indices = self._make_buffers(case)
                for slot, index in enumerate(output_indices):
                    _zero(buffers[index], *case.output_shapes[slot])
                self._dispatch(buffers, scalars, grid, group, memory)
                outputs = [_read_back(buffers[index], *case.output_shapes[slot])
                           for slot, index in enumerate(output_indices)]
                gpu_samples, wall_samples = [], []
                for repeat in range(warmup + repeats):
                    gpu, wall = self._dispatch(buffers, scalars, grid, group,
                                               memory)
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


def _zero_span(buffer, start: int, stop: int) -> None:
    """Clear ``[start, stop)`` of a pooled buffer: the slack a smaller case
    inherits from whatever used this binding index before it."""
    memory = buffer.contents().as_buffer(stop)
    np.frombuffer(memory, dtype=np.uint8, count=stop - start,
                  offset=start)[...] = 0


def _write_into(buffer, array: np.ndarray) -> None:
    """Copy this case's input into its pooled buffer, in full."""
    view = np.frombuffer(buffer.contents().as_buffer(array.nbytes),
                         dtype=array.dtype)
    view[...] = array.ravel()


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
