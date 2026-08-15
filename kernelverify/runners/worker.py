"""The subprocess that actually touches the GPU.

Run as `python -m kernelverify.runners.worker`. It reads one JSON request from
stdin and streams one JSON line per event to stdout.

The isolation is the point. A candidate kernel is generated code: it can fail
to compile, write past the end of a buffer, or occupy the GPU long enough that
the only way out is to kill the process. None of those are recoverable inside
the process that hosts them, so the parent keeps its distance and reads results
through a pipe.

One session serves one candidate, and a candidate may arrive as several specs:
the same template specialized to different shapes or dtypes. The specs run in
order through the one process, each compiled once and dispatched over its own
cases, because compiling MSL costs far more than dispatching a small kernel.
Isolation stays per candidate: damage done by one spec can only reach its
sibling specializations, never another candidate's verdict.

Results stream rather than arriving as one payload at the end, so a batch that
dies halfway still reports the cases that already finished, and the case that
was in flight when the parent's patience ran out is identified rather than
guessed at.

Wire format, one JSON object per line:

    {"event": "device", "device": {...}}      always first when a GPU is present
    {"event": "compiled", "spec": s, ...}     spec s's pipeline is built
    {"event": "spec_failed", "spec": s, "result": {...}}
                                              spec s cannot run at all; its
                                              siblings still do
    {"event": "case", "spec": s, "index": i, "result": {...}}
    {"event": "fatal", "result": {...}}       nothing could run: no GPU, or an
                                              unreadable request
    {"event": "done"}
"""

from __future__ import annotations

import json
import sys
import traceback

from kernelverify.runners.result import RunResult, RunStatus
from kernelverify.runners.spec import KernelSpec, RunCase, SpecError


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def fatal(status: RunStatus, detail: str) -> int:
    emit({"event": "fatal", "result": RunResult(status=status, detail=detail).to_json()})
    emit({"event": "done"})
    return 0


def _run_case(kernel, raw_case: dict, warmup: int, repeats: int) -> RunResult:
    """Run one case on a compiled kernel, folding every failure into a result."""
    try:
        case = RunCase.from_json(raw_case)
        return kernel.run(case, warmup=warmup, repeats=repeats)
    except SpecError as error:
        return RunResult(status=RunStatus.INVALID_SPEC, detail=str(error),
                         label=raw_case.get("label", ""))
    except Exception:  # a PyObjC or driver failure is the kernel's problem
        return RunResult(status=RunStatus.LAUNCH_ERROR,
                         detail=traceback.format_exc(limit=3).strip(),
                         label=raw_case.get("label", ""))


def main(argv: list[str]) -> int:
    # The import is inside main so that a machine without Metal reports an
    # honest status instead of failing at module import with a traceback.
    try:
        from kernelverify.runners.device import CompileError, MetalDevice
    except ImportError as error:
        return fatal(RunStatus.UNSUPPORTED, f"the Metal bindings are unavailable: {error}")

    try:
        device = MetalDevice()
    except Exception as error:  # no GPU, no driver, headless VM
        return fatal(RunStatus.UNSUPPORTED, str(error))

    emit({"event": "device", "device": device.info().to_json()})
    if "--probe" in argv:
        emit({"event": "done"})
        return 0
    if "--stream" in argv:
        return stream(device)

    try:
        request = json.loads(sys.stdin.read())
    except ValueError as error:
        return fatal(RunStatus.INVALID_SPEC, f"the request was not valid JSON: {error}")
    warmup = int(request.get("warmup", 1))
    repeats = int(request.get("repeats", 5))
    math_mode = request.get("math_mode", "safe")

    for spec_index, entry in enumerate(request.get("specs", [])):
        try:
            spec = KernelSpec.from_json(entry["spec"])
        except (SpecError, KeyError, TypeError) as error:
            emit({"event": "spec_failed", "spec": spec_index,
                  "result": RunResult(status=RunStatus.INVALID_SPEC,
                                      detail=str(error)).to_json()})
            continue

        try:
            kernel = device.compile(spec, math_mode=math_mode)
        except CompileError as error:
            emit({"event": "spec_failed", "spec": spec_index,
                  "result": RunResult(status=RunStatus.COMPILE_ERROR,
                                      detail=str(error)).to_json()})
            continue
        # Compilation is the one unbounded-but-legitimate wait per spec.
        # Saying so lets the parent stop granting the start-up budget to the
        # first case, which would otherwise be the slowest case to detect a
        # hang in.
        # The compile options are echoed rather than assumed, so the parent
        # records what the shader was actually built with; a certificate cites
        # them as inputs.
        emit({"event": "compiled", "spec": spec_index,
              "max_threads_per_threadgroup": kernel.max_threads,
              "thread_execution_width": kernel.execution_width,
              "compile_options": {"math_mode": kernel.math_mode}})

        for index, raw_case in enumerate(entry.get("cases", [])):
            result = _run_case(kernel, raw_case, warmup, repeats)
            emit({"event": "case", "spec": spec_index, "index": index,
                  "result": result.to_json()})

    emit({"event": "done"})
    return 0


def stream(device) -> int:
    """One spec, cases arriving one line at a time, paced by the parent.

    This mode exists for interleaved A/B timing: two of these workers stay
    live, each holding one compiled pipeline, and the parent alternates
    dispatches between them so a power-state excursion lands on both arms.
    The batch mode above cannot do that, because it drains its whole request
    as fast as the GPU allows.

    First line: {"spec": ..., "warmup": w, "repeats": r, "math_mode": m}
    Then per case: {"index": i, "case": {...}} ... and {"end": true} to finish.
    Events mirror the batch mode, with spec index 0.
    """
    from kernelverify.runners.device import CompileError

    header = json.loads(sys.stdin.readline())
    warmup = int(header.get("warmup", 1))
    repeats = int(header.get("repeats", 5))
    try:
        spec = KernelSpec.from_json(header["spec"])
    except (SpecError, KeyError, TypeError) as error:
        return fatal(RunStatus.INVALID_SPEC, str(error))
    try:
        kernel = device.compile(spec, math_mode=header.get("math_mode", "safe"))
    except CompileError as error:
        return fatal(RunStatus.COMPILE_ERROR, str(error))
    emit({"event": "compiled", "spec": 0,
          "max_threads_per_threadgroup": kernel.max_threads,
          "thread_execution_width": kernel.execution_width,
          "compile_options": {"math_mode": kernel.math_mode}})

    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        if message.get("end"):
            break
        raw_case = message.get("case", {})
        index = int(message.get("index", 0))
        result = _run_case(kernel, raw_case, warmup, repeats)
        emit({"event": "case", "spec": 0, "index": index, "result": result.to_json()})

    emit({"event": "done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
