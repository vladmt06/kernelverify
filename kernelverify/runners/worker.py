"""The subprocess that actually touches the GPU.

Run as `python -m kernelverify.runners.worker`. It reads one JSON request from
stdin and streams one JSON line per event to stdout.

The isolation is the point. A candidate kernel is generated code: it can fail
to compile, write past the end of a buffer, or occupy the GPU long enough that
the only way out is to kill the process. None of those are recoverable inside
the process that hosts them, so the parent keeps its distance and reads results
through a pipe.

Results stream rather than arriving as one payload at the end, so a batch that
dies halfway still reports the cases that already finished, and the case that
was in flight when the parent's patience ran out is identified rather than
guessed at. The whole batch shares one compilation, because compiling MSL costs
far more than dispatching a small kernel and a battery runs one candidate over
many cases.

Wire format, one JSON object per line:

    {"event": "device", "device": {...}}      always first when a GPU is present
    {"event": "compiled", ...}                the pipeline is built; dispatch begins
    {"event": "case", "index": i, "result": {...}}
    {"event": "fatal", "result": {...}}       nothing could run: no GPU, or no
                                              pipeline built from this source
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

    request = json.loads(sys.stdin.read())
    try:
        spec = KernelSpec.from_json(request["spec"])
    except (SpecError, KeyError, TypeError) as error:
        return fatal(RunStatus.INVALID_SPEC, str(error))

    try:
        kernel = device.compile(spec)
    except CompileError as error:
        return fatal(RunStatus.COMPILE_ERROR, str(error))
    # Compilation is the one unbounded-but-legitimate wait in a batch. Saying
    # so lets the parent stop granting the start-up budget to the first case,
    # which would otherwise be the slowest case to detect a hang in.
    emit({"event": "compiled",
          "max_threads_per_threadgroup": kernel.max_threads,
          "thread_execution_width": kernel.execution_width})

    warmup = int(request.get("warmup", 1))
    repeats = int(request.get("repeats", 5))
    for index, raw_case in enumerate(request.get("cases", [])):
        try:
            case = RunCase.from_json(raw_case)
            result = kernel.run(case, warmup=warmup, repeats=repeats)
        except SpecError as error:
            result = RunResult(status=RunStatus.INVALID_SPEC, detail=str(error),
                               label=raw_case.get("label", ""))
        except Exception:  # a PyObjC or driver failure is the kernel's problem
            result = RunResult(status=RunStatus.LAUNCH_ERROR,
                               detail=traceback.format_exc(limit=3).strip(),
                               label=raw_case.get("label", ""))
        emit({"event": "case", "index": index, "result": result.to_json()})

    emit({"event": "done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
