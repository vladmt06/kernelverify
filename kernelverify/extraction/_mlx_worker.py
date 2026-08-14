"""The MLX-side subprocess: capture one specialization, or run cases live.

Run as `python -m kernelverify.extraction._mlx_worker`. One JSON request on
stdin:

    {"surface": {...}, "calls": [...],
     "capture": null | {"dump_path": ..., "result_path": ...}}

Live mode (`capture` null) streams one JSON line per case on stdout -
`{"case": i, "outputs": [...]}` or `{"case": i, "error": "..."}` - then
`{"done": true, "mlx_version": ...}`. This arm has no verdict authority; it
only produces arrays for the parent to judge.

Capture mode rebinds file descriptor 1 to `dump_path` before MLX runs,
because the verbose dump is written by the C++ layer straight to fd 1 and a
Python-level redirect captures nothing (measured on mlx 0.32.0). Exactly one
call is allowed per capture process: with a single dump on the stream there
is nothing to mis-attribute, whatever a future MLX does about cache reprints.
The call's outputs and status go to `result_path` as JSON, since stdout is
the dump.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import traceback

import numpy as np

from kernelverify.extraction.surface import LiveCall, MLXKernelSurface
from kernelverify.runners.spec import array_to_json

_MX_DTYPES = {}  # filled after the mlx import, which must wait for fd games


def _flush_c_stdout() -> None:
    ctypes.CDLL(None).fflush(None)


def _template_args(call: LiveCall, mx):
    resolved = [(name, _MX_DTYPES.get(value, value) if isinstance(value, str) else value)
                for name, value in call.template]
    return resolved or None


def _run_call(kernel, surface: MLXKernelSurface, call: LiveCall, mx, verbose: bool):
    inputs = [mx.array(np.ascontiguousarray(call.inputs[name]))
              for name in surface.input_names]
    outputs = kernel(
        inputs=inputs,
        output_shapes=[tuple(shape) for shape, _ in call.output_shapes],
        output_dtypes=[_MX_DTYPES[dtype] for _, dtype in call.output_shapes],
        grid=tuple(call.grid),
        threadgroup=tuple(call.threadgroup),
        template=_template_args(call, mx),
        verbose=verbose,
    )
    mx.eval(*outputs)
    return [array_to_json(np.ascontiguousarray(np.array(o))) for o in outputs]


def main() -> int:
    request = json.loads(sys.stdin.read())
    surface = MLXKernelSurface.from_json(request["surface"])
    calls = [LiveCall.from_json(c) for c in request.get("calls", [])]
    capture = request.get("capture")

    if capture is not None and len(calls) != 1:
        sys.stderr.write("capture mode takes exactly one call per process\n")
        return 2

    if capture is not None:
        # The dump file takes over fd 1 before MLX loads, so the one verbose
        # dump this process ever prints is the whole content of the file.
        dump = os.open(capture["dump_path"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(dump, 1)
        os.close(dump)

    import mlx.core as mx  # noqa: E402  (fd 1 must be settled first)

    _MX_DTYPES.update({
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "int32": mx.int32,
        "int64": mx.int64,
        "uint32": mx.uint32,
    })

    kernel = mx.fast.metal_kernel(
        name=surface.name,
        input_names=list(surface.input_names),
        output_names=list(surface.output_names),
        source=surface.source,
        header=surface.header,
        ensure_row_contiguous=surface.ensure_row_contiguous,
        atomic_outputs=surface.atomic_outputs,
        compile_options={"math_mode": surface.math_mode},
    )

    if capture is not None:
        try:
            outputs = _run_call(kernel, surface, calls[0], mx, verbose=True)
            payload = {"ok": True, "outputs": outputs, "mlx_version": mx.__version__}
        except Exception:
            payload = {"ok": False, "error": traceback.format_exc(limit=4).strip(),
                       "mlx_version": mx.__version__}
        _flush_c_stdout()  # the C++ dump buffer drains into the dump file
        with open(capture["result_path"], "w") as sink:
            json.dump(payload, sink)
        return 0 if payload["ok"] else 1

    for index, call in enumerate(calls):
        try:
            outputs = _run_call(kernel, surface, call, mx, verbose=False)
            line = {"case": index, "outputs": outputs}
        except Exception as error:
            line = {"case": index, "error": f"{type(error).__name__}: {error}"}
        sys.stdout.write(json.dumps(line) + "\n")
        sys.stdout.flush()
    sys.stdout.write(json.dumps({"done": True, "mlx_version": mx.__version__}) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
