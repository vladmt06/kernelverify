"""The Metal worker: a crash-isolated subprocess that executes candidate kernels.

Run as `python -m kernelverify.runners.metal_worker`. Protocol (eng review
decisions 1A + 5A): one JSON request on stdin, newline-delimited JSON results
on stdout - one line per case as it completes, then {"done": true}. The parent
owns timeouts and relaunches; this process owns nothing but execution, so a
kernel that hangs or kills the GPU context takes down this process and only
this process.

Request shape:
    {
      "kernel": {
        "name": str,
        "source": str,           # MSL body for mx.fast.metal_kernel
        "header": str,           # optional includes/defines
        "input_names": [str],
        "output_names": [str],
      },
      "cases": [
        {
          "inputs":  {name: {"shape": [...], "dtype": "float16", "data": b64}},
          "outputs": [{"shape": [...], "dtype": "float16"}],
          "grid": [x, y, z],
          "threadgroup": [x, y, z],
          "template": {name: value},   # optional, e.g. {"T": "float16"}
        }, ...
      ]
    }

Result lines: {"case": i, "outputs": [{"shape", "dtype", "data": b64}]}
          or  {"case": i, "error": "..."}.
The kernel compiles once (first call JIT-compiles; subsequent cases reuse it),
which is the batching decision 5A exists for.
"""

from __future__ import annotations

import base64
import json
import sys

import mlx.core as mx
import numpy as np

_MX_DTYPES = {
    "float16": mx.float16,
    "float32": mx.float32,
    "bfloat16": mx.bfloat16,
    "int32": mx.int32,
    "uint32": mx.uint32,
}


def _decode(spec: dict) -> mx.array:
    arr = np.frombuffer(base64.b64decode(spec["data"]), dtype=np.dtype(spec["dtype"]))
    return mx.array(arr.reshape(spec["shape"]))


def _encode(arr: mx.array) -> dict:
    host = np.array(arr)
    return {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "data": base64.b64encode(np.ascontiguousarray(host).tobytes()).decode(),
    }


def main() -> int:
    request = json.loads(sys.stdin.read())
    spec = request["kernel"]
    kernel = mx.fast.metal_kernel(
        name=spec["name"],
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        source=spec["source"],
        header=spec.get("header", ""),
    )

    for index, case in enumerate(request["cases"]):
        try:
            inputs = [_decode(case["inputs"][name]) for name in spec["input_names"]]
            template = [(k, _MX_DTYPES.get(v, v)) for k, v in
                        case.get("template", {}).items()]
            outputs = kernel(
                inputs=inputs,
                output_shapes=[tuple(o["shape"]) for o in case["outputs"]],
                output_dtypes=[_MX_DTYPES[o["dtype"]] for o in case["outputs"]],
                grid=tuple(case["grid"]),
                threadgroup=tuple(case["threadgroup"]),
                template=template or None,
            )
            mx.eval(*outputs)
            line = {"case": index, "outputs": [_encode(o) for o in outputs]}
        except Exception as exc:  # surfaced to the parent, never swallowed
            line = {"case": index, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(line) + "\n")
        sys.stdout.flush()

    sys.stdout.write(json.dumps({"done": True}) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
