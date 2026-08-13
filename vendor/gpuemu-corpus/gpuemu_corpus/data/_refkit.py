"""Shared decode/encode helpers for fp64 reference scripts.

The gpuemu daemon feeds a reference script a JSON payload on stdin:
    {"inputs": {name: {shape, strides, dtype, data(base64)}}, "kwargs": {...}}
and expects a single JSON tensor on stdout: {shape, dtype, data(base64)}.

The reference must return the SAME dtype as the kernel output (the validator
compares same-dtype). References here compute in float64 and round back to the
input dtype — i.e. the correctly-rounded "ideal" result.
"""

import base64
import json
import sys

import numpy as np

_DTYPES = {
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "int32": "int32",
    "int64": "int64",
}


def _decode(t: dict) -> np.ndarray:
    dtype = np.dtype(_DTYPES.get(t["dtype"], "float32"))
    arr = np.frombuffer(base64.b64decode(t["data"]), dtype=dtype)
    shape = tuple(t["shape"])
    # Data is generated contiguous (numel row-major elements); strides are
    # layout metadata for the kernel-under-test, not the value layout.
    return arr.reshape(shape) if shape else arr.reshape(())


def read_inputs():
    """Return (inputs: dict[str, np.ndarray], kwargs: dict)."""
    payload = json.load(sys.stdin)
    inputs = {name: _decode(t) for name, t in payload["inputs"].items()}
    return inputs, payload.get("kwargs", {})


def emit(arr: np.ndarray) -> None:
    """Write a tensor to stdout in the gpuemu protocol format."""
    arr = np.ascontiguousarray(arr)
    out = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }
    sys.stdout.write(json.dumps(out))
