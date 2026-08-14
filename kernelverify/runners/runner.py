"""The parent side of the Metal runner: batching, timeouts, crash recovery.

MetalRunner ships a kernel plus a batch of cases to the worker subprocess,
reads streamed per-case results, and owns every failure mode the worker
cannot own about itself: a hung case is killed at `case_timeout` and recorded
as a failure, a crashed worker is relaunched at the next case, and both are
verdicts ("this candidate kernel failed this case"), never harness errors.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class KernelSpec:
    name: str
    source: str
    input_names: list
    output_names: list
    header: str = ""


@dataclass
class DeviceCase:
    inputs: dict            # name -> np.ndarray
    output_shapes: list     # [(shape, dtype_str)]
    grid: tuple
    threadgroup: tuple
    template: dict = field(default_factory=dict)


@dataclass
class CaseResult:
    ok: bool
    outputs: list | None = None   # [np.ndarray]
    error: str = ""


def _encode_case(case: DeviceCase) -> dict:
    return {
        "inputs": {
            name: {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "data": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode(),
            }
            for name, arr in case.inputs.items()
        },
        "outputs": [{"shape": list(s), "dtype": d} for s, d in case.output_shapes],
        "grid": list(case.grid),
        "threadgroup": list(case.threadgroup),
        "template": case.template,
    }


def _decode_outputs(payload: list) -> list:
    return [
        np.frombuffer(base64.b64decode(o["data"]), dtype=np.dtype(o["dtype"]))
        .reshape(o["shape"])
        .copy()
        for o in payload
    ]


class MetalRunner:
    """Executes candidate Metal kernels with per-case isolation semantics."""

    def __init__(self, python: str = sys.executable, case_timeout: float = 30.0):
        self.python = python
        self.case_timeout = case_timeout

    def run(self, kernel: KernelSpec, cases: list) -> list:
        results: list = [None] * len(cases)
        start = 0
        while start < len(cases):
            completed = self._run_batch(kernel, cases, start, results)
            if completed >= len(cases):
                break
            # The case at `completed` hung or killed the worker: record and
            # resume with a fresh worker at the next case.
            if results[completed] is None:
                results[completed] = CaseResult(
                    ok=False, error="worker died or timed out on this case")
            start = completed + 1
        return results

    def _run_batch(self, kernel: KernelSpec, cases: list, start: int,
                   results: list) -> int:
        request = {
            "kernel": {
                "name": kernel.name,
                "source": kernel.source,
                "header": kernel.header,
                "input_names": kernel.input_names,
                "output_names": kernel.output_names,
            },
            "cases": [_encode_case(c) for c in cases[start:]],
        }
        proc = subprocess.Popen(
            [self.python, "-m", "kernelverify.runners.metal_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, cwd=str(_REPO_ROOT),
        )
        try:
            proc.stdin.write(json.dumps(request))
            proc.stdin.close()
        except BrokenPipeError:
            proc.kill()
            return start

        cursor = start
        while True:
            line = self._read_line(proc)
            if line is None:  # timeout or worker death mid-case
                proc.kill()
                proc.wait()
                return cursor
            message = json.loads(line)
            if message.get("done"):
                proc.wait()
                return len(cases)
            index = start + message["case"]
            if "error" in message:
                results[index] = CaseResult(ok=False, error=message["error"])
            else:
                results[index] = CaseResult(
                    ok=True, outputs=_decode_outputs(message["outputs"]))
            cursor = index + 1

    def _read_line(self, proc) -> str | None:
        """One line from the worker, bounded by case_timeout; None on hang/EOF."""
        holder: dict = {}

        def _read():
            holder["line"] = proc.stdout.readline()

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(self.case_timeout)
        if thread.is_alive():
            return None
        line = holder.get("line", "")
        return line if line else None
