"""Parent-side drivers for the MLX subprocesses: capture and live arm.

Both spawn `kernelverify.extraction._mlx_worker` and neither ever imports MLX
into the caller's process. The capture arm is fresh-process-per-specialization
by construction: one call, one dump, one record. The live arm is a disposable
batch runner with crash isolation and no verdict authority - a case that
raises inside MLX comes back as an error string, a worker that hangs or dies
is killed and its unreported cases are named, and nothing here judges an
array.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from kernelverify.extraction.surface import LiveCall, MLXKernelSurface
from kernelverify.runners.spec import array_from_json

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_MODULE = "kernelverify.extraction._mlx_worker"

DEFAULT_CAPTURE_TIMEOUT = 120.0
DEFAULT_LIVE_TIMEOUT = 300.0

# One verbose dump, as mlx 0.32.0 prints it. DOTALL because the source spans
# lines; non-greedy so multiple dumps in one stream parse as multiple blocks.
_DUMP_BLOCK = re.compile(
    r"Generated source code for `(?P<name>[^`]+)`:\n```\n(?P<source>.*?)\n```",
    re.DOTALL,
)
_HOST_NAME = re.compile(r'\[\[host_name\("(?P<name>[^"]+)"\)\]\]')
_KERNEL_FN = re.compile(r"\[\[kernel\]\]\s+void\s+(?P<name>\w+)\s*\(")


class ExtractionError(RuntimeError):
    """The capture or live arm could not produce what was asked of it."""


@dataclass(frozen=True)
class CaptureRecord:
    """One specialization's extracted source, with its provenance.

    `source` is the translation unit exactly as MLX generated it; `entry_point`
    is the host-visible kernel function name (the `host_name` instantiation for
    templated kernels). `outputs` are the capturing call's own results - the
    first live data point, produced by the very compilation that was dumped.
    """

    surface_name: str
    raw_dump: str
    source: str
    entry_point: str
    outputs: list
    mlx_version: str
    math_mode: str


@dataclass
class LiveResult:
    """What the live arm returned for one case. Arrays only, never a verdict."""

    ok: bool
    outputs: list = field(default_factory=list)
    error: str = ""
    label: str = ""


def _spawn(python: str) -> subprocess.Popen:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )
    return subprocess.Popen(
        [python, "-m", WORKER_MODULE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(REPO_ROOT),
        env=environment,
        start_new_session=True,
    )


def _communicate(process: subprocess.Popen, payload: str, timeout: float):
    """stdin in, (stdout, stderr) out; on a hang, kill and keep what arrived."""
    try:
        return process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        out, err = process.communicate()
        return out, (err or "") + f"\n[killed after {timeout:.0f}s]"


def parse_dump(raw: str) -> list[tuple[str, str, str]]:
    """(kernel name, translation unit, entry point) per dump block."""
    parsed = []
    for match in _DUMP_BLOCK.finditer(raw):
        source = match.group("source")
        host = _HOST_NAME.search(source)
        if host is not None:
            entry = host.group("name")
        else:
            fn = _KERNEL_FN.search(source)
            entry = fn.group("name") if fn else ""
        parsed.append((match.group("name"), source, entry))
    return parsed


def capture_specialization(surface: MLXKernelSurface, call: LiveCall, *,
                           python: str = sys.executable,
                           timeout: float = DEFAULT_CAPTURE_TIMEOUT) -> CaptureRecord:
    """Extract the MSL one specialization compiles to, in a fresh process."""
    with tempfile.TemporaryDirectory(prefix="kv-capture-") as workdir:
        dump_path = os.path.join(workdir, "dump.metal.txt")
        result_path = os.path.join(workdir, "result.json")
        request = json.dumps({
            "surface": surface.to_json(),
            "calls": [call.to_json()],
            "capture": {"dump_path": dump_path, "result_path": result_path},
        })
        process = _spawn(python)
        _, stderr = _communicate(process, request, timeout)

        try:
            with open(result_path) as sink:
                payload = json.load(sink)
        except (OSError, json.JSONDecodeError):
            raise ExtractionError(
                "the capture worker died before reporting: "
                + (stderr or "").strip()[-500:]
            ) from None
        if not payload.get("ok"):
            raise ExtractionError(f"the capturing call failed: {payload.get('error', '')}")

        try:
            with open(dump_path) as sink:
                raw_dump = sink.read()
        except OSError as error:
            raise ExtractionError(f"no dump was written: {error}") from None

    blocks = parse_dump(raw_dump)
    if len(blocks) != 1:
        raise ExtractionError(
            f"expected exactly one generated-source dump, found {len(blocks)}; "
            "a capture process must make exactly one call"
        )
    name, source, entry = blocks[0]
    if not entry:
        raise ExtractionError("no [[kernel]] function found in the extracted source")
    return CaptureRecord(
        surface_name=name,
        raw_dump=raw_dump,
        source=source,
        entry_point=entry,
        outputs=[array_from_json(o) for o in payload.get("outputs", ())],
        mlx_version=payload.get("mlx_version", ""),
        math_mode=surface.math_mode,
    )


def run_live(surface: MLXKernelSurface, calls, *, python: str = sys.executable,
             timeout: float = DEFAULT_LIVE_TIMEOUT) -> list[LiveResult]:
    """Run the surface natively over `calls`; one `LiveResult` per call."""
    calls = list(calls)
    request = json.dumps({
        "surface": surface.to_json(),
        "calls": [c.to_json() for c in calls],
        "capture": None,
    })
    process = _spawn(python)
    stdout, stderr = _communicate(process, request, timeout)

    results: list[LiveResult | None] = [None] * len(calls)
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("done"):
            break
        index = event.get("case")
        if not isinstance(index, int) or not 0 <= index < len(calls):
            continue
        if "error" in event:
            results[index] = LiveResult(ok=False, error=event["error"],
                                        label=calls[index].label)
        else:
            results[index] = LiveResult(
                ok=True,
                outputs=[array_from_json(o) for o in event.get("outputs", ())],
                label=calls[index].label,
            )

    tail = (stderr or "").strip().splitlines()[-4:]
    missing = "the live arm died before this case" + (f": {' / '.join(tail)}" if tail else "")
    return [
        result if result is not None
        else LiveResult(ok=False, error=missing, label=calls[index].label)
        for index, result in enumerate(results)
    ]
