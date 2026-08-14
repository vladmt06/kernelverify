"""What a run returns: a status, an output tensor, and a timing sample.

The status exists because a generated kernel fails in ways a numerical oracle
cannot interpret. A kernel that does not compile, that Metal refuses to
dispatch, or that never returns has no output tensor at all, and reporting that
as "the outputs differ" would confuse a broken harness with a caught fault. So
execution failures are named here, and the verifier decides what each one
means. The mapping that keeps the verdict honest:

- OK: the oracle compares the output as usual.
- COMPILE_ERROR, LAUNCH_ERROR: the candidate is rejected. It is not a working
  kernel, and the compiler diagnostic in `detail` says why.
- TIMEOUT: rejected. A kernel that does not finish inside a budget calibrated
  on the reference is not a faster kernel, which is the whole point of the
  optimiser phase.
- CRASH: rejected, and worth surfacing separately, because a worker that dies
  mid-batch usually means the kernel wrote outside its buffers.
- INVALID_SPEC: not a verdict on the kernel at all. The launch description is
  unusable, so nothing ran; fix the spec and rerun.
- UNSUPPORTED: no Metal device on this machine, so the battery cannot run here.
  Never a verdict on the kernel.

Timing carries every sample rather than one number, because the optimiser
compares candidates that can differ by a few percent and needs to see the
spread before believing a win. `gpu_seconds` comes from the command buffer's
own GPU timestamps, so it excludes host encode overhead; `wall_seconds` is the
commit-to-complete time the caller actually pays.

Compare candidates on `gpu_best`, the minimum, because interference only ever
makes a kernel look slower and never faster.

That estimator still has a resolution floor, and it is higher than it looks.
On an idle M3 Pro the samples are tight at every scale, median over minimum
between 1.00 and 1.07 from 2.4 microseconds up. On the same machine with other
work running, the picture changes completely. Best-of-5 reproducibility over
12 runs, and the spread within a single 24-repeat run:

    2.5 us of GPU work   24.1x across runs                unmeasurable
     22 us                2.75x across runs               unmeasurable
    200 us                up to 4x across runs, 1.35x within  unreliable
   14.5 ms                1.05x across runs, 1.001 within     solid

The short-kernel variation is not random jitter. At 200 microseconds the
samples move together, drifting from 150 to 200 microseconds across one run
and sitting near 600 in another, which is what a GPU changing power state
looks like rather than an outlier being sampled.

So rank candidates on kernels that run for milliseconds. A speed claim made on
a shape small enough to finish in microseconds is not a measurement, however
many repeats it averages. `gpu_spread` is the per-run diagnostic that says when
a particular number should not be believed at all.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum

import numpy as np

from kernelverify.runners.spec import array_from_json, array_to_json


class RunStatus(str, Enum):
    OK = "ok"
    COMPILE_ERROR = "compile_error"
    LAUNCH_ERROR = "launch_error"
    TIMEOUT = "timeout"
    CRASH = "crash"
    INVALID_SPEC = "invalid_spec"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Timing:
    """GPU and wall samples from the timed repeats, warmup excluded."""

    gpu_seconds: tuple = ()
    wall_seconds: tuple = ()
    warmup: int = 0

    @property
    def repeats(self) -> int:
        return len(self.gpu_seconds)

    @property
    def gpu_best(self) -> float:
        return min(self.gpu_seconds) if self.gpu_seconds else float("nan")

    @property
    def gpu_median(self) -> float:
        return statistics.median(self.gpu_seconds) if self.gpu_seconds else float("nan")

    @property
    def wall_median(self) -> float:
        return statistics.median(self.wall_seconds) if self.wall_seconds else float("nan")

    @property
    def gpu_spread(self) -> float:
        """Worst over best. A candidate whose spread is wide has not been measured."""
        if not self.gpu_seconds or self.gpu_best <= 0.0:
            return float("nan")
        return max(self.gpu_seconds) / self.gpu_best

    def to_json(self) -> dict:
        return {"gpu_seconds": list(self.gpu_seconds),
                "wall_seconds": list(self.wall_seconds),
                "warmup": self.warmup}

    @staticmethod
    def from_json(raw: dict) -> "Timing":
        return Timing(gpu_seconds=tuple(raw.get("gpu_seconds", ())),
                      wall_seconds=tuple(raw.get("wall_seconds", ())),
                      warmup=int(raw.get("warmup", 0)))


@dataclass
class RunResult:
    status: RunStatus = RunStatus.OK
    output: np.ndarray | None = None
    timing: Timing | None = None
    detail: str = ""
    label: str = ""

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.OK

    def to_json(self) -> dict:
        return {
            "status": self.status.value,
            "output": array_to_json(self.output) if self.output is not None else None,
            "timing": self.timing.to_json() if self.timing is not None else None,
            "detail": self.detail,
            "label": self.label,
        }

    @staticmethod
    def from_json(raw: dict) -> "RunResult":
        return RunResult(
            status=RunStatus(raw["status"]),
            output=array_from_json(raw["output"]) if raw.get("output") else None,
            timing=Timing.from_json(raw["timing"]) if raw.get("timing") else None,
            detail=raw.get("detail", ""),
            label=raw.get("label", ""),
        )


@dataclass(frozen=True)
class DeviceInfo:
    """What the worker found on the other side, for the record in a report."""

    name: str = ""
    max_threads_per_threadgroup: int = 0
    max_threadgroup_memory: int = 0
    has_unified_memory: bool = True
    registry_id: int = 0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "max_threads_per_threadgroup": self.max_threads_per_threadgroup,
            "max_threadgroup_memory": self.max_threadgroup_memory,
            "has_unified_memory": self.has_unified_memory,
            "registry_id": self.registry_id,
        }

    @staticmethod
    def from_json(raw: dict) -> "DeviceInfo":
        return DeviceInfo(**{k: raw[k] for k in raw if k in DeviceInfo.__annotations__})
