"""Run a candidate Metal kernel from a process that the candidate cannot break.

`MetalRunner` is the parent half of the runner. It never imports Metal and
never executes generated code; it starts `kernelverify.runners.worker` in its
own session, hands it one compiled-once batch of cases, and reads results off a
pipe as they arrive.

Three properties are what a verifier needs from this and a plain function call
cannot give:

- A hang is survivable. Nothing in Python can interrupt a dispatched Metal
  command buffer, so the only reliable stop is killing the process group. The
  parent holds a rolling deadline: every result received resets it, and when it
  expires the group dies and the case in flight is reported as a timeout.
- Partial progress survives. Results stream one line at a time, so a batch that
  hangs on case 7 still returns 6 measured cases and names the seventh.
- A crash is a status, not an exception. A kernel that writes outside its
  buffers takes the worker down with it; the parent sees the pipe close early
  and reports the unfinished cases as crashes, with the worker's stderr
  attached.

Cost: one process start and one Metal compile per batch, which is why the API
takes a list of cases rather than one. A fresh process per *candidate* is
deliberate and not amortised away, because reusing a worker across candidates
would let one candidate's damage reach the next one's verdict.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

from kernelverify.runners.result import DeviceInfo, RunResult, RunStatus
from kernelverify.runners.spec import KernelSpec, RunCase, SpecError

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_MODULE = "kernelverify.runners.worker"

DEFAULT_CASE_TIMEOUT = 10.0
DEFAULT_STARTUP_TIMEOUT = 30.0
DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 5
STDERR_LINES = 200  # tail kept from a worker that talks too much

_EOF = object()


@dataclass
class BatchResult:
    """One result per case, in the order the cases were given."""

    results: list = field(default_factory=list)
    device: DeviceInfo | None = None
    stderr: str = ""

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]

    @property
    def all_ok(self) -> bool:
        return all(r.ok for r in self.results)


@dataclass
class MetalRunner:
    """Runs candidate kernels on the GPU through an isolated worker process.

    `case_timeout` is the wall budget for one case including its warmup and
    timed repeats. `startup_timeout` covers the slower events: the worker's
    Python and Metal start-up, and the first case, which pays for compiling the
    shader.
    """

    case_timeout: float = DEFAULT_CASE_TIMEOUT
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT
    warmup: int = DEFAULT_WARMUP
    repeats: int = DEFAULT_REPEATS
    python: str = sys.executable

    # -- device ------------------------------------------------------------
    def probe(self) -> DeviceInfo | None:
        """The GPU the worker sees, or None when there is no Metal device."""
        try:
            process = self._spawn(probe=True)
        except OSError:
            return None
        try:
            process.stdin.close()
            events = _EventStream(process.stdout)
            while True:
                event = events.next_event(self.startup_timeout)
                if event is None:
                    return None
                if event.get("event") == "device":
                    return DeviceInfo.from_json(event["device"])
                if event.get("event") in ("fatal", "done"):
                    return None
        except TimeoutError:
            return None
        finally:
            self._terminate(process)

    def available(self) -> bool:
        return self.probe() is not None

    # -- running -----------------------------------------------------------
    def run_one(self, spec: KernelSpec, case: RunCase, **kwargs) -> RunResult:
        return self.run(spec, [case], **kwargs).results[0]

    def run(self, spec: KernelSpec, cases, *, warmup: int | None = None,
            repeats: int | None = None) -> BatchResult:
        """Compile `spec` once, dispatch it over `cases`, return one result each."""
        cases = list(cases)
        invalid = _validate(spec, cases)
        if invalid is not None:
            return BatchResult(results=[
                RunResult(status=RunStatus.INVALID_SPEC, detail=invalid, label=c.label)
                for c in cases
            ])

        request = {
            "spec": spec.to_json(),
            "cases": [c.to_json() for c in cases],
            "warmup": self.warmup if warmup is None else warmup,
            "repeats": self.repeats if repeats is None else repeats,
        }

        try:
            process = self._spawn()
        except OSError as error:  # no interpreter, no fork: nothing ran
            return BatchResult(results=[
                RunResult(status=RunStatus.CRASH, detail=f"the worker would not start: {error}",
                          label=c.label) for c in cases
            ])
        errors: list[str] = []
        reader = _stream(process.stderr, errors)
        writer = threading.Thread(
            target=_write_request, args=(process, request), daemon=True
        )
        writer.start()

        results: list[RunResult | None] = [None] * len(cases)
        device = None
        fatal: RunResult | None = None
        timed_out = False
        events = _EventStream(process.stdout)
        # Starting Python, opening the device and compiling the shader all take
        # an unpredictable but legitimate amount of time, so they run on the
        # start-up budget. The worker announces the end of compilation, and
        # from there every case is pure dispatch on the per-case budget, which
        # is what makes the first case as quick to time out as the last.
        budget = self.startup_timeout
        try:
            while True:
                event = events.next_event(budget)
                if event is None:  # the worker closed its pipe
                    break
                kind = event.get("event")
                if kind == "device":
                    device = DeviceInfo.from_json(event["device"])
                elif kind == "compiled":
                    budget = self.case_timeout
                elif kind == "case":
                    # A malformed or out-of-range event is treated as if the
                    # case never reported, which is what it means: the stream
                    # is damaged, so the case below fills in as a crash rather
                    # than this loop raising inside a battery run.
                    try:
                        index = int(event["index"])
                        result = RunResult.from_json(event["result"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if 0 <= index < len(cases):
                        result.label = result.label or cases[index].label
                        results[index] = result
                    budget = self.case_timeout
                elif kind == "fatal":
                    fatal = RunResult.from_json(event["result"])
                    break
                elif kind == "done":
                    break
        except TimeoutError:
            timed_out = True
        finally:
            self._terminate(process)
            reader.join(timeout=1.0)

        stderr = "".join(errors).strip()
        filler = _unfinished(fatal, timed_out, stderr)
        for index, result in enumerate(results):
            if result is None:
                results[index] = RunResult(status=filler.status, detail=filler.detail,
                                           label=cases[index].label)
        return BatchResult(results=results, device=device, stderr=stderr)

    # -- process plumbing --------------------------------------------------
    def _spawn(self, probe: bool = False) -> subprocess.Popen:
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
        )
        command = [self.python, "-m", WORKER_MODULE] + (["--probe"] if probe else [])
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
            env=environment,
            start_new_session=True,  # so a hung kernel can be killed by group
        )

    def _terminate(self, process: subprocess.Popen) -> None:
        if process.poll() is None:
            for send in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(process.pid), send)
                except (ProcessLookupError, PermissionError):
                    break
                if _wait_briefly(process, 0.5):
                    break
        for pipe in (process.stdin, process.stdout, process.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass


def _validate(spec: KernelSpec, cases: list[RunCase]) -> str | None:
    """Catch an unusable spec before paying for a process."""
    for case in cases:
        try:
            case.validate_against(spec)
        except SpecError as error:
            return str(error)
    return None


def _unfinished(fatal: RunResult | None, timed_out: bool, stderr: str) -> RunResult:
    """What a case that never reported should be called."""
    if fatal is not None:
        return fatal
    if timed_out:
        return RunResult(status=RunStatus.TIMEOUT,
                         detail="the worker was killed after exceeding its time budget")
    tail = stderr.strip().splitlines()[-4:]
    return RunResult(status=RunStatus.CRASH,
                     detail="the worker exited before reporting this case"
                            + (f": {' / '.join(tail)}" if tail else ""))


def _write_request(process: subprocess.Popen, request: dict) -> None:
    """Feed stdin from its own thread so a full stdout pipe cannot deadlock us."""
    try:
        json.dump(request, process.stdin)
        process.stdin.close()
    except (BrokenPipeError, ValueError, OSError):
        pass


class _EventStream:
    """Worker stdout, one decoded JSON event at a time, on a rolling deadline.

    A background thread does the reading so that waiting for the next event is
    a timed queue get rather than a blocking pipe read, which is what makes a
    hung kernel interruptible from here.
    """

    def __init__(self, pipe):
        self.queue: Queue = Queue()
        self.thread = threading.Thread(target=self._pump, args=(pipe,), daemon=True)
        self.thread.start()

    def _pump(self, pipe) -> None:
        try:
            for line in pipe:
                self.queue.put(line)
        except (ValueError, OSError):
            pass
        finally:
            self.queue.put(_EOF)

    def next_event(self, budget: float) -> dict | None:
        """The next event, None at end of stream, TimeoutError past `budget`."""
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"the worker produced nothing for {budget:.1f}s")
            try:
                line = self.queue.get(timeout=remaining)
            except Empty:
                raise TimeoutError(f"the worker produced nothing for {budget:.1f}s") from None
            if line is _EOF:
                return None
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue  # a stray print from a driver is not an event


def _stream(pipe, sink: list, keep: int = STDERR_LINES) -> threading.Thread:
    """Drain stderr, keeping only the tail: a driver in a bad mood can write a
    great deal of it, and only the last few lines ever explain anything."""

    def drain():
        try:
            for line in pipe:
                sink.append(line)
                if len(sink) > keep:
                    del sink[:-keep]
        except (ValueError, OSError):
            pass

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return thread


def _wait_briefly(process: subprocess.Popen, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.01)
    return False
