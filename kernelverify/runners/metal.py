"""Run a candidate Metal kernel from a process that the candidate cannot break.

`MetalRunner` is the parent half of the runner. It never imports Metal and
never executes generated code; it starts `kernelverify.runners.worker` in its
own session, hands it one candidate's work, and reads results off a pipe as
they arrive.

The unit of isolation is the candidate, and a candidate may arrive as several
specs: the same template specialized to different shapes or dtypes. All specs
of one candidate share one worker session through `run_candidate`, paying one
process start and one compile per spec. A fresh process per *candidate* is
deliberate and not amortised away, because reusing a worker across candidates
would let one candidate's damage reach the next one's verdict.

Three properties are what a verifier needs from this and a plain function call
cannot give:

- A hang costs one case, not the batch. Nothing in Python can interrupt a
  dispatched Metal command buffer, so the only reliable stop is killing the
  process group. The parent holds a rolling deadline: every result received
  resets it, and when it expires the group dies, the case in flight is
  recorded as a timeout, and a fresh worker resumes from the case after it.
  The same resume happens when a kernel takes the worker down mid-batch: the
  case in flight is recorded as a crash and the rest still run. A death during
  *compilation* is attributed to the spec being compiled - all its cases fail,
  its sibling specs still run - because no case was in flight to blame.
- Partial progress survives. Results stream one line at a time, so nothing a
  later case does can lose an earlier case's measurement.
- Nothing here raises about the candidate. Compile failures, launch failures,
  hangs and crashes all come back as statuses; the only resume that does not
  happen is when the worker dies before opening the Metal device, which means
  the environment, not the candidate, is broken.
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
    """One result per case, in the order the cases were given.

    `compile_options` is what the worker reported actually building the shader
    with; a certificate records it as an input alongside the device.
    """

    results: list = field(default_factory=list)
    device: DeviceInfo | None = None
    stderr: str = ""
    compile_options: dict = field(default_factory=dict)

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
class _Session:
    """What one worker process reported before it finished, died, or hung."""

    saw_device: bool = False
    device: DeviceInfo | None = None
    fatal: RunResult | None = None
    done: bool = False
    timed_out: bool = False
    spawn_error: str = ""
    #: runnable positions whose pipeline was built in this session
    compiled: set = field(default_factory=set)
    compile_options: dict = field(default_factory=dict)
    stderr: str = ""


@dataclass
class MetalRunner:
    """Runs candidate kernels on the GPU through an isolated worker process.

    `case_timeout` is the wall budget for one case including its warmup and
    timed repeats. `startup_timeout` covers the slower events: the worker's
    Python and Metal start-up, and each shader compilation.
    """

    case_timeout: float = DEFAULT_CASE_TIMEOUT
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT
    warmup: int = DEFAULT_WARMUP
    repeats: int = DEFAULT_REPEATS
    python: str = sys.executable
    #: Metal math mode for shader compilation: "safe", "relaxed", or "fast".
    #: The default mirrors MLX's documented default (Metal's own is fast), so
    #: candidates are judged under the same math regime the framework runs.
    math_mode: str = "safe"

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
        return self.run_candidate([(spec, cases)], warmup=warmup, repeats=repeats)[0]

    def run_candidate(self, spec_batches, *, warmup: int | None = None,
                      repeats: int | None = None) -> list[BatchResult]:
        """Run one candidate's specs, in order, through one worker session.

        `spec_batches` is a sequence of `(spec, cases)` pairs; the return is
        one `BatchResult` per pair, in the same order. One worker process
        serves the whole candidate unless a hang or crash forces a resume,
        and a resume never costs more than the case or compile it blames.
        """
        batches = [(spec, list(cases)) for spec, cases in spec_batches]
        slots: list[list[RunResult | None]] = [[None] * len(cases) for _, cases in batches]

        # An unusable pair is caught before paying for a process, and it fails
        # alone: the sibling specs of the candidate still run.
        runnable: list[int] = []
        for position, (spec, cases) in enumerate(batches):
            problem = _validate(spec, cases)
            if problem is not None:
                slots[position] = [
                    RunResult(status=RunStatus.INVALID_SPEC, detail=problem, label=c.label)
                    for c in cases
                ]
            elif cases:
                runnable.append(position)

        device: DeviceInfo | None = None
        compile_options: dict = {}
        stderr_parts: list[str] = []
        spec_cursor, case_cursor = 0, 0
        while spec_cursor < len(runnable):
            session = self._session(batches, runnable, spec_cursor, case_cursor,
                                    slots, warmup, repeats)
            if session.device is not None and device is None:
                device = session.device
            if session.compile_options and not compile_options:
                compile_options = session.compile_options
            if session.stderr:
                stderr_parts.append(session.stderr)

            if session.done:
                break
            if session.spawn_error:
                _fill_remaining(batches, runnable, spec_cursor, case_cursor, slots,
                                RunStatus.CRASH,
                                f"the worker would not start: {session.spawn_error}")
                break
            if session.fatal is not None:
                _fill_remaining(batches, runnable, spec_cursor, case_cursor, slots,
                                session.fatal.status, session.fatal.detail)
                break
            if not session.saw_device:
                # The environment, not the candidate, is broken; resuming
                # would spawn one doomed worker per case.
                status, detail = _no_device_verdict(session)
                _fill_remaining(batches, runnable, spec_cursor, case_cursor, slots,
                                status, detail)
                break

            hole = _first_hole(batches, runnable, slots, spec_cursor, case_cursor)
            if hole is None:
                break  # everything reported; only the done line was lost
            hole_spec, hole_case = hole
            position = runnable[hole_spec]
            cases = batches[position][1]
            if hole_spec in session.compiled:
                # A case was in flight; it alone takes the blame.
                status, detail = _case_verdict(session)
                slots[position][hole_case] = RunResult(
                    status=status, detail=detail, label=cases[hole_case].label)
                spec_cursor, case_cursor = hole_spec, hole_case + 1
            else:
                # The death happened compiling this spec; no case can be blamed,
                # so the spec fails whole and its siblings resume.
                status, detail = _compile_verdict(session)
                for index in range(hole_case, len(cases)):
                    if slots[position][index] is None:
                        slots[position][index] = RunResult(
                            status=status, detail=detail, label=cases[index].label)
                spec_cursor, case_cursor = hole_spec + 1, 0
            if (spec_cursor < len(runnable)
                    and case_cursor >= len(batches[runnable[spec_cursor]][1])):
                spec_cursor, case_cursor = spec_cursor + 1, 0

        stderr = "\n".join(part for part in stderr_parts if part).strip()
        batch_results = []
        for position, (spec, cases) in enumerate(batches):
            results = slots[position]
            for index, result in enumerate(results):
                if result is None:
                    results[index] = RunResult(
                        status=RunStatus.CRASH,
                        detail="the worker never reported this case",
                        label=cases[index].label)
            batch_results.append(BatchResult(results=results, device=device, stderr=stderr,
                                             compile_options=dict(compile_options)))
        return batch_results

    # -- one worker session ------------------------------------------------
    def _session(self, batches, runnable, spec_cursor, case_cursor, slots,
                 warmup, repeats) -> _Session:
        """Spawn one worker for the remaining work and stream its events into
        `slots`. Returns what the parent needs to decide whether and where to
        resume."""
        outcome = _Session()

        request_specs = []
        mapping: list[tuple[int, int]] = []  # request index -> (runnable pos, case offset)
        for rpos in range(spec_cursor, len(runnable)):
            spec, cases = batches[runnable[rpos]]
            offset = case_cursor if rpos == spec_cursor else 0
            request_specs.append({
                "spec": spec.to_json(),
                "cases": [c.to_json() for c in cases[offset:]],
            })
            mapping.append((rpos, offset))
        request = {
            "specs": request_specs,
            "warmup": self.warmup if warmup is None else warmup,
            "repeats": self.repeats if repeats is None else repeats,
            "math_mode": self.math_mode,
        }

        try:
            process = self._spawn()
        except OSError as error:  # no interpreter, no fork: nothing ran
            outcome.spawn_error = str(error)
            return outcome

        errors: list[str] = []
        reader = _stream(process.stderr, errors)
        writer = threading.Thread(
            target=_write_request, args=(process, request), daemon=True
        )
        writer.start()

        events = _EventStream(process.stdout)
        # Starting Python, opening the device and compiling a shader all take
        # an unpredictable but legitimate amount of time, so they run on the
        # start-up budget. The worker announces the end of each compilation,
        # and from there every case is pure dispatch on the per-case budget;
        # the budget returns to start-up size whenever the next event is
        # another spec's compilation.
        budget = self.startup_timeout
        try:
            while True:
                event = events.next_event(budget)
                if event is None:  # the worker closed its pipe
                    break
                kind = event.get("event")
                if kind == "device":
                    outcome.saw_device = True
                    outcome.device = DeviceInfo.from_json(event["device"])
                elif kind == "compiled":
                    entry = _mapped(event, mapping)
                    if entry is not None:
                        request_index, (rpos, offset) = entry
                        outcome.compiled.add(rpos)
                        if isinstance(event.get("compile_options"), dict):
                            outcome.compile_options = event["compile_options"]
                        remaining = len(batches[runnable[rpos]][1]) - offset
                        budget = self.case_timeout if remaining else self.startup_timeout
                elif kind == "spec_failed":
                    # The spec cannot run at all - a compile error or an
                    # unreadable spec - which fails every case it owns and is
                    # final: there is nothing to resume for this spec.
                    entry = _mapped(event, mapping)
                    try:
                        result = RunResult.from_json(event["result"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if entry is not None:
                        request_index, (rpos, offset) = entry
                        outcome.compiled.discard(rpos)
                        position = runnable[rpos]
                        cases = batches[position][1]
                        for index in range(offset, len(cases)):
                            slots[position][index] = RunResult(
                                status=result.status, detail=result.detail,
                                label=cases[index].label)
                    budget = self.startup_timeout  # the next spec compiles now
                elif kind == "case":
                    # A malformed or out-of-range event is treated as if the
                    # case never reported, which is what it means: the stream
                    # is damaged, so the case fills in as a crash at the end
                    # rather than this loop raising inside a battery run.
                    entry = _mapped(event, mapping)
                    try:
                        index = int(event["index"])
                        result = RunResult.from_json(event["result"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    budget = self.case_timeout
                    if entry is not None:
                        request_index, (rpos, offset) = entry
                        position = runnable[rpos]
                        cases = batches[position][1]
                        absolute = offset + index
                        if 0 <= absolute < len(cases):
                            result.label = result.label or cases[absolute].label
                            slots[position][absolute] = result
                        if absolute == len(cases) - 1 and request_index < len(mapping) - 1:
                            budget = self.startup_timeout  # the next spec compiles now
                elif kind == "fatal":
                    outcome.fatal = RunResult.from_json(event["result"])
                    break
                elif kind == "done":
                    outcome.done = True
                    break
        except TimeoutError:
            outcome.timed_out = True
        finally:
            self._terminate(process)
            reader.join(timeout=1.0)

        outcome.stderr = "".join(errors).strip()
        return outcome

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


def _mapped(event: dict, mapping: list) -> tuple[int, tuple[int, int]] | None:
    """The event's request spec index resolved through the resume mapping."""
    try:
        request_index = int(event["spec"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= request_index < len(mapping):
        return None
    return request_index, mapping[request_index]


def _first_hole(batches, runnable, slots, spec_cursor, case_cursor):
    """The earliest case at or after the cursor that never reported."""
    for rpos in range(spec_cursor, len(runnable)):
        position = runnable[rpos]
        start = case_cursor if rpos == spec_cursor else 0
        for index in range(start, len(batches[position][1])):
            if slots[position][index] is None:
                return rpos, index
    return None


def _fill_remaining(batches, runnable, spec_cursor, case_cursor, slots,
                    status: RunStatus, detail: str) -> None:
    for rpos in range(spec_cursor, len(runnable)):
        position = runnable[rpos]
        cases = batches[position][1]
        start = case_cursor if rpos == spec_cursor else 0
        for index in range(start, len(cases)):
            if slots[position][index] is None:
                slots[position][index] = RunResult(status=status, detail=detail,
                                                   label=cases[index].label)


def _tail(stderr: str) -> str:
    lines = stderr.strip().splitlines()[-4:]
    return f": {' / '.join(lines)}" if lines else ""


def _case_verdict(session: _Session) -> tuple[RunStatus, str]:
    """What the case in flight should be called when its worker stopped."""
    if session.timed_out:
        return (RunStatus.TIMEOUT,
                "the worker was killed after exceeding its time budget")
    return (RunStatus.CRASH,
            "the worker exited before reporting this case" + _tail(session.stderr))


def _compile_verdict(session: _Session) -> tuple[RunStatus, str]:
    if session.timed_out:
        return (RunStatus.TIMEOUT,
                "the worker was killed while compiling this kernel")
    return (RunStatus.CRASH,
            "the worker died while compiling this kernel" + _tail(session.stderr))


def _no_device_verdict(session: _Session) -> tuple[RunStatus, str]:
    if session.timed_out:
        return (RunStatus.TIMEOUT,
                "the worker produced nothing before its start-up budget expired")
    return (RunStatus.CRASH,
            "the worker exited before opening the Metal device" + _tail(session.stderr))


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
