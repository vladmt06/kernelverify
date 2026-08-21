"""Machine state, and the rules that decide whether a number may bind.

A baseline is only worth what its measurement conditions are worth, so both
conditions this project has been burned by are checked here rather than
remembered:

  idle          a shared machine depressed the first-pass numbers, so every
                run samples load before and after and marks the rows it
                produced as non-binding when the machine was busy, on battery,
                in low power mode, or thermally warned. Load average alone is
                not enough: the kernels lane saw 2.2x-9.7x spreads on an AC
                machine at load 1.79, caused by interactive apps driving the
                display stack, so busy processes are named individually.

  timing floor  kv-runner-e9 measured that GPU timings around 200 us move
                together by up to 4x with power state. Anything measured below
                a millisecond is therefore reporting the power manager, not
                the kernel, and is refused as an absolute claim.

Neither rule blocks a run. Both refuse to let the result be called binding,
which is the distinction the per-chip matrix needs.

The third rule DOES block, and lives here for the same reason: it is a
property of the machine, not of any one harness.

  measurement lock   one heavy measurement at a time on this machine. On
                     2026-08-15 at 03:29 the serving calibration (39.5 GB
                     phys_footprint) and the pricing probe (27.9 GB) ran
                     together on a 36 GB machine and Jetsam killed the pair.
                     Each harness locking its own file would have permitted
                     exactly that, so there is ONE lock and every heavy
                     harness takes it.
"""

from __future__ import annotations

import fcntl
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path

# Load average is per-runnable-thread, so the threshold scales with cores. An
# idle 12-core Mac sits near 1.5 to 2.5 with background daemons; a quarter of
# the cores busy is the line between that and real work.
IDLE_LOAD_FRACTION = 0.25

# Below this, a single sample is measuring the power manager (see module docstring).
TIMING_FLOOR_MS = 1.0

# A run whose own repeats disagree by more than this is not a measurement,
# whatever the machine said about itself. Added after a run in which load
# average read 1.93 throughout while one spec's third sample came in at 22.07
# against 99.38 and 90.06: load is averaged over a minute and says nothing
# about a transient that lands inside one sample. The samples are the direct
# evidence and they were being ignored.
MAX_SPREAD_PCT = 10.0


# One lock, one machine, every heavy measurement harness. /tmp is shared
# across worktrees and cleared at boot, which is exactly the scope a lock
# wants; the file is never deleted, because unlinking it would let a second
# process create a fresh inode and lock that instead of this one.
MEASUREMENT_LOCK_PATH = Path("/tmp/kernelverify.measurement.lock")


class MeasurementLock:
    """The machine-wide measurement lock, held for the life of one run.

    fcntl.flock on a fixed path, deliberately, rather than a pid file. The
    kernel owns the lock: it releases on process death however the process
    died, which is the case that matters here - the harnesses this guards
    were killed by Jetsam, not shut down. That deletes the two defects a pid
    file has, both of which shipped in the first version of this lock:

    - staleness has to be JUDGED (read the pid, ask whether it is alive), and
      a pid that cannot be signalled, was recycled, or was never written
      reads as dead;
    - creating the file and writing the pid into it are two steps, so a
      reader landing between them sees an empty file, fails to parse it,
      concludes the holder is stale, and steals the lock from a LIVE run.

    The file's contents are diagnostics only - who holds it, for a refusal
    message a human can act on. Nothing decides anything by reading them.
    """

    def __init__(self, owner: str, path: Path = MEASUREMENT_LOCK_PATH):
        self.owner = owner
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> tuple[bool, str]:
        """``(acquired, detail)``. A refusal names the holder when it can and
        modifies nothing either way."""
        if self._fd is not None:
            return True, f"already held by pid {os.getpid()} ({self.owner})"
        fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = self._holder_note()
            os.close(fd)
            return False, holder
        # Stored HERE, before the note is written. The lock is owned the
        # moment `flock` returns, and until 2026-08-21 the descriptor was kept
        # only once truncate, write and fsync had all succeeded, so a failure
        # in any of them left this process holding the machine-wide lock with
        # no object able to release it. The note is diagnostics; the flock is
        # the lock.
        self._fd = fd
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()} ({self.owner})".encode())
        os.fsync(fd)
        return True, f"acquired by pid {os.getpid()} ({self.owner})"

    def release(self) -> None:
        """Release what this object holds, and nothing else: an object that
        never acquired cannot free another process's lock."""
        if self._fd is None:
            return
        os.ftruncate(self._fd, 0)
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    def _holder_note(self) -> str:
        try:
            note = self.path.read_text().strip()
        except OSError:
            note = ""
        return f"held by {note}" if note else "held by another process"


def _sysctl(key: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", key], capture_output=True, text=True
    ).stdout.strip()


def load_averages() -> tuple[float, float, float]:
    raw = _sysctl("vm.loadavg")  # "{ 2.11 2.30 2.42 }"
    nums = [float(x) for x in re.findall(r"[\d.]+", raw)]
    return tuple(nums[:3]) if len(nums) >= 3 else (float("nan"),) * 3


# A process using this much CPU is assumed to be driving the display stack or
# competing for the GPU. Measured cause: the kernels lane saw 2.2x-9.7x spreads
# on an AC-powered machine at load average 1.79, with Terminal at 36% CPU and
# VS Code at 25% sharing the GPU. Load average and power state both passed.
BUSY_PROCESS_PCT = 15.0


def _own_process_tree() -> set[int]:
    """Our own pid and every descendant, so the harness does not flag itself."""
    import os

    ps = subprocess.run(
        ["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True
    ).stdout
    children: dict[int, list[int]] = {}
    for line in ps.splitlines():
        parts = line.split()
        if len(parts) == 2:
            pid, ppid = int(parts[0]), int(parts[1])
            children.setdefault(ppid, []).append(pid)

    mine, stack = set(), [os.getpid()]
    while stack:
        pid = stack.pop()
        if pid in mine:
            continue
        mine.add(pid)
        stack.extend(children.get(pid, []))
    return mine


def competing_processes(threshold_pct: float = BUSY_PROCESS_PCT) -> list[dict]:
    """Processes busy enough to be competing for the GPU, excluding our own.

    There is no unprivileged way to read GPU utilisation on macOS
    (powermetrics needs sudo), so this uses CPU as the observable proxy for
    the display stack being active. It catches the case load average misses,
    which is one or two interactive apps driving WindowServer hard while the
    one-minute average still reads low.
    """
    ours = _own_process_tree()
    ps = subprocess.run(
        ["ps", "-Ao", "pid=,pcpu=,comm="], capture_output=True, text=True
    ).stdout
    busy = []
    for line in ps.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid, pcpu, comm = int(parts[0]), float(parts[1]), parts[2].strip()
        if pid in ours or pcpu < threshold_pct:
            continue
        busy.append({"name": comm.rsplit("/", 1)[-1], "pcpu": pcpu})
    return sorted(busy, key=lambda p: -p["pcpu"])


def power_state() -> dict:
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    everything = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    lpm = next(
        (ln.split()[-1] for ln in everything.splitlines() if "lowpowermode" in ln), "?"
    )
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    return {
        "source": "AC" if "AC Power" in batt else "battery",
        "low_power_mode": lpm,
        "thermal_warning": "No thermal warning level" not in therm,
    }


def fingerprint() -> dict:
    """Everything a row needs to be attributable to a machine and a toolchain."""
    return {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "hw_model": _sysctl("hw.model"),
        "cores": int(_sysctl("hw.ncpu") or 0),
        "performance_cores": int(_sysctl("hw.perflevel0.physicalcpu") or 0),
        "efficiency_cores": int(_sysctl("hw.perflevel1.physicalcpu") or 0),
        "memory_bytes": int(_sysctl("hw.memsize") or 0),
        "os": f"{platform.system()} {platform.mac_ver()[0]}",
        "os_build": subprocess.run(
            ["sw_vers", "-buildVersion"], capture_output=True, text=True
        ).stdout.strip(),
    }


def idle_check(cores: int | None = None) -> dict:
    """Sample the machine and say whether measurements taken now may bind."""
    if cores is None:
        cores = int(_sysctl("hw.ncpu") or 1)
    load1, load5, load15 = load_averages()
    power = power_state()

    threshold = IDLE_LOAD_FRACTION * cores
    competitors = competing_processes()
    blockers = []
    if load1 > threshold:
        blockers.append(f"load1 {load1:.2f} over threshold {threshold:.2f}")
    for p in competitors:
        blockers.append(f"{p['name']} at {p['pcpu']:.0f}% CPU competing for the GPU")
    if power["source"] != "AC":
        blockers.append(f"power source {power['source']}")
    if power["low_power_mode"] not in ("0", "?"):
        blockers.append("low power mode on")
    if power["thermal_warning"]:
        blockers.append("thermal warning recorded")

    return {
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "load_threshold": round(threshold, 2),
        "power": power,
        "competing_processes": competitors,
        "idle": not blockers,
        "blockers": blockers,
    }


def timing_verdict(min_sample_ms: float) -> dict:
    """Whether a measurement is above the scale where timings mean anything."""
    below = min_sample_ms < TIMING_FLOOR_MS
    return {
        "min_sample_ms": round(min_sample_ms, 4),
        "floor_ms": TIMING_FLOOR_MS,
        "below_timing_floor": below,
    }


def spread_pct(vals) -> float:
    """Repeat disagreement as (max - min) / median, in percent: the exact
    quantity MAX_SPREAD_PCT bounds, computed in one place."""
    return (max(vals) - min(vals)) / statistics.median(vals) * 100


def dispersion_verdict(spread_pct: float | None) -> dict:
    """Whether the repeats agreed well enough for the median to mean anything."""
    over = spread_pct is not None and spread_pct > MAX_SPREAD_PCT
    return {
        "spread_pct": spread_pct,
        "max_spread_pct": MAX_SPREAD_PCT,
        "over_spread_limit": over,
    }


def binding_verdict(before: dict, after: dict, timing: dict,
                    dispersion: dict | None = None) -> dict:
    """A row binds only if the machine was clean, the scale is real, and the
    repeats agree. The third condition is not implied by the first two: a
    transient can land inside one sample without moving a one-minute load
    average at all."""
    blockers = []
    blockers += [f"before: {b}" for b in before["blockers"]]
    blockers += [f"after: {b}" for b in after["blockers"]]
    if timing["below_timing_floor"]:
        blockers.append(
            f"sample {timing['min_sample_ms']} ms under the "
            f"{timing['floor_ms']} ms timing floor"
        )
    if dispersion and dispersion["over_spread_limit"]:
        blockers.append(
            f"repeats disagree by {dispersion['spread_pct']}%, over the "
            f"{dispersion['max_spread_pct']}% limit"
        )
    return {"binding": not blockers, "binding_blockers": blockers}
