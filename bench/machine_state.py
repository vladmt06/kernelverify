"""Machine state, and the two rules that decide whether a number may bind.

A baseline is only worth what its measurement conditions are worth, so both
conditions this project has been burned by are checked here rather than
remembered:

  idle          a shared machine depressed the first-pass numbers, so every
                run samples load before and after and marks the rows it
                produced as non-binding when the machine was busy, on battery,
                in low power mode, or thermally warned.

  timing floor  kv-runner-e9 measured that GPU timings around 200 us move
                together by up to 4x with power state. Anything measured below
                a millisecond is therefore reporting the power manager, not
                the kernel, and is refused as an absolute claim.

Neither rule blocks a run. Both refuse to let the result be called binding,
which is the distinction the per-chip matrix needs.
"""

from __future__ import annotations

import platform
import re
import subprocess

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


def _sysctl(key: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", key], capture_output=True, text=True
    ).stdout.strip()


def load_averages() -> tuple[float, float, float]:
    raw = _sysctl("vm.loadavg")  # "{ 2.11 2.30 2.42 }"
    nums = [float(x) for x in re.findall(r"[\d.]+", raw)]
    return tuple(nums[:3]) if len(nums) >= 3 else (float("nan"),) * 3


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
    blockers = []
    if load1 > threshold:
        blockers.append(f"load1 {load1:.2f} over threshold {threshold:.2f}")
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
